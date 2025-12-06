from copy import deepcopy
import torch
import time
import torch.nn as nn
import numpy as np
from Params_PDR import configs
from torch.distributions.categorical import Categorical

device = torch.device(configs.device)


class Memory:
    """Memory buffer for storing episode data"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
    
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.logprobs[:]


def select_action(pi, memory=None):
    """
    Select PDR action from policy distribution
    
    Args:
        pi: Probability distribution over PDRs
        memory: Memory buffer to store logprob (optional)
    
    Returns:
        action: Selected PDR index
    """
    dist = Categorical(pi.squeeze())
    action = dist.sample()
    
    if memory is not None:
        memory.logprobs.append(dist.log_prob(action))
    
    return action.item()


def eval_actions(pi, actions):
    """
    Evaluate actions under current policy
    
    Args:
        pi: Probability distribution over PDRs
        actions: Taken actions
    
    Returns:
        logprobs: Log probabilities of actions
        entropy: Policy entropy
    """
    dist = Categorical(pi)
    logprobs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    return logprobs, entropy


class PPO_PDR:
    """PPO algorithm for PDR selection"""
    
    def __init__(self,
                 feature_dim,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 num_pdrs=10,
                 num_mlp_layers_actor=3,
                 hidden_dim_actor=64,
                 num_mlp_layers_critic=3,
                 hidden_dim_critic=64):
        
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Import here to avoid circular dependency
        from models.actor_critic_pdr import ActorCriticPDR
        
        # Create policy networks
        self.policy = ActorCriticPDR(
            feature_dim=feature_dim,
            num_pdrs=num_pdrs,
            num_mlp_layers_actor=num_mlp_layers_actor,
            hidden_dim_actor=hidden_dim_actor,
            num_mlp_layers_critic=num_mlp_layers_critic,
            hidden_dim_critic=hidden_dim_critic,
            device=device
        ).to(device)
        
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=configs.decay_step_size,
            gamma=configs.decay_ratio
        )
        
        # Loss function
        self.V_loss_fn = nn.MSELoss()
    
    def update(self, memories):
        """
        Update policy using collected experience
        
        Args:
            memories: List of Memory objects from parallel environments
        
        Returns:
            loss: Average total loss
            v_loss: Average value loss
        """
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef
        
        # Process memories from all environments
        all_states = []
        all_actions = []
        all_rewards = []
        all_old_logprobs = []
        
        for memory in memories:
            # Compute discounted rewards
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.dones)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            
            # Normalize rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            # Stack states and convert to tensor
            states = torch.stack([torch.from_numpy(s).float() for s in memory.states]).to(device)
            actions = torch.tensor(memory.actions, dtype=torch.long).to(device)
            old_logprobs = torch.stack(memory.logprobs).to(device).detach()
            
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_old_logprobs.append(old_logprobs)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            total_loss = 0
            total_v_loss = 0
            
            for states, actions, rewards, old_logprobs in zip(
                all_states, all_actions, all_rewards, all_old_logprobs
            ):
                # Evaluate actions with current policy
                pi, values = self.policy(states)
                logprobs, entropy = eval_actions(pi, actions)
                
                # Calculate advantages
                advantages = rewards - values.squeeze().detach()
                
                # Policy loss (PPO clipped objective)
                ratios = torch.exp(logprobs - old_logprobs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = self.V_loss_fn(values.squeeze(), rewards)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy
                
                # Total loss
                loss = (vloss_coef * value_loss + 
                       ploss_coef * policy_loss + 
                       entloss_coef * entropy_loss)
                
                total_loss += loss
                total_v_loss += value_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Learning rate decay
        if configs.decayflag:
            self.scheduler.step()
        
        return total_loss.item() / len(memories), total_v_loss.item() / len(memories)


def validate(vali_data, policy, n_j, n_m):
    """
    Validate policy on validation set
    
    Args:
        vali_data: List of problem instances
        policy: Policy network
        n_j: Number of jobs
        n_m: Number of machines
    
    Returns:
        makespans: Array of makespans for validation instances
    """
    from JSSP_Env_PDR import SJSSP_PDR
    
    env = SJSSP_PDR(n_j=n_j, n_m=n_m)
    makespans = []
    
    for data in vali_data:
        state = env.reset(data)
        ep_reward = -env.initQuality
        
        while not env.done():
            state_tensor = torch.from_numpy(state).float().to(device)
            
            with torch.no_grad():
                pi, _ = policy(state_tensor)
            
            # Greedy action selection
            action = torch.argmax(pi.squeeze()).item()
            
            state, reward, done = env.step(action)
            ep_reward += reward
        
        makespan = -ep_reward + env.posRewards
        makespans.append(makespan)
    
    return np.array(makespans)


def main():
    """Main training loop"""
    from JSSP_Env_PDR import SJSSP_PDR
    from uniform_instance_gen import uni_instance_gen
    
    # Create environments
    envs = [SJSSP_PDR(n_j=configs.n_j, n_m=configs.n_m) for _ in range(configs.num_envs)]
    
    # Data generator
    data_generator = uni_instance_gen
    
    # Load validation data
    dataLoaded = np.load('./DataGen/generatedData' + str(configs.n_j) + '_' + 
                        str(configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    vali_data = [(dataLoaded[i][0], dataLoaded[i][1]) for i in range(dataLoaded.shape[0])]
    
    # Set seeds
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)
    
    # Create memories for each environment
    memories = [Memory() for _ in range(configs.num_envs)]
    
    # Create PPO agent
    ppo = PPO_PDR(
        feature_dim=envs[0].feature_dim,
        lr=configs.lr,
        gamma=configs.gamma,
        k_epochs=configs.k_epochs,
        eps_clip=configs.eps_clip,
        num_pdrs=envs[0].num_pdrs,
        num_mlp_layers_actor=configs.num_mlp_layers_actor,
        hidden_dim_actor=configs.hidden_dim_actor,
        num_mlp_layers_critic=configs.num_mlp_layers_critic,
        hidden_dim_critic=configs.hidden_dim_critic
    )
    
    print(f"Training PPO with PDR selection")
    print(f"Problem size: {configs.n_j}x{configs.n_m}")
    print(f"Feature dimension: {envs[0].feature_dim}")
    print(f"Number of PDRs: {envs[0].num_pdrs}")
    print(f"Number of environments: {configs.num_envs}")
    print(f"Device: {device}")
    print("-" * 60)
    
    # Training loop
    log = []
    validation_log = []
    record = float('inf')
    
    for i_update in range(configs.max_updates):
        t_start = time.time()
        
        # Initialize episode rewards
        ep_rewards = [0 for _ in range(configs.num_envs)]
        states = []
        
        # Reset all environments
        for i, env in enumerate(envs):
            state = env.reset(data_generator(n_j=configs.n_j, n_m=configs.n_m, 
                                            low=configs.low, high=configs.high))
            states.append(state)
            ep_rewards[i] = -env.initQuality
        
        # Rollout episodes
        while True:
            # Convert states to tensors
            state_tensors = [torch.from_numpy(s).float().to(device) for s in states]
            
            # Select actions
            with torch.no_grad():
                actions = []
                for i in range(configs.num_envs):
                    pi, _ = ppo.policy_old(state_tensors[i])
                    action = select_action(pi, memories[i])
                    actions.append(action)
            
            # Store current states
            for i in range(configs.num_envs):
                memories[i].states.append(states[i])
                memories[i].actions.append(actions[i])
            
            # Execute actions
            next_states = []
            all_done = True
            
            for i in range(configs.num_envs):
                state, reward, done = envs[i].step(actions[i])
                next_states.append(state)
                ep_rewards[i] += reward
                
                memories[i].rewards.append(reward)
                memories[i].dones.append(done)
                
                if not done:
                    all_done = False
            
            states = next_states
            
            if all_done:
                break
        
        # Adjust rewards for positive rewards
        for i in range(configs.num_envs):
            ep_rewards[i] -= envs[i].posRewards
        
        # Update policy
        loss, v_loss = ppo.update(memories)
        
        # Clear memories
        for memory in memories:
            memory.clear_memory()
        
        # Log results
        mean_reward = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_reward])
        
        t_end = time.time()
        
        # Print progress
        print(f'Episode {i_update + 1:5d} | Reward: {mean_reward:8.2f} | '
              f'V-loss: {v_loss:8.6f} | Time: {t_end - t_start:6.2f}s')
        
        # Validation and checkpointing
        if (i_update + 1) % 100 == 0:
            vali_result = -validate(vali_data, ppo.policy, configs.n_j, configs.n_m).mean()
            validation_log.append(vali_result)
            
            print(f'  Validation makespan: {-vali_result:.2f}')
            
            # Save best model
            if vali_result < record:
                torch.save(ppo.policy.state_dict(), 
                          f'./SavedNetwork/{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}_PDR.pth')
                record = vali_result
                print(f'  *** New best model saved! ***')
            
            # Save logs
            with open(f'./log_{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}_PDR.txt', 'w') as f:
                f.write(str(log))
            
            with open(f'./vali_{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}_PDR.txt', 'w') as f:
                f.write(str(validation_log))
            
            print("-" * 60)


if __name__ == '__main__':
    total_start = time.time()
    main()
    total_end = time.time()
    print(f"\nTotal training time: {(total_end - total_start) / 3600:.2f} hours")