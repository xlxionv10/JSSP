import torch
import argparse
import numpy as np
import time
from Params import configs

device = torch.device(configs.device)

parser = argparse.ArgumentParser(description='Test learned PDR policy')
parser.add_argument('--Pn_j', type=int, default=15, help='Number of jobs to test')
parser.add_argument('--Pn_m', type=int, default=15, help='Number of machines to test')
parser.add_argument('--Nn_j', type=int, default=15, help='Number of jobs model was trained on')
parser.add_argument('--Nn_m', type=int, default=15, help='Number of machines model was trained on')
parser.add_argument('--low', type=int, default=1, help='LB of duration')
parser.add_argument('--high', type=int, default=99, help='UB of duration')
parser.add_argument('--seed', type=int, default=200, help='Seed for test set generation')
parser.add_argument('--greedy', action='store_true', help='Use greedy action selection')
params = parser.parse_args()

N_JOBS_P = params.Pn_j
N_MACHINES_P = params.Pn_m
LOW = params.low
HIGH = params.high
SEED = params.seed
N_JOBS_N = params.Nn_j
N_MACHINES_N = params.Nn_m
GREEDY = params.greedy


def test(dataset, policy, env, greedy=True):
    """
    Test policy on dataset
    
    Args:
        dataset: List of problem instances
        policy: Trained policy network
        env: JSSP environment
        greedy: Whether to use greedy action selection
    
    Returns:
        results: List of makespans
        pdr_stats: Statistics on PDR usage
    """
    results = []
    all_pdr_stats = []
    
    t1 = time.time()
    
    for i, data in enumerate(dataset):
        state = env.reset(data)
        ep_reward = -env.max_endTime
        
        while not env.done():
            state_tensor = torch.from_numpy(state).float().to(device)
            
            with torch.no_grad():
                pi, _ = policy(state_tensor)
            
            if greedy:
                # Greedy selection
                action = torch.argmax(pi.squeeze()).item()
            else:
                # Sample from distribution
                from torch.distributions.categorical import Categorical
                dist = Categorical(pi.squeeze())
                action = dist.sample().item()
            
            state, reward, done = env.step(action)
            ep_reward += reward
        
        makespan = -ep_reward + env.posRewards
        results.append(makespan)
        
        # Get PDR usage statistics
        pdr_stats = env.get_pdr_usage_stats()
        all_pdr_stats.append(pdr_stats)
        
        print(f'Instance {i + 1:3d} | Makespan: {makespan:7.2f}')
    
    t2 = time.time()
    
    print(f'\n{"="*60}')
    print(f'Total time: {t2 - t1:.2f}s')
    print(f'Average time per instance: {(t2 - t1) / len(dataset):.4f}s')
    print(f'Average makespan: {np.mean(results):.2f}')
    print(f'Std makespan: {np.std(results):.2f}')
    print(f'Min makespan: {np.min(results):.2f}')
    print(f'Max makespan: {np.max(results):.2f}')
    
    # Aggregate PDR statistics
    print(f'\n{"="*60}')
    print('PDR Usage Statistics (averaged over all instances):')
    print(f'{"="*60}')
    
    avg_pdr_stats = {}
    for pdr_name in all_pdr_stats[0].keys():
        avg_pdr_stats[pdr_name] = np.mean([stats[pdr_name] for stats in all_pdr_stats])
    
    for pdr_name, usage in sorted(avg_pdr_stats.items(), key=lambda x: x[1], reverse=True):
        print(f'  {pdr_name:10s}: {usage * 100:5.1f}%')
    
    return results, all_pdr_stats


def main():
    from JSSP_Env_PDR import SJSSP_PDR
    from models.actor_critic_pdr import ActorCriticPDR
    
    # Create environment
    env = SJSSP_PDR(n_j=N_JOBS_P, n_m=N_MACHINES_P)
    
    print(f"Testing PDR Policy")
    print(f"Problem size: {N_JOBS_P}x{N_MACHINES_P}")
    print(f"Model trained on: {N_JOBS_N}x{N_MACHINES_N}")
    print(f"Feature dimension: {env.feature_dim}")
    print(f"Number of PDRs: {env.num_pdrs}")
    print(f"Action selection: {'Greedy' if GREEDY else 'Stochastic'}")
    print(f"Device: {device}")
    print(f'{"="*60}\n')
    
    # Create and load policy
    policy = ActorCriticPDR(
        feature_dim=env.feature_dim,
        num_pdrs=env.num_pdrs,
        num_mlp_layers_actor=configs.num_mlp_layers_actor,
        hidden_dim_actor=configs.hidden_dim_actor,
        num_mlp_layers_critic=configs.num_mlp_layers_critic,
        hidden_dim_critic=configs.hidden_dim_critic,
        device=device
    ).to(device)
    
    # Load trained weights
    path = f'./SavedNetwork/{N_JOBS_N}_{N_MACHINES_N}_{LOW}_{HIGH}_PDR.pth'
    try:
        policy.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from: {path}\n")
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}")
        print("Please train the model first using PPO_jssp_PDR.py")
        return
    
    policy.eval()
    
    # Load test data
    dataLoaded = np.load(f'./DataGen/generatedData{N_JOBS_P}_{N_MACHINES_P}_Seed{SEED}.npy')
    dataset = [(dataLoaded[i][0], dataLoaded[i][1]) for i in range(dataLoaded.shape[0])]
    
    print(f"Test set size: {len(dataset)}")
    print(f'{"="*60}\n')
    
    # Test policy
    results, pdr_stats = test(dataset, policy, env, greedy=GREEDY)
    
    # Save results
    result_filename = f'drlResult_PDR_{N_JOBS_N}x{N_MACHINES_N}_{N_JOBS_P}x{N_MACHINES_P}_Seed{SEED}.npy'
    np.save(result_filename, np.array(results, dtype=np.single))
    print(f'\nResults saved to: {result_filename}')
    
    # Save timing
    time_filename = f'drltime_PDR_{N_JOBS_N}x{N_MACHINES_N}_{N_JOBS_P}x{N_MACHINES_P}.txt'
    # This would be populated during the test() function
    
    print(f'\n{"="*60}')
    print('Testing complete!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()