import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs
from PDRs import PriorityDispatchingRules
from state_features import StateFeatureExtractor


class SJSSP_PDR(gym.Env, EzPickle):
    """
    Modified JSSP Environment for PDR Selection
    
    Instead of selecting operations directly, the agent selects a PDR,
    which then deterministically selects an operation.
    """
    
    def __init__(self, n_j, n_m):
        EzPickle.__init__(self)
        
        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        
        # Task IDs for first and last columns
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        
        # Initialize PDR system
        self.pdrs = PriorityDispatchingRules(n_j, n_m)
        self.num_pdrs = self.pdrs.num_rules
        
        # Initialize feature extractor
        self.feature_extractor = StateFeatureExtractor(n_j, n_m)
        self.feature_dim = self.feature_extractor.feature_dim
        
        # Utility functions
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        
        # For tracking PDR usage statistics
        self.pdr_usage_count = np.zeros(self.num_pdrs, dtype=np.int32)
    
    def done(self):
        """Check if all operations are scheduled"""
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False
    
    @override
    def step(self, pdr_action):
        """
        Step function - agent selects a PDR, PDR selects an operation
        
        Args:
            pdr_action: Index of PDR to apply (0 to num_pdrs-1)
        
        Returns:
            state_features: Feature vector for next state
            reward: Reward for this transition
            done: Whether episode is complete
        """
        # Get eligible operations
        eligible_ops = self.omega[~self.mask]
        
        if len(eligible_ops) == 0:
            # Should not happen, but handle gracefully
            raise ValueError("No eligible operations available!")
        
        # Apply selected PDR to choose operation
        operation = self.pdrs.apply_rule(
            rule_idx=pdr_action,
            eligible_ops=eligible_ops,
            dur=self.dur,
            finished_mark=self.finished_mark,
            m=self.m,
            opIDsOnMchs=self.opIDsOnMchs
        )
        
        # Track PDR usage
        self.pdr_usage_count[pdr_action] += 1
        
        # Execute the selected operation (same as original code)
        if operation not in self.partial_sol_sequeence:
            # UPDATE BASIC INFO
            row = operation // self.number_of_machines
            col = operation % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(operation)
            
            # UPDATE STATE
            startTime_a, flag = permissibleLeftShift(
                a=operation,
                durMat=self.dur,
                mchMat=self.m,
                mchsStartTimes=self.mchsStartTimes,
                opIDsOnMchs=self.opIDsOnMchs
            )
            self.flags.append(flag)
            
            # Update omega or mask
            if operation not in self.last_col:
                self.omega[operation // self.number_of_machines] += 1
            else:
                self.mask[operation // self.number_of_machines] = 1
            
            self.temp1[row, col] = startTime_a + dur_a
            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)
            
            # Update adjacency matrix
            precd, succd = self.getNghbs(operation, self.opIDsOnMchs)
            self.adj[operation] = 0
            self.adj[operation, operation] = 1
            if operation not in self.first_col:
                self.adj[operation, operation - 1] = 1
            self.adj[operation, precd] = 1
            self.adj[succd, operation] = 1
            if flag and precd != operation and succd != operation:
                self.adj[succd, precd] = 0
        
        # Calculate reward
        reward = -(self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()
        
        # Extract features for next state
        state_features = self.feature_extractor.extract_features(self)
        
        return state_features, reward, self.done()
    
    @override
    def reset(self, data):
        """Reset environment with new problem instance"""
        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        
        # Record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0
        
        # Reset PDR usage tracking
        self.pdr_usage_count = np.zeros(self.num_pdrs, dtype=np.int32)
        
        # Initialize adjacency matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        conj_nei_up_stream[self.first_col] = 0
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream
        
        # Initialize features
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)
        
        # Initialize feasible omega
        self.omega = self.first_col.astype(np.int64)
        
        # Initialize mask
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)
        
        # Start time of operations on machines
        self.mchsStartTimes = -configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)
        
        self.temp1 = np.zeros_like(self.dur, dtype=np.single)
        
        # Extract initial state features
        state_features = self.feature_extractor.extract_features(self)
        
        return state_features
    
    def get_pdr_usage_stats(self):
        """Return statistics on PDR usage during episode"""
        total = self.pdr_usage_count.sum()
        if total == 0:
            return {name: 0.0 for name in self.pdrs.rule_names.values()}
        
        stats = {}
        for idx, name in self.pdrs.rule_names.items():
            stats[name] = self.pdr_usage_count[idx] / total
        return stats


if __name__ == "__main__":
    # Test the modified environment
    from uniform_instance_gen import uni_instance_gen
    import time
    
    n_j = 6
    n_m = 6
    
    env = SJSSP_PDR(n_j=n_j, n_m=n_m)
    print(f"Environment created: {n_j}x{n_m}")
    print(f"Feature dimension: {env.feature_dim}")
    print(f"Number of PDRs: {env.num_pdrs}")
    
    # Generate random instance
    np.random.seed(42)
    data = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=99)
    
    # Reset and run random episode
    state = env.reset(data)
    print(f"\nInitial state shape: {state.shape}")
    print(f"Initial quality (makespan LB): {env.initQuality}")
    
    rewards = [-env.initQuality]
    t1 = time.time()
    
    while not env.done():
        # Random PDR selection
        pdr_action = np.random.randint(0, env.num_pdrs)
        state, reward, done = env.step(pdr_action)
        rewards.append(reward)
    
    t2 = time.time()
    
    makespan = sum(rewards) - env.posRewards
    print(f"\nEpisode completed in {t2-t1:.4f} seconds")
    print(f"Final makespan: {makespan}")
    print(f"Steps taken: {env.step_count}")
    
    # Print PDR usage statistics
    print("\nPDR Usage Statistics:")
    stats = env.get_pdr_usage_stats()
    for pdr_name, usage in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pdr_name}: {usage*100:.1f}%")