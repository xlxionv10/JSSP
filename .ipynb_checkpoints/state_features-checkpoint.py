import numpy as np


class StateFeatureExtractor:
    """
    Extracts ~155 features from the JSSP environment state for PDR selection
    
    Feature categories:
    - Per-job features (n_j × 5)
    - Per-machine features (n_m × 3)
    - Global statistics (~20)
    - Eligible operation features (~15)
    """
    
    def __init__(self, n_j, n_m):
        self.n_j = n_j
        self.n_m = n_m
        self.feature_dim = self._compute_feature_dim()
    
    def _compute_feature_dim(self):
        """Calculate total number of features"""
        job_features = self.n_j * 5
        machine_features = self.n_m * 3
        global_features = 21
        eligible_features = 15
        return job_features + machine_features + global_features + eligible_features
    
    def extract_features(self, env):
        """
        Extract all features from environment state
        
        Args:
            env: JSSP environment instance
            
        Returns:
            features: numpy array of shape (feature_dim,)
        """
        features = []
        
        # ========== PER-JOB FEATURES ==========
        job_features = self._extract_job_features(env)
        features.extend(job_features)
        
        # ========== PER-MACHINE FEATURES ==========
        machine_features = self._extract_machine_features(env)
        features.extend(machine_features)
        
        # ========== GLOBAL FEATURES ==========
        global_features = self._extract_global_features(env)
        features.extend(global_features)
        
        # ========== ELIGIBLE OPERATION FEATURES ==========
        eligible_features = self._extract_eligible_features(env)
        features.extend(eligible_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_job_features(self, env):
        """
        Extract per-job features (5 features per job)
        
        Features per job:
        1. Total processing time (sum of all operations)
        2. Remaining operations count
        3. Work remaining (sum of unfinished operation times)
        4. Critical ratio (work_remaining / remaining_ops)
        5. Completion ratio (finished_ops / total_ops)
        """
        features = []
        
        for j in range(self.n_j):
            # Total processing time
            total_time = env.dur[j, :].sum()
            
            # Remaining operations
            remaining_ops = (1 - env.finished_mark[j, :]).sum()
            
            # Work remaining
            work_remaining = (env.dur[j, :] * (1 - env.finished_mark[j, :])).sum()
            
            # Critical ratio (average time per remaining operation)
            critical_ratio = work_remaining / (remaining_ops + 1e-8)
            
            # Completion ratio
            completion_ratio = env.finished_mark[j, :].sum() / self.n_m
            
            features.extend([total_time, remaining_ops, work_remaining, 
                           critical_ratio, completion_ratio])
        
        return features
    
    def _extract_machine_features(self, env):
        """
        Extract per-machine features (3 features per machine)
        
        Features per machine:
        1. Queue length (number of operations scheduled)
        2. Total work on machine (sum of durations of scheduled ops)
        3. Is bottleneck (binary indicator)
        """
        features = []
        machine_loads = []
        
        for m in range(self.n_m):
            # Queue length
            queue_length = (env.opIDsOnMchs[m] >= 0).sum()
            
            # Total work on this machine
            ops_on_machine = env.opIDsOnMchs[m][env.opIDsOnMchs[m] >= 0]
            if len(ops_on_machine) > 0:
                total_work = sum(env.dur.flat[op] for op in ops_on_machine)
            else:
                total_work = 0
            
            machine_loads.append(total_work)
            
            # Placeholder for bottleneck indicator (will set after)
            features.extend([queue_length, total_work, 0])
        
        # Identify bottleneck machine (highest total work)
        if max(machine_loads) > 0:
            bottleneck_idx = np.argmax(machine_loads)
            # Set bottleneck flag (3rd feature of bottleneck machine)
            features[bottleneck_idx * 3 + 2] = 1
        
        return features
    
    def _extract_global_features(self, env):
        """
        Extract global statistics (21 features)
        
        Features:
        - Job-level statistics (12): min/max/mean/std of total_time, remaining_ops, work_remaining
        - Machine-level statistics (5): min/max/mean/std of queue_length, load_imbalance
        - Progress features (4): finished_ops_ratio, current_makespan_LB, 
                                 avg_critical_ratio, completion_variance
        """
        features = []
        
        # Compute job-level arrays
        job_total_times = np.array([env.dur[j, :].sum() for j in range(self.n_j)])
        job_remaining_ops = np.array([(1 - env.finished_mark[j, :]).sum() 
                                      for j in range(self.n_j)])
        job_work_remaining = np.array([(env.dur[j, :] * (1 - env.finished_mark[j, :])).sum() 
                                       for j in range(self.n_j)])
        
        # Job-level statistics
        features.extend([
            np.min(job_total_times), np.max(job_total_times),
            np.mean(job_total_times), np.std(job_total_times),
            
            np.min(job_remaining_ops), np.max(job_remaining_ops),
            np.mean(job_remaining_ops), np.std(job_remaining_ops),
            
            np.min(job_work_remaining), np.max(job_work_remaining),
            np.mean(job_work_remaining), np.std(job_work_remaining),
        ])
        
        # Machine-level statistics
        machine_queues = np.array([(env.opIDsOnMchs[m] >= 0).sum() 
                                   for m in range(self.n_m)])
        features.extend([
            np.min(machine_queues), np.max(machine_queues),
            np.mean(machine_queues), np.std(machine_queues),
            np.max(machine_queues) - np.min(machine_queues),  # Load imbalance
        ])
        
        # Progress features
        total_ops = self.n_j * self.n_m
        finished_ops = env.finished_mark.sum()
        finished_ratio = finished_ops / total_ops
        
        # Current makespan lower bound
        current_makespan_lb = env.LBs.max()
        
        # Average critical ratio across jobs
        critical_ratios = job_work_remaining / (job_remaining_ops + 1e-8)
        avg_critical_ratio = np.mean(critical_ratios)
        
        # Completion variance (how evenly are jobs progressing?)
        completion_ratios = np.array([env.finished_mark[j, :].sum() / self.n_m 
                                      for j in range(self.n_j)])
        completion_variance = np.std(completion_ratios)
        
        features.extend([finished_ratio, current_makespan_lb, 
                        avg_critical_ratio, completion_variance])
        
        return features
    
    def _extract_eligible_features(self, env):
        """
        Extract features about eligible operations (15 features)
        
        Features:
        - Eligible operation statistics (5): count, min/max/mean/std of processing times
        - Next operation statistics (10): min/max/mean of next_op_times and next_op_queues,
                                          count of ops with next_op, avg_next_op_time, avg_next_op_queue
        """
        features = []
        
        # Get eligible operations
        eligible_ops = env.omega[~env.mask]
        
        if len(eligible_ops) == 0:
            # Should not happen, but handle gracefully
            return [0] * 15
        
        # Eligible operation processing times
        eligible_times = np.array([env.dur.flat[op] for op in eligible_ops])
        
        features.extend([
            len(eligible_ops),
            np.min(eligible_times), np.max(eligible_times),
            np.mean(eligible_times), np.std(eligible_times),
        ])
        
        # Next operation features (for LQNO/MQNO)
        next_op_times = []
        next_op_queues = []
        
        for op in eligible_ops:
            pos_in_job = op % self.n_m
            
            # Check if there's a next operation
            if pos_in_job < self.n_m - 1:
                next_op = op + 1
                next_op_time = env.dur.flat[next_op]
                next_op_machine = env.m.flat[next_op] - 1  # Convert to 0-indexed
                next_op_queue = (env.opIDsOnMchs[next_op_machine] >= 0).sum()
                
                next_op_times.append(next_op_time)
                next_op_queues.append(next_op_queue)
        
        # Next operation statistics
        if len(next_op_times) > 0:
            features.extend([
                len(next_op_times),  # Count of ops with next operation
                np.min(next_op_times), np.max(next_op_times), np.mean(next_op_times),
                np.min(next_op_queues), np.max(next_op_queues), np.mean(next_op_queues),
                np.mean(next_op_times),  # Duplicate for compatibility
                np.mean(next_op_queues),  # Duplicate for compatibility
                np.std(next_op_times) if len(next_op_times) > 1 else 0,
            ])
        else:
            # All eligible ops are last in their jobs
            features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        return features


if __name__ == "__main__":
    # Test feature extraction
    from JSSP_Env import SJSSP
    from uniform_instance_gen import uni_instance_gen
    
    n_j, n_m = 6, 6
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    # Generate random instance
    data = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=99)
    env.reset(data)
    
    # Extract features
    extractor = StateFeatureExtractor(n_j, n_m)
    features = extractor.extract_features(env)
    
    print(f"Feature dimension: {extractor.feature_dim}")
    print(f"Actual features shape: {features.shape}")
    print(f"Feature sample (first 20): {features[:20]}")
    print(f"\nFeature breakdown:")
    print(f"  Job features: {n_j * 5} = {n_j} × 5")
    print(f"  Machine features: {n_m * 3} = {n_m} × 3")
    print(f"  Global features: 21")
    print(f"  Eligible features: 15")
    print(f"  Total: {n_j * 5 + n_m * 3 + 21 + 15}")