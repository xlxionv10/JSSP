import numpy as np


class PriorityDispatchingRules:
    """
    Collection of Priority Dispatching Rules for JSSP
    Each rule takes the current state and eligible operations,
    and returns the operation with highest priority.
    """
    
    def __init__(self, n_j, n_m):
        self.n_j = n_j
        self.n_m = n_m
        self.num_rules = 10
        
        # Map rule index to function
        self.rules = {
            0: self.SPT,    # Shortest Processing Time
            1: self.LPT,    # Longest Processing Time
            2: self.STPT,   # Shortest Total Processing Time
            3: self.LTPT,   # Longest Total Processing Time
            4: self.LOR,    # Least Operation Remaining
            5: self.MOR,    # Most Operation Remaining
            6: self.LWKR,   # Least Work Remaining
            7: self.MWKR,   # Most Work Remaining
            8: self.LQNO,   # Least Queue Next Operation
            9: self.RANDOM  # Random selection
        }
        
        self.rule_names = {
            0: "SPT", 1: "LPT", 2: "STPT", 3: "LTPT", 4: "LOR",
            5: "MOR", 6: "LWKR", 7: "MWKR", 8: "LQNO", 9: "RANDOM"
        }
    
    def apply_rule(self, rule_idx, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """
        Apply the selected rule to choose an operation
        
        Args:
            rule_idx: Index of the rule to apply (0-9)
            eligible_ops: Array of eligible operation indices
            dur: Duration matrix (n_j, n_m)
            finished_mark: Binary matrix indicating finished operations
            m: Machine assignment matrix (n_j, n_m)
            opIDsOnMchs: Operations scheduled on each machine
        
        Returns:
            selected_op: Index of the selected operation
        """
        return self.rules[rule_idx](eligible_ops, dur, finished_mark, m, opIDsOnMchs)
    
    def _get_processing_times(self, eligible_ops, dur):
        """Get processing times for eligible operations"""
        return np.array([dur.flat[op] for op in eligible_ops])
    
    def _get_job_from_op(self, op):
        """Get job index from operation index"""
        return op // self.n_m
    
    def _get_position_in_job(self, op):
        """Get position of operation within its job"""
        return op % self.n_m
    
    # ========== OPERATION-LEVEL RULES ==========
    
    def SPT(self, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """Shortest Processing Time - select operation with minimum duration"""
        processing_times = self._get_processing_times(eligible_ops, dur)
        return eligible_ops[np.argmin(processing_times)]
    
    def LPT(self, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """Longest Processing Time - select operation with maximum duration"""
        processing_times = self._get_processing_times(eligible_ops, dur)
        return eligible_ops[np.argmax(processing_times)]
    
    # ========== JOB-LEVEL RULES ==========
    
    def STPT(self, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """Shortest Total Processing Time - select op from job with minimum total time"""
        total_times = []
        for op in eligible_ops:
            job = self._get_job_from_op(op)
            total_time = dur[job, :].sum()
            total_times.append(total_time)
        return eligible_ops[np.argmin(total_times)]
    
    def LTPT(self, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """Longest Total Processing Time - select op from job with maximum total time"""
        total_times = []
        for op in eligible_ops:
            job = self._get_job_from_op(op)
            total_time = dur[job, :].sum()
            total_times.append(total_time)
        return eligible_ops[np.argmax(total_times)]
    
    def LOR(self, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """Least Operation Remaining - select op from job with fewest remaining operations"""
        remaining_ops = []
        for op in eligible_ops:
            job = self._get_job_from_op(op)
            remaining = (1 - finished_mark[job, :]).sum()
            remaining_ops.append(remaining)
        return eligible_ops[np.argmin(remaining_ops)]
    
    def MOR(self, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """Most Operation Remaining - select op from job with most remaining operations"""
        remaining_ops = []
        for op in eligible_ops:
            job = self._get_job_from_op(op)
            remaining = (1 - finished_mark[job, :]).sum()
            remaining_ops.append(remaining)
        return eligible_ops[np.argmax(remaining_ops)]
    
    def LWKR(self, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """Least Work Remaining - select op from job with minimum remaining processing time"""
        work_remaining = []
        for op in eligible_ops:
            job = self._get_job_from_op(op)
            remaining_work = (dur[job, :] * (1 - finished_mark[job, :])).sum()
            work_remaining.append(remaining_work)
        return eligible_ops[np.argmin(work_remaining)]
    
    def MWKR(self, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """Most Work Remaining - select op from job with maximum remaining processing time"""
        work_remaining = []
        for op in eligible_ops:
            job = self._get_job_from_op(op)
            remaining_work = (dur[job, :] * (1 - finished_mark[job, :])).sum()
            work_remaining.append(remaining_work)
        return eligible_ops[np.argmax(work_remaining)]
    
    # ========== MACHINE-LEVEL RULES ==========
    
    def LQNO(self, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """
        Least Queue Next Operation - select op whose next operation 
        goes to machine with shortest queue
        """
        next_queue_lengths = []
        
        for op in eligible_ops:
            pos_in_job = self._get_position_in_job(op)
            
            # Check if there's a next operation in this job
            if pos_in_job < self.n_m - 1:
                next_op = op + 1
                next_machine = m.flat[next_op] - 1  # Machine indices are 1-indexed
                next_queue_length = (opIDsOnMchs[next_machine] >= 0).sum()
                next_queue_lengths.append(next_queue_length)
            else:
                # Last operation in job - assign high value (won't be selected unless all are last)
                next_queue_lengths.append(float('inf'))
        
        # If all operations are last in their jobs, fall back to SPT
        if all(q == float('inf') for q in next_queue_lengths):
            return self.SPT(eligible_ops, dur, finished_mark, m, opIDsOnMchs)
        
        return eligible_ops[np.argmin(next_queue_lengths)]
    
    # ========== OTHER RULES ==========
    
    def RANDOM(self, eligible_ops, dur, finished_mark, m, opIDsOnMchs):
        """Random selection from eligible operations"""
        return np.random.choice(eligible_ops)


if __name__ == "__main__":
    # Test the PDRs
    n_j, n_m = 3, 3
    pdrs = PriorityDispatchingRules(n_j, n_m)
    
    # Create dummy data
    dur = np.array([[10, 20, 30],
                    [5, 15, 25],
                    [50, 40, 30]])
    
    finished_mark = np.array([[1, 0, 0],
                              [1, 1, 0],
                              [0, 0, 0]])
    
    m = np.array([[1, 2, 3],
                  [2, 3, 1],
                  [3, 1, 2]])
    
    opIDsOnMchs = np.array([[0, -1, -1],
                            [3, -1, -1],
                            [-1, -1, -1]])
    
    eligible_ops = np.array([1, 2, 5, 6, 7, 8])  # Example eligible operations
    
    # Test each rule
    for rule_idx in range(pdrs.num_rules):
        selected_op = pdrs.apply_rule(rule_idx, eligible_ops, dur, finished_mark, m, opIDsOnMchs)
        print(f"{pdrs.rule_names[rule_idx]}: Selected operation {selected_op}")