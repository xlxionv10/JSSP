import argparse

parser = argparse.ArgumentParser(description='Arguments for PPO with PDR selection')

# Device args
parser.add_argument('--device', type=str, default="cpu", help='Device: cuda or cpu')

# Environment args
parser.add_argument('--n_j', type=int, default=6, help='Number of jobs')
parser.add_argument('--n_m', type=int, default=6, help='Number of machines')
parser.add_argument('--rewardscale', type=float, default=0., help='Reward scale for zero-change rewards')
parser.add_argument('--init_quality_flag', type=bool, default=False, help='If True, initial quality is 0')
parser.add_argument('--low', type=int, default=1, help='Lower bound of operation duration')
parser.add_argument('--high', type=int, default=99, help='Upper bound of operation duration')

# Random seeds
parser.add_argument('--np_seed_train', type=int, default=200, help='Numpy seed for training')
parser.add_argument('--np_seed_validation', type=int, default=200, help='Numpy seed for validation')
parser.add_argument('--torch_seed', type=int, default=600, help='PyTorch seed')

# Normalization (kept for compatibility, though less critical with statistics)
parser.add_argument('--et_normalize_coef', type=int, default=1000, 
                   help='Normalization coefficient for end time features')

# PDR-specific args
parser.add_argument('--num_pdrs', type=int, default=10, help='Number of PDRs')

# Network architecture args (simplified for PDR)
parser.add_argument('--num_mlp_layers_actor', type=int, default=3, 
                   help='Number of layers in actor MLP')
parser.add_argument('--hidden_dim_actor', type=int, default=64, 
                   help='Hidden dimension of actor MLP')
parser.add_argument('--num_mlp_layers_critic', type=int, default=3, 
                   help='Number of layers in critic MLP')
parser.add_argument('--hidden_dim_critic', type=int, default=64, 
                   help='Hidden dimension of critic MLP')

# PPO hyperparameters
parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments')
parser.add_argument('--max_updates', type=int, default=10000, help='Maximum number of updates')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--decayflag', type=bool, default=False, help='Enable learning rate decay')
parser.add_argument('--decay_step_size', type=int, default=2000, help='Steps between LR decay')
parser.add_argument('--decay_ratio', type=float, default=0.9, help='LR decay ratio')
parser.add_argument('--gamma', type=float, default=1.0, help='Discount factor')
parser.add_argument('--k_epochs', type=int, default=4, help='Number of PPO epochs per update')
parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clipping parameter')

# Loss coefficients
parser.add_argument('--vloss_coef', type=float, default=1.0, help='Value loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=1.0, help='Policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='Entropy loss coefficient')

configs = parser.parse_args()


# Print configuration (useful for debugging)
if __name__ == "__main__":
    print("Configuration Parameters for PDR-based JSSP:")
    print("=" * 60)
    print(f"Problem Size: {configs.n_j} jobs × {configs.n_m} machines")
    print(f"Device: {configs.device}")
    print(f"Number of PDRs: {configs.num_pdrs}")
    print(f"\nNetwork Architecture:")
    print(f"  Actor: {configs.num_mlp_layers_actor} layers, hidden_dim={configs.hidden_dim_actor}")
    print(f"  Critic: {configs.num_mlp_layers_critic} layers, hidden_dim={configs.hidden_dim_critic}")
    print(f"\nPPO Hyperparameters:")
    print(f"  Learning rate: {configs.lr}")
    print(f"  Gamma: {configs.gamma}")
    print(f"  Epochs per update: {configs.k_epochs}")
    print(f"  Epsilon clip: {configs.eps_clip}")
    print(f"  Number of envs: {configs.num_envs}")
    print(f"  Max updates: {configs.max_updates}")
    print("=" * 60)