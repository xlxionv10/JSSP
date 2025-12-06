import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm instead of BatchNorm"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        
        self.linear_or_not = True
        self.num_layers = num_layers
        
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.layer_norms = torch.nn.ModuleList()
            
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            
            # Use LayerNorm instead of BatchNorm (works with batch_size=1)
            for layer in range(num_layers - 1):
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                h = self.layer_norms[layer](h)
                h = F.relu(h)
            return self.linears[self.num_layers - 1](h)


class ActorCriticPDR(nn.Module):
    """
    Simplified Actor-Critic for PDR selection
    
    Takes state features as input and outputs:
    - Actor: Probability distribution over PDRs
    - Critic: Value estimate of current state
    """
    
    def __init__(self,
                 feature_dim,
                 num_pdrs=10,
                 num_mlp_layers_actor=3,
                 hidden_dim_actor=64,
                 num_mlp_layers_critic=3,
                 hidden_dim_critic=64,
                 device='cpu'):
        super(ActorCriticPDR, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_pdrs = num_pdrs
        self.device = device
        
        # Actor network: features → PDR scores
        self.actor = MLP(
            num_layers=num_mlp_layers_actor,
            input_dim=feature_dim,
            hidden_dim=hidden_dim_actor,
            output_dim=num_pdrs
        ).to(device)
        
        # Critic network: features → value
        self.critic = MLP(
            num_layers=num_mlp_layers_critic,
            input_dim=feature_dim,
            hidden_dim=hidden_dim_critic,
            output_dim=1
        ).to(device)
    
    def forward(self, state_features):
        """
        Forward pass
        
        Args:
            state_features: Tensor of shape (batch_size, feature_dim) or (feature_dim,)
        
        Returns:
            pi: Probability distribution over PDRs, shape (batch_size, num_pdrs) or (num_pdrs,)
            value: Value estimate, shape (batch_size, 1) or (1,)
        """
        # Ensure input is 2D
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        
        # Get PDR scores
        pdr_scores = self.actor(state_features)
        
        # Convert to probabilities
        pi = F.softmax(pdr_scores, dim=-1)
        
        # Get value estimate
        value = self.critic(state_features)
        
        return pi, value


if __name__ == "__main__":
    # Test the model
    feature_dim = 156  # For 15x15 problem
    num_pdrs = 10
    batch_size = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ActorCriticPDR(
        feature_dim=feature_dim,
        num_pdrs=num_pdrs,
        num_mlp_layers_actor=3,
        hidden_dim_actor=64,
        num_mlp_layers_critic=3,
        hidden_dim_critic=64,
        device=device
    )
    
    print(f"Model created on device: {device}")
    print(f"Actor parameters: {sum(p.numel() for p in model.actor.parameters())}")
    print(f"Critic parameters: {sum(p.numel() for p in model.critic.parameters())}")
    
    # Test forward pass with single sample
    features_single = torch.randn(feature_dim).to(device)
    pi_single, value_single = model(features_single)
    print(f"\nSingle sample:")
    print(f"  Input shape: {features_single.shape}")
    print(f"  Pi shape: {pi_single.shape}")
    print(f"  Value shape: {value_single.shape}")
    print(f"  Pi sum: {pi_single.sum().item():.4f} (should be ~1.0)")
    
    # Test forward pass with batch
    features_batch = torch.randn(batch_size, feature_dim).to(device)
    pi_batch, value_batch = model(features_batch)
    print(f"\nBatch:")
    print(f"  Input shape: {features_batch.shape}")
    print(f"  Pi shape: {pi_batch.shape}")
    print(f"  Value shape: {value_batch.shape}")
    print(f"  Pi sums: {pi_batch.sum(dim=-1)}")  # Each should be ~1.0