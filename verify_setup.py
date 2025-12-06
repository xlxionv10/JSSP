"""
Verification script to check if all PDR components are properly set up
Run this before training to ensure everything works correctly
"""

import sys
import numpy as np

def check_imports():
    """Check if all required modules can be imported"""
    print("=" * 60)
    print("Checking imports...")
    print("=" * 60)
    
    required_modules = [
        ('torch', 'PyTorch'),
        ('gym', 'OpenAI Gym'),
        ('numpy', 'NumPy'),
    ]
    
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            return False
    
    return True


def check_custom_files():
    """Check if all custom files exist"""
    print("\n" + "=" * 60)
    print("Checking custom files...")
    print("=" * 60)
    
    required_files = [
        ('PDRs.py', 'Priority Dispatching Rules'),
        ('state_features.py', 'Feature Extractor'),
        ('JSSP_Env_PDR.py', 'PDR Environment'),
        ('PPO_jssp_PDR.py', 'Training Script'),
        ('test_learned_PDR.py', 'Testing Script'),
        ('uniform_instance_gen.py', 'Instance Generator'),
        ('permissibleLS.py', 'Permissible Left Shift'),
        ('updateEntTimeLB.py', 'Lower Bound Update'),
        ('updateAdjMat.py', 'Adjacency Matrix Update'),
    ]
    
    all_exist = True
    for filename, description in required_files:
        try:
            with open(filename, 'r') as f:
                print(f"✓ {description:30s} ({filename})")
        except FileNotFoundError:
            print(f"✗ {description:30s} ({filename}) NOT FOUND")
            all_exist = False
    
    # Check model directory
    try:
        with open('models/actor_critic_pdr.py', 'r') as f:
            print(f"✓ {'Actor-Critic PDR Model':30s} (models/actor_critic_pdr.py)")
    except FileNotFoundError:
        print(f"✗ {'Actor-Critic PDR Model':30s} (models/actor_critic_pdr.py) NOT FOUND")
        all_exist = False
    
    return all_exist


def test_pdrs():
    """Test PDR implementations"""
    print("\n" + "=" * 60)
    print("Testing PDR implementations...")
    print("=" * 60)
    
    try:
        from PDRs import PriorityDispatchingRules
        
        n_j, n_m = 3, 3
        pdrs = PriorityDispatchingRules(n_j, n_m)
        
        # Create dummy data
        dur = np.array([[10, 20, 30],
                        [5, 15, 25],
                        [50, 40, 30]], dtype=np.float32)
        
        finished_mark = np.array([[1, 0, 0],
                                  [1, 1, 0],
                                  [0, 0, 0]], dtype=np.float32)
        
        m = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2]], dtype=np.int32)
        
        opIDsOnMchs = np.array([[0, -1, -1],
                                [3, -1, -1],
                                [-1, -1, -1]], dtype=np.int32)
        
        eligible_ops = np.array([1, 2, 5, 6, 7, 8])
        
        # Test each rule
        print(f"Testing {pdrs.num_rules} PDRs on dummy problem...")
        for rule_idx in range(pdrs.num_rules):
            try:
                selected_op = pdrs.apply_rule(rule_idx, eligible_ops, dur, 
                                              finished_mark, m, opIDsOnMchs)
                print(f"  ✓ {pdrs.rule_names[rule_idx]:10s} → operation {selected_op}")
            except Exception as e:
                print(f"  ✗ {pdrs.rule_names[rule_idx]:10s} → ERROR: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing PDRs: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction"""
    print("\n" + "=" * 60)
    print("Testing feature extraction...")
    print("=" * 60)
    
    try:
        from state_features import StateFeatureExtractor
        from JSSP_Env_PDR import SJSSP_PDR
        from uniform_instance_gen import uni_instance_gen
        
        n_j, n_m = 6, 6
        
        # Create environment
        env = SJSSP_PDR(n_j=n_j, n_m=n_m)
        
        # Generate instance
        data = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=99)
        env.reset(data)
        
        # Extract features
        extractor = StateFeatureExtractor(n_j, n_m)
        features = extractor.extract_features(env)
        
        expected_dim = extractor.feature_dim
        actual_dim = features.shape[0]
        
        if actual_dim == expected_dim:
            print(f"✓ Feature extraction successful")
            print(f"  Expected dimension: {expected_dim}")
            print(f"  Actual dimension: {actual_dim}")
            print(f"  Feature sample: {features[:5]}")
            return True
        else:
            print(f"✗ Feature dimension mismatch")
            print(f"  Expected: {expected_dim}, Got: {actual_dim}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """Test PDR environment"""
    print("\n" + "=" * 60)
    print("Testing PDR environment...")
    print("=" * 60)
    
    try:
        from JSSP_Env_PDR import SJSSP_PDR
        from uniform_instance_gen import uni_instance_gen
        
        n_j, n_m = 6, 6
        env = SJSSP_PDR(n_j=n_j, n_m=n_m)
        
        # Generate instance
        data = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=99)
        
        # Reset
        state = env.reset(data)
        print(f"✓ Environment reset successful")
        print(f"  State shape: {state.shape}")
        print(f"  Initial quality: {env.initQuality}")
        
        # Take a few random steps
        for step in range(5):
            pdr_action = np.random.randint(0, env.num_pdrs)
            state, reward, done = env.step(pdr_action)
            
            if done:
                break
        
        print(f"✓ Environment stepping successful")
        print(f"  Steps taken: {step + 1}")
        print(f"  Episode done: {done}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_network():
    """Test neural network"""
    print("\n" + "=" * 60)
    print("Testing neural network...")
    print("=" * 60)
    
    try:
        import torch
        from models.actor_critic_pdr import ActorCriticPDR
        
        n_j, n_m = 6, 6
        feature_dim = n_j * 5 + n_m * 3 + 36  # Expected feature dimension
        num_pdrs = 10
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {device}")
        
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
        
        print(f"✓ Model created successfully")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Number of PDRs: {num_pdrs}")
        
        # Test forward pass
        features = torch.randn(feature_dim).to(device)
        pi, value = model(features)
        
        print(f"✓ Forward pass successful")
        print(f"  Policy shape: {pi.shape}")
        print(f"  Value shape: {value.shape}")
        print(f"  Policy sum: {pi.sum().item():.4f} (should be ~1.0)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing network: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "PDR SETUP VERIFICATION" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    tests = [
        ("Imports", check_imports),
        ("Files", check_custom_files),
        ("PDRs", test_pdrs),
        ("Feature Extraction", test_feature_extraction),
        ("Environment", test_environment),
        ("Neural Network", test_network),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:25s}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. Generate validation data: python generate_data.py")
        print("  2. Train model: python PPO_jssp_PDR.py")
        print("  3. Test model: python test_learned_PDR.py --greedy")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon issues:")
        print("  - Missing files: Ensure all new files are created")
        print("  - Import errors: Check file paths and dependencies")
        print("  - Module errors: Verify Python environment")
        return 1


if __name__ == "__main__":
    sys.exit(main())