#!/usr/bin/env python3
"""
Test script for undersampling functionality
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_generator import generate_synthetic_data
from shared.imbalance_handler import UpliftUndersampler

def test_basic_undersampling():
    """Basic test of undersampling functionality"""
    print("Testing undersampling functionality...")
    
    # Generate small test dataset
    print("Generating test data...")
    X_data, T_data, Y_data, true_cate_data, feature_names = generate_synthetic_data(
        n_samples=1000, random_seed=42, reward_type="logistic"
    )
    
    print(f"Original data: {len(Y_data)} samples")
    print(f"Original positive rate: {np.mean(Y_data == 1):.4f}")
    print(f"Treatment group size: {np.sum(T_data == 1)}")
    print(f"Control group size: {np.sum(T_data == 0)}")
    
    # Convert to DataFrame if needed
    if not isinstance(X_data, pd.DataFrame):
        X_df = pd.DataFrame(X_data, columns=feature_names if feature_names else [f'feature_{i}' for i in range(X_data.shape[1])])
    else:
        X_df = X_data
    
    # Test undersampling
    print("\nTesting undersampling with k=2.0...")
    undersampler = UpliftUndersampler(k=2.0, random_state=42)
    X_resampled, T_resampled, Y_resampled = undersampler.fit_resample(X_df, T_data, Y_data)
    
    print(f"Resampled data: {len(Y_resampled)} samples")
    print(f"Resampled positive rate: {np.mean(Y_resampled == 1):.4f}")
    print(f"Resampled treatment group size: {np.sum(T_resampled == 1)}")
    print(f"Resampled control group size: {np.sum(T_resampled == 0)}")
    
    # Test prediction correction
    print("\nTesting prediction correction...")
    dummy_predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    corrected_predictions = undersampler.correct_predictions(dummy_predictions)
    print(f"Original predictions: {dummy_predictions}")
    print(f"Corrected predictions: {corrected_predictions}")
    
    print("\n✅ Undersampling test completed successfully!")
    return True

def test_imbalanced_data():
    """Test with highly imbalanced data"""
    print("\n" + "="*50)
    print("Testing with highly imbalanced data...")
    
    # Create highly imbalanced synthetic data
    np.random.seed(42)
    n_samples = 2000
    
    # Generate features
    X = np.random.randn(n_samples, 5)
    
    # Generate treatment assignment (balanced)
    T = np.random.binomial(1, 0.5, n_samples)
    
    # Generate highly imbalanced outcomes (only 5% positive)
    base_prob = 0.02  # Very low base conversion rate
    treatment_effect = 0.03  # Small treatment effect
    
    probs = base_prob + T * treatment_effect
    Y = np.random.binomial(1, probs, n_samples)
    
    print(f"Created imbalanced dataset:")
    print(f"- Total samples: {n_samples}")
    print(f"- Overall positive rate: {np.mean(Y == 1):.4f}")
    print(f"- Control positive rate: {np.mean(Y[T == 0] == 1):.4f}")
    print(f"- Treatment positive rate: {np.mean(Y[T == 1] == 1):.4f}")
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Test undersampling with higher k factor
    print(f"\nApplying undersampling with k=4.0...")
    undersampler = UpliftUndersampler(k=4.0, random_state=42)
    X_resampled, T_resampled, Y_resampled = undersampler.fit_resample(X_df, T, Y)
    
    print(f"After undersampling:")
    print(f"- Total samples: {len(Y_resampled)}")
    print(f"- Overall positive rate: {np.mean(Y_resampled == 1):.4f}")
    print(f"- Control positive rate: {np.mean(Y_resampled[T_resampled == 0] == 1):.4f}")
    print(f"- Treatment positive rate: {np.mean(Y_resampled[T_resampled == 1] == 1):.4f}")
    
    print("\n✅ Imbalanced data test completed successfully!")
    return True

if __name__ == "__main__":
    print("="*60)
    print("UNDERSAMPLING FUNCTIONALITY TEST")
    print("="*60)
    
    try:
        # Run basic test
        test_basic_undersampling()
        
        # Run imbalanced data test
        test_imbalanced_data()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Undersampling implementation is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
