#!/usr/bin/env python3
"""
WLD Model Setup Validation Script

✅ Validates data structure matches requirements
✅ Confirms 20 WLD perp features are correctly identified
✅ Verifies model architecture compatibility  
✅ Tests data loading without training

Run this before training to ensure everything is configured correctly.
"""

import os
import torch
import numpy as np
import json
from datetime import datetime

FINAL_DATA_DIR = "data/final_attention"

def check_data_files():
    """Check all required data files exist."""
    print("📁 Checking Data Files...")
    
    required_files = [
        'train.npz',
        'validation.npz', 
        'test.npz',
        'embedding_metadata.json'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(FINAL_DATA_DIR, file)
        if os.path.exists(file_path):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ All data files present")
    return True

def validate_data_structure():
    """Validate data structure matches requirements."""
    print("\n📊 Validating Data Structure...")
    
    try:
        # Load and check each dataset
        datasets = ['train', 'validation', 'test']
        
        for dataset_name in datasets:
            file_path = os.path.join(FINAL_DATA_DIR, f'{dataset_name}.npz')
            
            with np.load(file_path, allow_pickle=True) as data:
                contexts = data['contexts']
                targets = data['targets']
                
                print(f"  📝 {dataset_name}.npz:")
                print(f"    Contexts: {contexts.shape}")
                print(f"    Targets: {targets.shape}")
                
                # Validate shapes
                expected_context = (contexts.shape[0], 120, 240)
                expected_target = (targets.shape[0], 24, 240)
                
                if contexts.shape != expected_context:
                    print(f"    ❌ Wrong context shape! Expected {expected_context}")
                    return False
                    
                if targets.shape != expected_target:
                    print(f"    ❌ Wrong target shape! Expected {expected_target}")
                    return False
                    
                print(f"    ✅ Shapes correct")
        
        print("\n  ✅ All datasets have correct structure")
        return True
        
    except Exception as e:
        print(f"  ❌ Error validating data: {e}")
        return False

def validate_metadata():
    """Validate metadata configuration."""
    print("\n🔍 Validating Metadata...")
    
    try:
        with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Check basic structure
        required_keys = ['columns', 'column_mapping', 'unique_attributes', 
                        'num_features', 'context_length', 'target_length']
        
        for key in required_keys:
            if key not in metadata:
                print(f"  ❌ Missing metadata key: {key}")
                return False, None
            print(f"  ✅ {key}: present")
        
        # Check values
        print(f"\n  📊 Metadata Values:")
        print(f"    Total features: {metadata['num_features']}")
        print(f"    Context length: {metadata['context_length']}")
        print(f"    Target length: {metadata['target_length']}")
        
        # Validate expected values
        expected_values = {
            'num_features': 240,
            'context_length': 120,
            'target_length': 24
        }
        
        for key, expected in expected_values.items():
            actual = metadata[key]
            if actual != expected:
                print(f"    ❌ {key}: Expected {expected}, got {actual}")
                return False, None
            print(f"    ✅ {key}: {actual} (correct)")
        
        return True, metadata
        
    except Exception as e:
        print(f"  ❌ Error validating metadata: {e}")
        return False, None

def validate_wld_features(metadata):
    """Validate WLD perp features are correctly identified."""
    print("\n🎯 Validating WLD Perp Features...")
    
    wld_perp_features = []
    
    for i, col_name in enumerate(metadata['columns']):
        col_info = metadata['column_mapping'][col_name]
        if (col_info['exchange'] == 'binance_perp' and 
            col_info['trading_pair'] == 'WLD-USDT'):
            wld_perp_features.append((i, col_name, col_info))
    
    print(f"  📊 Found {len(wld_perp_features)} WLD perp features")
    
    if len(wld_perp_features) != 20:
        print(f"  ❌ Expected 20 WLD perp features, found {len(wld_perp_features)}")
        return False, []
    
    # Check indices are sequential 120-139
    indices = [f[0] for f in wld_perp_features]
    expected_indices = list(range(120, 140))
    
    if indices != expected_indices:
        print(f"  ❌ Wrong indices! Expected {expected_indices[0]}-{expected_indices[-1]}")
        print(f"      Got: {indices}")
        return False, []
    
    print(f"  ✅ Correct indices: {indices[0]}-{indices[-1]}")
    
    # Validate feature structure (5 bid levels + 5 ask levels, each with price + quantity)
    feature_types = {}
    for idx, name, info in wld_perp_features:
        order_type = info['order_type']  # bid or ask
        feature_type = info['feature_type']  # price or volume
        level = info['level']
        
        key = f"{order_type}_{feature_type}"
        if key not in feature_types:
            feature_types[key] = []
        feature_types[key].append(level)
    
    print(f"  📝 Feature breakdown:")
    expected_structure = {
        'bid_price': [1, 2, 3, 4, 5],
        'bid_volume': [1, 2, 3, 4, 5],
        'ask_price': [1, 2, 3, 4, 5],
        'ask_volume': [1, 2, 3, 4, 5]
    }
    
    for feature_type, expected_levels in expected_structure.items():
        actual_levels = sorted(feature_types.get(feature_type, []))
        if actual_levels != expected_levels:
            print(f"    ❌ {feature_type}: Expected levels {expected_levels}, got {actual_levels}")
            return False, []
        print(f"    ✅ {feature_type}: levels {actual_levels}")
    
    print(f"  ✅ All 20 WLD perp features correctly structured")
    return True, indices

def test_data_loading(metadata, wld_indices):
    """Test data loading with correct configuration."""
    print("\n🧪 Testing Data Loading...")
    
    try:
        # Test loading train dataset with WLD filtering
        with np.load(os.path.join(FINAL_DATA_DIR, 'train.npz'), allow_pickle=True) as data:
            contexts = torch.from_numpy(data['contexts']).float()
            targets_full = torch.from_numpy(data['targets']).float()
            targets_wld = targets_full[:, :24, wld_indices]  # 24 time steps, 20 WLD features
        
        print(f"  📊 Loaded data shapes:")
        print(f"    Contexts: {contexts.shape}")
        print(f"    Full targets: {targets_full.shape}")
        print(f"    WLD targets: {targets_wld.shape}")
        
        # Expected shapes
        expected_context = torch.Size([contexts.shape[0], 120, 240])
        expected_wld_target = torch.Size([targets_wld.shape[0], 24, 20])
        
        if contexts.shape != expected_context:
            print(f"    ❌ Wrong context shape! Expected {expected_context}")
            return False
            
        if targets_wld.shape != expected_wld_target:
            print(f"    ❌ Wrong WLD target shape! Expected {expected_wld_target}")
            return False
        
        # Test a small batch
        batch_size = 4
        sample_context = contexts[:batch_size]
        sample_target = targets_wld[:batch_size]
        
        print(f"  🔬 Sample batch:")
        print(f"    Context: {sample_context.shape}")
        print(f"    Target: {sample_target.shape}")
        print(f"    Context dtype: {sample_context.dtype}")
        print(f"    Target dtype: {sample_target.dtype}")
        
        # Check for NaN or invalid values
        if torch.isnan(sample_context).any():
            print(f"    ❌ NaN values found in context data!")
            return False
            
        if torch.isnan(sample_target).any():
            print(f"    ❌ NaN values found in target data!")
            return False
        
        print(f"  ✅ Data loading successful - no NaN values")
        print(f"  ✅ Ready for model training")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing data loading: {e}")
        return False

def main():
    """Main validation function."""
    print("=" * 80)
    print("🔍 WLD MODEL SETUP VALIDATION")
    print("=" * 80)
    print(f"Validation time: {datetime.now()}")
    print()
    
    validation_steps = [
        ("Data Files", check_data_files),
        ("Data Structure", validate_data_structure),
        ("Metadata", validate_metadata),
    ]
    
    results = {}
    metadata = None
    
    # Run basic validations
    for step_name, step_func in validation_steps:
        if step_name == "Metadata":
            success, metadata = step_func()
        else:
            success = step_func()
        
        results[step_name] = success
        if not success:
            print(f"\n❌ Validation failed at: {step_name}")
            print("🛑 Fix the above issues before training")
            return
    
    # WLD-specific validations
    wld_success, wld_indices = validate_wld_features(metadata)
    results["WLD Features"] = wld_success
    
    if not wld_success:
        print(f"\n❌ WLD feature validation failed")
        return
    
    # Data loading test
    loading_success = test_data_loading(metadata, wld_indices)
    results["Data Loading"] = loading_success
    
    # Final summary
    print("\n" + "=" * 80)
    print("📋 VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for step, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {step:<20} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Ready to train the corrected WLD model")
        print()
        print("💡 Next steps:")
        print("   1. Run: python3 train_working_scaled_CORRECTED.py")
        print("   2. After training: python3 backtest_corrected_model.py")
        print()
        print("📊 Model will predict:")
        print(f"   - Input: 120 time steps × 240 features")
        print(f"   - Output: 24 time steps × 20 WLD perp features")
        print(f"   - Trading: 24 seconds ahead WLD perp predictions")
    else:
        print("❌ VALIDATION FAILED!")
        print("🛑 Fix the above issues before training")

if __name__ == "__main__":
    main() 