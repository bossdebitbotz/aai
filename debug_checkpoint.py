#!/usr/bin/env python3
"""
Debug script to inspect the trained model checkpoint structure
"""

import torch
import json

def inspect_checkpoint():
    """Inspect the checkpoint to see what's actually saved."""
    print("🔍 Inspecting Model Checkpoint...")
    
    checkpoint_path = "models/working_scaled_multigpu/best_model_scaled.pt"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("📋 Checkpoint Keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        print("\n⚙️ Model Config (if exists):")
        if 'model_config' in checkpoint:
            for key, value in checkpoint['model_config'].items():
                print(f"  - {key}: {value}")
        else:
            print("  ❌ No 'model_config' found")
        
        print("\n🎯 Target Feature Indices:")
        if 'target_feature_indices' in checkpoint:
            indices = checkpoint['target_feature_indices']
            print(f"  - Count: {len(indices)}")
            print(f"  - Indices: {indices[:10]}{'...' if len(indices) > 10 else ''}")
        else:
            print("  ❌ No 'target_feature_indices' found")
        
        print("\n📊 Training Metrics:")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Train Loss: {checkpoint.get('train_loss', 'N/A')}")
        print(f"  - Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        
        print("\n🧠 Model State Dict Keys (first 10):")
        if 'model_state_dict' in checkpoint:
            state_keys = list(checkpoint['model_state_dict'].keys())
            for i, key in enumerate(state_keys[:10]):
                print(f"  - {key}")
            if len(state_keys) > 10:
                print(f"  - ... and {len(state_keys) - 10} more")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_checkpoint() 