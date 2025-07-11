#!/usr/bin/env python3
"""
Quick script to check WLD model dimensions and target features
"""

import torch

def main():
    print("🔍 Checking WLD Model Configuration...")
    
    # Load WLD model
    checkpoint = torch.load('models/wld_3min_fixed/best_wld_3min_model.pt', map_location='cpu')
    target_indices = checkpoint['target_feature_indices']
    
    print(f"WLD Model Target Features: {len(target_indices)}")
    print(f"Target Pair: {checkpoint.get('target_pair', 'Unknown')}")
    print(f"Target Steps: {checkpoint.get('target_steps', 'Unknown')}")
    print(f"First 10 target indices: {target_indices[:10]}")
    print()
    print("Expected for WLD-USDT only: 20 features")
    print("Backtest script expects: 80 features")
    print()
    
    if len(target_indices) == 20:
        print("✅ CORRECT: Model outputs 20 WLD features")
    elif len(target_indices) == 80:
        print("⚠️  Model outputs 80 Binance perp features (not WLD-only)")
    else:
        print(f"❌ UNEXPECTED: Model outputs {len(target_indices)} features")
    
    print()
    print("Final validation loss:", checkpoint.get('best_val_loss', 'Unknown'))

if __name__ == "__main__":
    main() 