#!/usr/bin/env python3
"""
Quick test script to verify the attention-based LOB model setup
before starting full training.
"""

import os
import torch
import json
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_setup_test")

def test_data_loading():
    """Test that data can be loaded correctly."""
    logger.info("Testing data loading...")
    
    data_dir = "data/final_attention"
    
    # Check if all required files exist
    required_files = [
        'train.npz', 'validation.npz', 'test.npz',
        'embedding_metadata.json', 'config.json',
        'scaler.pkl', 'dataset_stats.json'
    ]
    
    for file in required_files:
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            logger.error(f"Missing required file: {path}")
            return False
        else:
            logger.info(f"✓ Found: {file}")
    
    # Load and check data shapes
    with np.load(os.path.join(data_dir, 'train.npz')) as data:
        x_shape = data['x'].shape
        y_shape = data['y'].shape
        logger.info(f"Train data shapes - X: {x_shape}, Y: {y_shape}")
    
    # Load metadata
    with open(os.path.join(data_dir, 'embedding_metadata.json'), 'r') as f:
        metadata = json.load(f)
        logger.info(f"Number of features: {metadata['num_features']}")
        logger.info(f"Number of columns: {len(metadata['columns'])}")
    
    # Load config
    with open(os.path.join(data_dir, 'config.json'), 'r') as f:
        config = json.load(f)
        logger.info(f"Context length: {config['context_length']}")
        logger.info(f"Target length: {config['target_length']}")
    
    return True

def test_model_initialization():
    """Test that the model can be initialized."""
    logger.info("Testing model initialization...")
    
    try:
        # Import the model components
        from train_attention_model import LOBForecaster, CompoundMultivariateEmbedding, StructuralLoss
        
        # Load metadata
        with open("data/final_attention/embedding_metadata.json", 'r') as f:
            embedding_metadata = json.load(f)
        
        with open("data/final_attention/config.json", 'r') as f:
            config = json.load(f)
        
        # Initialize model
        model = LOBForecaster(
            embedding_metadata=embedding_metadata,
            embed_dim=126,  # Divisible by 3 heads
            num_heads=3,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dropout=0.1,
            target_len=config['target_length']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model initialized with {total_params:,} parameters")
        
        # Test structural loss
        struct_loss = StructuralLoss(embedding_metadata)
        logger.info("✓ Structural loss initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return False

def test_forward_pass():
    """Test a forward pass with sample data."""
    logger.info("Testing forward pass...")
    
    try:
        from train_attention_model import LOBForecaster, LOBDataset
        
        # Load metadata and config
        with open("data/final_attention/embedding_metadata.json", 'r') as f:
            embedding_metadata = json.load(f)
        
        with open("data/final_attention/config.json", 'r') as f:
            config = json.load(f)
        
        # Initialize model
        model = LOBForecaster(
            embedding_metadata=embedding_metadata,
            embed_dim=126,  # Divisible by 3 heads
            num_heads=3,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dropout=0.1,
            target_len=config['target_length']
        )
        
        model.eval()
        
        # Load a small sample from the dataset
        dataset = LOBDataset("data/final_attention/validation.npz")
        sample_x, sample_y = dataset[0]
        
        # Add batch dimension
        context = sample_x.unsqueeze(0)  # (1, context_len, num_features)
        target = sample_y.unsqueeze(0)   # (1, target_len, num_features)
        
        logger.info(f"Input shapes - Context: {context.shape}, Target: {target.shape}")
        
        # Create decoder input (shifted target)
        decoder_input = torch.zeros_like(target)
        decoder_input[:, 1:] = target[:, :-1]
        
        # Forward pass
        with torch.no_grad():
            predictions = model(context, decoder_input)
        
        logger.info(f"✓ Forward pass successful - Output shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting model setup verification...")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Initialization", test_model_initialization),
        ("Forward Pass", test_forward_pass)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Model is ready for training.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 