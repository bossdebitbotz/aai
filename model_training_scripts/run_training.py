#!/usr/bin/env python3
"""
Master Training Script for LOB Forecasting Models

This script provides an easy interface to run any of the 16 specialized
training scripts for different prediction timeframes and target pairs.

Usage:
    python run_training.py --timeframe 1 --pair wld                    # Single training
    python run_training.py --timeframe 1 --pair all                    # All pairs for 1min
    python run_training.py --timeframe all --pair btc                  # All timeframes for BTC
    python run_training.py --timeframe all --pair all                  # Train all 16 models
    python run_training.py --timeframe all --pair all --parallel       # Parallel training
"""

import os
import sys
import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("master_training")

def run_training_script(script_path, background=False):
    """Run a single training script."""
    script_name = os.path.basename(script_path)
    logger.info(f"Starting training: {script_name}")
    
    try:
        if background:
            # Run in background and return immediately
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            return script_name, process
        else:
            # Run and wait for completion
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Completed successfully: {script_name}")
                return script_name, True, result.stdout
            else:
                logger.error(f"❌ Failed: {script_name}")
                logger.error(f"Error output: {result.stderr}")
                return script_name, False, result.stderr
                
    except Exception as e:
        logger.error(f"❌ Exception running {script_name}: {e}")
        return script_name, False, str(e)

def get_script_paths(timeframes, pairs):
    """Get list of script paths based on timeframes and pairs."""
    script_paths = []
    
    for timeframe in timeframes:
        for pair in pairs:
            script_name = f"train_{timeframe}min_binance_perp_{pair}.py"
            script_path = os.path.join(
                "model_training_scripts", 
                f"{timeframe}min_predictions", 
                script_name
            )
            
            if os.path.exists(script_path):
                script_paths.append(script_path)
            else:
                logger.warning(f"Script not found: {script_path}")
    
    return script_paths

def run_parallel_training(script_paths, max_workers=4):
    """Run multiple training scripts in parallel."""
    logger.info(f"Starting parallel training of {len(script_paths)} models with {max_workers} workers")
    
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_script = {
            executor.submit(run_training_script, script_path): script_path 
            for script_path in script_paths
        }
        
        # Process results as they complete
        for future in as_completed(future_to_script):
            script_path = future_to_script[future]
            script_name = os.path.basename(script_path)
            
            try:
                script_name, success, output = future.result()
                if success:
                    completed += 1
                    logger.info(f"✅ Completed ({completed}/{len(script_paths)}): {script_name}")
                else:
                    failed += 1
                    logger.error(f"❌ Failed ({failed}/{len(script_paths)}): {script_name}")
            except Exception as e:
                failed += 1
                logger.error(f"❌ Exception ({failed}/{len(script_paths)}): {script_name} - {e}")
    
    logger.info(f"Parallel training completed: {completed} successful, {failed} failed")
    return completed, failed

def run_sequential_training(script_paths):
    """Run training scripts one by one."""
    logger.info(f"Starting sequential training of {len(script_paths)} models")
    
    completed = 0
    failed = 0
    
    for i, script_path in enumerate(script_paths, 1):
        script_name = os.path.basename(script_path)
        logger.info(f"Running {i}/{len(script_paths)}: {script_name}")
        
        script_name, success, output = run_training_script(script_path)
        
        if success:
            completed += 1
        else:
            failed += 1
            # Ask if user wants to continue after failure
            response = input(f"Training failed for {script_name}. Continue with remaining scripts? (y/n): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("Training stopped by user")
                break
    
    logger.info(f"Sequential training completed: {completed} successful, {failed} failed")
    return completed, failed

def main():
    parser = argparse.ArgumentParser(description="Master training script for LOB forecasting models")
    
    parser.add_argument(
        "--timeframe", 
        choices=['1', '2', '3', '5', 'all'],
        required=True,
        help="Prediction timeframe in minutes (1, 2, 3, 5, or 'all')"
    )
    
    parser.add_argument(
        "--pair",
        choices=['wld', 'sol', 'eth', 'btc', 'all'],
        required=True,
        help="Target trading pair (wld, sol, eth, btc, or 'all')"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run training scripts in parallel (default: sequential)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list the scripts that would be run, don't execute them"
    )
    
    args = parser.parse_args()
    
    # Parse timeframes
    if args.timeframe == 'all':
        timeframes = [1, 2, 3, 5]
    else:
        timeframes = [int(args.timeframe)]
    
    # Parse pairs
    if args.pair == 'all':
        pairs = ['wld', 'sol', 'eth', 'btc']
    else:
        pairs = [args.pair]
    
    # Get script paths
    script_paths = get_script_paths(timeframes, pairs)
    
    if not script_paths:
        logger.error("No valid script paths found!")
        return 1
    
    # List scripts
    logger.info(f"Found {len(script_paths)} training scripts:")
    for i, path in enumerate(script_paths, 1):
        logger.info(f"  {i}. {os.path.basename(path)}")
    
    if args.list_only:
        return 0
    
    # Confirm execution
    if len(script_paths) > 1:
        response = input(f"\nProceed with training {len(script_paths)} models? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Training cancelled by user")
            return 0
    
    # Run training
    start_time = time.time()
    
    if args.parallel and len(script_paths) > 1:
        completed, failed = run_parallel_training(script_paths, args.max_workers)
    else:
        completed, failed = run_sequential_training(script_paths)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total scripts: {len(script_paths)}")
    logger.info(f"Completed successfully: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {duration/3600:.2f} hours")
    logger.info(f"Average time per model: {duration/len(script_paths)/60:.1f} minutes")
    
    if failed == 0:
        logger.info("🎉 All training jobs completed successfully!")
        return 0
    else:
        logger.warning(f"⚠️  {failed} training jobs failed")
        return 1

if __name__ == "__main__":
    exit(main()) 