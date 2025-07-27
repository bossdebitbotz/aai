#!/usr/bin/env python3
"""
Complete Maximum Dataset Pipeline
Monitors extraction progress and automatically runs V5 data preparation 
when the final 60-day extraction is complete.
"""

import os
import time
import subprocess
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expected final state
EXPECTED_TOTAL_DAYS = 60
EXPECTED_JULY_DAYS = 16  # All July days 01-04, 13-24
TARGET_DAYS = [
    # All the days we should have when complete
    '2025-05-18', '2025-05-19', '2025-05-20', '2025-05-21', '2025-05-22',
    '2025-05-23', '2025-05-24', '2025-05-25', '2025-05-26', '2025-05-27',
    '2025-05-28', '2025-05-29', '2025-05-30', '2025-05-31', '2025-06-01',
    '2025-06-02', '2025-06-03', '2025-06-04', '2025-06-05', '2025-06-06',
    '2025-06-07', '2025-06-08', '2025-06-09', '2025-06-10', '2025-06-11',
    '2025-06-12', '2025-06-13', '2025-06-14', '2025-06-15', '2025-06-16',
    '2025-06-17', '2025-06-18', '2025-06-19', '2025-06-20', '2025-06-21',
    '2025-06-22', '2025-06-23', '2025-06-24', '2025-06-25', '2025-06-26',
    '2025-06-27', '2025-06-28', '2025-06-29', '2025-06-30', '2025-07-01',
    '2025-07-02', '2025-07-03', '2025-07-04', '2025-07-13', '2025-07-14',
    '2025-07-15', '2025-07-16', '2025-07-17', '2025-07-18', '2025-07-19',
    '2025-07-20', '2025-07-21', '2025-07-22', '2025-07-23', '2025-07-24'
]

def check_extraction_progress():
    """Check current extraction progress."""
    try:
        extracted_dirs = [d for d in os.listdir('data/full_lob_data') 
                         if d.startswith('2025-') and os.path.isdir(f'data/full_lob_data/{d}')]
        
        total_days = len(extracted_dirs)
        july_days = len([d for d in extracted_dirs if d.startswith('2025-07')])
        
        # Check completion
        missing_days = set(TARGET_DAYS) - set(extracted_dirs)
        
        return {
            'total_days': total_days,
            'july_days': july_days,
            'missing_days': sorted(list(missing_days)),
            'completion_percentage': (total_days / EXPECTED_TOTAL_DAYS) * 100,
            'is_complete': len(missing_days) == 0
        }
    except Exception as e:
        logger.error(f"Error checking progress: {e}")
        return None

def count_parquet_files():
    """Count total parquet files."""
    try:
        resampled_dir = "data/full_lob_data/resampled_5s"
        if os.path.exists(resampled_dir):
            return len([f for f in os.listdir(resampled_dir) if f.endswith('.parquet')])
        return 0
    except:
        return 0

def monitor_extraction(check_interval=300):  # Check every 5 minutes
    """Monitor extraction progress."""
    logger.info("🔍 MONITORING EXTRACTION PROGRESS...")
    logger.info(f"⏱️ Checking every {check_interval//60} minutes")
    logger.info(f"🎯 Target: {EXPECTED_TOTAL_DAYS} days with {EXPECTED_JULY_DAYS} July days")
    logger.info("")
    
    start_time = datetime.now()
    last_total = 0
    
    while True:
        progress = check_extraction_progress()
        if not progress:
            time.sleep(check_interval)
            continue
        
        current_time = datetime.now()
        elapsed = current_time - start_time
        
        # Progress update
        if progress['total_days'] != last_total:
            logger.info(f"📊 PROGRESS UPDATE:")
            logger.info(f"   Total days: {progress['total_days']}/{EXPECTED_TOTAL_DAYS} ({progress['completion_percentage']:.1f}%)")
            logger.info(f"   July days: {progress['july_days']}/{EXPECTED_JULY_DAYS}")
            logger.info(f"   Missing: {len(progress['missing_days'])} days")
            logger.info(f"   Elapsed: {elapsed.total_seconds()/60:.0f} minutes")
            
            if len(progress['missing_days']) <= 5:
                logger.info(f"   Remaining: {progress['missing_days']}")
            
            logger.info("")
            last_total = progress['total_days']
        
        # Check if complete
        if progress['is_complete']:
            parquet_count = count_parquet_files()
            logger.info("🎉 EXTRACTION COMPLETE!")
            logger.info(f"✅ Final stats: {progress['total_days']} days, {parquet_count} parquet files")
            logger.info(f"⏱️ Total extraction time: {elapsed.total_seconds()/60:.0f} minutes")
            return True
        
        time.sleep(check_interval)

def run_v5_preparation():
    """Run the V5 data preparation for maximum sequences."""
    logger.info("🚀 STARTING V5 DATA PREPARATION...")
    logger.info("🎯 Target: 50,000+ sequences from 60-day dataset")
    logger.info("")
    
    try:
        # Run V5 data preparation
        start_time = datetime.now()
        result = subprocess.run(
            ['python3', 'prepare_attention_model_data_v5.py'],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        elapsed = datetime.now() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ V5 PREPARATION COMPLETED in {elapsed.total_seconds()/60:.1f} minutes!")
            logger.info("🎯 Checking final sequence count...")
            
            # Check final dataset stats
            try:
                with open('data/final_attention/dataset_stats.json', 'r') as f:
                    stats = json.load(f)
                    
                final_sequences = stats.get('total_sequences', 0)
                logger.info(f"🎉 FINAL RESULT: {final_sequences:,} sequences!")
                logger.info(f"📈 Improvement: {final_sequences/5720:.1f}x over original 5,720 sequences")
                
                if final_sequences >= 50000:
                    logger.info("🏆 TARGET ACHIEVED: 50,000+ sequences!")
                elif final_sequences >= 15000:
                    logger.info("🎯 EXCELLENT RESULT: 15,000+ sequences (3x improvement)")
                else:
                    logger.info("⚠️ Lower than expected - investigating...")
                    
            except Exception as e:
                logger.error(f"Error reading final stats: {e}")
            
            return True
        else:
            logger.error(f"❌ V5 preparation failed:")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ V5 preparation timed out after 2 hours")
        return False
    except Exception as e:
        logger.error(f"❌ Error running V5 preparation: {e}")
        return False

def create_final_summary():
    """Create final completion summary."""
    try:
        progress = check_extraction_progress()
        parquet_count = count_parquet_files()
        
        # Get sequence count if available
        sequence_count = 0
        try:
            with open('data/final_attention/dataset_stats.json', 'r') as f:
                stats = json.load(f)
                sequence_count = stats.get('total_sequences', 0)
        except:
            pass
        
        summary = {
            'completion_date': datetime.now().isoformat(),
            'status': 'MAXIMUM DATASET COMPLETE',
            'extraction_stats': {
                'total_days': progress['total_days'] if progress else 0,
                'july_days': progress['july_days'] if progress else 0,
                'total_parquet_files': parquet_count
            },
            'training_data': {
                'final_sequences': sequence_count,
                'improvement_factor': f"{sequence_count/5720:.1f}x" if sequence_count > 0 else "TBD",
                'original_sequences': 5720
            },
            'ready_for_training': sequence_count > 0,
            'next_steps': [
                'Deploy to H100 infrastructure',
                'Train all 16 model variants',
                'Monitor training across 4 GPUs'
            ]
        }
        
        with open('maximum_dataset_completion_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("📊 FINAL SUMMARY CREATED:")
        logger.info(f"   Days: {summary['extraction_stats']['total_days']}")
        logger.info(f"   Parquet files: {summary['extraction_stats']['total_parquet_files']}")
        logger.info(f"   Sequences: {summary['training_data']['final_sequences']:,}")
        logger.info(f"   Improvement: {summary['training_data']['improvement_factor']}")
        logger.info("📂 Summary saved to: maximum_dataset_completion_summary.json")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error creating final summary: {e}")
        return None

def main():
    """Main completion pipeline."""
    logger.info("🚀 MAXIMUM DATASET COMPLETION PIPELINE")
    logger.info("📊 Step 1: Monitor extraction progress")
    logger.info("🔄 Step 2: Run V5 data preparation when complete")
    logger.info("📋 Step 3: Create final summary")
    logger.info("")
    
    # Step 1: Monitor extraction
    if monitor_extraction():
        logger.info("✅ Extraction monitoring complete")
    else:
        logger.error("❌ Extraction monitoring failed")
        return
    
    # Step 2: Run V5 preparation
    if run_v5_preparation():
        logger.info("✅ V5 data preparation complete")
    else:
        logger.error("❌ V5 data preparation failed")
        return
    
    # Step 3: Create final summary
    summary = create_final_summary()
    if summary:
        logger.info("✅ Final summary created")
    else:
        logger.error("❌ Final summary creation failed")
    
    logger.info("")
    logger.info("🎉 MAXIMUM DATASET PIPELINE COMPLETE!")
    logger.info("🚀 Ready for Monday H100 training!")

if __name__ == "__main__":
    main() 