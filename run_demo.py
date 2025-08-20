#!/usr/bin/env python3
"""
Quick start script for running the checkpoint evaluation demo
"""

import sys
import os
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['torch', 'transformers', 'pyyaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_checkpoint_exists(checkpoint_path):
    """Check if checkpoint directory exists"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint directory not found: {checkpoint_path}")
        return False
    
    # Check for essential files
    essential_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
    for file in essential_files:
        file_path = os.path.join(checkpoint_path, file)
        if not os.path.exists(file_path):
            logger.warning(f"Essential file not found: {file_path}")
    
    return True

def check_data_exists(data_path):
    """Check if QA data exists"""
    if not os.path.exists(data_path):
        logger.error(f"Data directory not found: {data_path}")
        return False
    
    # Check for QA files
    qa_files = [f for f in os.listdir(data_path) if f.startswith('QA') and f.endswith('.md')]
    if not qa_files:
        logger.error(f"No QA*.md files found in {data_path}")
        return False
    
    logger.info(f"Found {len(qa_files)} QA files: {qa_files}")
    return True

def run_demo():
    """Run the demo evaluation"""
    logger.info("Starting checkpoint evaluation demo...")
    
    # Check requirements
    if not check_requirements():
        return False
    
    # Check checkpoint
    checkpoint_path = "enhanced-qwen3-finetuned/checkpoint-450"
    if not check_checkpoint_exists(checkpoint_path):
        return False
    
    # Check data
    data_path = "data/raw"
    if not check_data_exists(data_path):
        return False
    
    # Run the demo
    try:
        logger.info("Running demo evaluation...")
        result = subprocess.run([
            sys.executable, "demo_checkpoint_evaluation.py",
            "--checkpoint", checkpoint_path,
            "--max-samples", "10",  # Start with fewer samples for quick demo
            "--output", "demo_evaluation_report.json"
        ], check=True, capture_output=True, text=True)
        
        logger.info("Demo completed successfully!")
        logger.info("Check demo_evaluation_report.json for detailed results")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Demo failed with error: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False

def main():
    """Main function"""
    print("="*60)
    print("Enhanced Qwen3 Checkpoint-450 Evaluation Demo")
    print("="*60)
    
    success = run_demo()
    
    if success:
        print("\n✅ Demo completed successfully!")
        print("📊 Check demo_evaluation_report.json for detailed results")
    else:
        print("\n❌ Demo failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()