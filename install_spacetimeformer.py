#!/usr/bin/env python3

import subprocess
import sys
import os

def install_spacetimeformer():
    """Install spacetimeformer package in development mode"""
    
    # Change to spacetimeformer directory
    spacetimeformer_path = os.path.join(os.getcwd(), 'spacetimeformer')
    
    if not os.path.exists(spacetimeformer_path):
        print(f"Error: spacetimeformer directory not found at {spacetimeformer_path}")
        sys.exit(1)
    
    print(f"Installing spacetimeformer from {spacetimeformer_path}")
    
    # Install in development mode
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-e', spacetimeformer_path
        ], check=True, capture_output=True, text=True)
        
        print("Successfully installed spacetimeformer!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing spacetimeformer: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    install_spacetimeformer() 