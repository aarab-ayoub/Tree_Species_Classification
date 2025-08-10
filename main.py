#!/usr/bin/env python3
"""
MAIN.PY - Project Entry Point
=============================
🎯 PURPOSE: Main script to run your tree species classification
🔧 USAGE: python main.py
📊 SHOWS: Dataset statistics and project status
"""

import sys
from pathlib import Path

# Add src directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from dataset_explorer import show_dataset_statistics, create_dataset_loader

def main():
    """Main function for the tree species classification project"""
    print(" Tree Species Classification Project")
    print("Course: Traitement des données multimédia")
    print("=" * 60)
    
    # Show dataset overview
    show_dataset_statistics()
    
    print(f"\n Project Status: Data Loading & Visualization ✅ COMPLETE")
    print(f" Next Phase: Feature Extraction & Classification")
    
    # TODO: Add feature extraction and classification here
    
if __name__ == "__main__":
    main()
