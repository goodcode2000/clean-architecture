#!/usr/bin/env python3
"""
Test script to verify 90-day data collection works correctly.
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.collector import BTCDataCollector
from loguru import logger

def test_data_collection():
    """Test collecting 90 days of historical data."""
    print("="*60)
    print("TESTING 90-DAY DATA COLLECTION")
    print("="*60)
    
    collector = BTCDataCollector()
    
    # Calculate expected intervals
    intervals_needed = (90 * 24 * 60) // 5  # 90 days, 5-minute intervals
    print(f"\nExpected intervals for 90 days: {intervals_needed}")
    print(f"Expected chunks (1000 per chunk): {(intervals_needed + 999) // 1000}")
    
    print("\nFetching historical data...")
    print("This may take a few minutes...\n")
    
    df = collector.fetch_historical_data(force_refresh=True)
    
    if df is not None:
        print("\n" + "="*60)
        print("✅ DATA COLLECTION SUCCESSFUL!")
        print("="*60)
        print(f"Total records collected: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Days covered: {(df['timestamp'].max() - df['timestamp'].min()).days}")
        print("\nFirst few records:")
        print(df.head())
        print("\nLast few records:")
        print(df.tail())
        return True
    else:
        print("\n" + "="*60)
        print("❌ DATA COLLECTION FAILED!")
        print("="*60)
        return False

if __name__ == "__main__":
    success = test_data_collection()
    sys.exit(0 if success else 1)
