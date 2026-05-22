"""
DuckDB One-time Database Migration Script
Reads all CSV files in the data directory and loads them into DuckDB tables.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_store import MarketDataStore


def main():
    print("=" * 60)
    print("STARTING DUCKDB MIGRATION")
    print("=" * 60)
    
    store = MarketDataStore()
    print(f"Database target: {store.db_path}\n")
    
    migrated_count = store.migrate_all_csvs()
    
    print("\n" + "=" * 60)
    print(f"MIGRATION COMPLETE: {migrated_count} tables populated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
