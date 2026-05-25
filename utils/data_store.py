"""
Market Data Store Abstraction
Manages embedded DuckDB connections, data operations, and Polars/Pandas integration.
"""

from __future__ import annotations
import os
import pandas as pd

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import polars as pl
except ImportError:
    pl = None


class MarketDataStore:
    """
    Handles database operations for the Market Intelligence system.
    Uses DuckDB for fast in-process querying and storage.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Locate db in the project root's 'data' directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(project_root, 'data', 'market_intelligence.db')
        
        self.db_path = os.path.abspath(db_path)
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def get_connection(self, read_only: bool = False):
        """
        Open a connection to DuckDB.
        Use read_only=True for concurrent read queries to prevent write locks.
        """
        if duckdb is None:
            raise ImportError("DuckDB is not installed. Please install it with 'pip install duckdb polars pyarrow'.")
            
        if read_only and not os.path.exists(self.db_path):
            # If DB doesn't exist, connect once in read-write mode to create it
            conn = duckdb.connect(self.db_path, read_only=False)
            conn.close()
        
        return duckdb.connect(self.db_path, read_only=read_only)

    def read_table(self, table_name: str, format: str = 'pandas') -> pd.DataFrame | pl.DataFrame:
        """
        Read a table from DuckDB as a Pandas or Polars DataFrame.
        """
        conn = self.get_connection(read_only=True)
        try:
            rel = conn.query(f"SELECT * FROM {table_name}")
            if format == 'polars':
                return rel.pl()
            else:
                return rel.df()
        except Exception as e:
            raise ValueError(f"Failed to read table '{table_name}' from DuckDB: {e}")
        finally:
            conn.close()

    def write_table(self, table_name: str, df: pd.DataFrame | pl.DataFrame, csv_backup_path: str = None):
        """
        Write/Overwrite a table in DuckDB and optionally save a CSV backup.
        """
        # Ensure directory for CSV backup exists
        if csv_backup_path:
            os.makedirs(os.path.dirname(os.path.abspath(csv_backup_path)), exist_ok=True)

        conn = None
        db_write_failed = False
        if duckdb is not None:
            try:
                conn = self.get_connection(read_only=False)
                # Register DataFrame and create/replace table
                conn.register('temp_df', df)
                conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM temp_df")
                conn.unregister('temp_df')
            except Exception as e:
                print(f"Warning: Failed to write table '{table_name}' to DuckDB: {e}")
                db_write_failed = True
            finally:
                if conn:
                    conn.close()
        else:
            db_write_failed = True
            if not csv_backup_path:
                raise ImportError("DuckDB is not installed. Cannot write table to database.")

        # CSV backup write
        if csv_backup_path:
            try:
                if pl is not None and isinstance(df, pl.DataFrame):
                    df.write_csv(csv_backup_path)
                elif isinstance(df, pd.DataFrame):
                    # Save index if it is meaningful (not a default RangeIndex) or explicitly named
                    save_index = not isinstance(df.index, pd.RangeIndex) or df.index.name is not None
                    df.to_csv(csv_backup_path, index=save_index)
            except Exception as e:
                if db_write_failed:
                    raise ValueError(f"Failed to write table '{table_name}': both DuckDB and CSV backup failed. CSV Error: {e}")
                else:
                    print(f"Warning: DuckDB table '{table_name}' written, but CSV backup failed: {e}")

    def migrate_all_csvs(self) -> int:
        """
        Migrates all CSV files in the data directory into DuckDB tables.
        """
        data_dir = os.path.dirname(self.db_path)
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        print(f"System: Found {len(csv_files)} CSV files to migrate to DuckDB.")
        conn = self.get_connection(read_only=False)
        
        migrated = 0
        for csv_file in csv_files:
            csv_path = os.path.join(data_dir, csv_file)
            # Table name is lowercase base name without extension
            table_name = os.path.splitext(csv_file)[0].lower()
            
            # Avoid migrating backup or temp files if any
            if table_name.startswith('.'):
                continue
                
            try:
                # DuckDB read_csv_auto requires unix-style slashes
                clean_path = csv_path.replace('\\', '/')
                conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{clean_path}')")
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                print(f"  -> Migrated '{csv_file}' to table '{table_name}' ({count} rows).")
                migrated += 1
            except Exception as e:
                print(f"  [!] Failed to migrate '{csv_file}': {e}")
                
        conn.close()
        print(f"System: Migration complete. {migrated}/{len(csv_files)} tables created.")
        return migrated
