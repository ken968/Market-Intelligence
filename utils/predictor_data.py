import pandas as pd
import os

class DataHandler:
    """Handles loading and parsing of historical data for prediction."""
    
    def __init__(self, asset_key: str, config: dict):
        self.asset_key = asset_key
        self.config = config
        self.data = None
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load historical data from DuckDB with fallback to CSV"""
        data_path = self.config['data_file']
        table_name = os.path.splitext(os.path.basename(data_path))[0].lower()
        
        from utils.data_store import MarketDataStore
        store = MarketDataStore()
        
        df = None
        try:
            df = store.read_table(table_name, format='pandas')
        except Exception as e:
            print(f"Warning: Could not read table '{table_name}' from DuckDB: {e}. Falling back to CSV.")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data not found in database or CSV: {data_path}")
            df = pd.read_csv(data_path)
        
        # Ensure all required features exist
        features = self.config['features']
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features in {self.asset_key}: {missing_features}")
            
        self.data = df[features].values
        self.df = df
        return df
        
    def get_latest_price(self) -> float:
        """Get the most recent actual price"""
        if self.df is None:
            self.load_data()
        
        # First feature is always the price
        return float(self.df[self.config['features'][0]].iloc[-1])
