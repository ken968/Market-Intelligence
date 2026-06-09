import re

file_path = 'd:/Market-Intelligence/utils/data_store.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add Structural Validation Gates inside append_dataframe or sync_csv_to_db
new_validation = '''    def _validate_data(self, df: pd.DataFrame) -> bool:
        if df.isnull().values.any():
            print("Validation Failed: NaN values detected.")
            return False
        # Check for price > 0 if price col exists (assuming first col is price or we check numeric cols)
        for col in df.select_dtypes(include='number').columns:
            if 'price' in col.lower() or col in ['Gold', 'BTC', 'SPY', 'QQQ', 'DIA']:
                if (df[col] <= 0).any():
                    print(f"Validation Failed: Zero or negative price in {col}.")
                    return False
        return True

    def append_dataframe(self, table_name: str, df: pd.DataFrame) -> bool:
        if not self._validate_data(df):
            return False'''

content = content.replace('    def append_dataframe(self, table_name: str, df: pd.DataFrame) -> bool:', new_validation)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated data_store.py')
