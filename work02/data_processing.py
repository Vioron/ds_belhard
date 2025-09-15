# data_processing.py
import pandas as pd
from typing import List, Optional

class DataProcessing:
    def _ensure_df(self, df):
        if not hasattr(df, 'shape') or not hasattr(df, 'columns'):
            raise TypeError("Ожидается pandas DataFrame")

    def count_missing(self, df: pd.DataFrame) -> pd.Series:
        self._ensure_df(df)
        return df.isnull().sum()

    def report_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_df(df)
        total = len(df)
        if total == 0:
            return pd.DataFrame(columns=['missing_count', 'missing_percent'])
        miss = df.isnull().sum()
        report = pd.DataFrame({
            'missing_count': miss,
            'missing_percent': (miss / total) * 100
        })
        report = report[report['missing_count'] > 0].sort_values('missing_percent', ascending=False)
        return report

    def fill_missing(self, df: pd.DataFrame, method: str = "median", columns: Optional[List[str]] = None, constant_value=None) -> pd.DataFrame:
        self._ensure_df(df)
        if columns is None:
            columns = df.columns.tolist()
        out = df.copy()
        allowed = {'mean', 'median', 'mode', 'constant'}
        if method not in allowed:
            raise ValueError(f"Неизвестный метод {method}")
        for col in columns:
            if col not in out.columns:
                continue
            if out[col].isnull().sum() == 0:
                continue
            if method == 'mean' and pd.api.types.is_numeric_dtype(out[col].dtype):
                fill = out[col].mean()
            elif method == 'median' and pd.api.types.is_numeric_dtype(out[col].dtype):
                fill = out[col].median()
            elif method == 'mode':
                m = out[col].mode()
                fill = m.iloc[0] if len(m) > 0 else (0 if pd.api.types.is_numeric_dtype(out[col].dtype) else "")
            elif method == 'constant':
                fill = 0 if constant_value is None else constant_value
            else:
                # fallback
                m = out[col].mode()
                fill = m.iloc[0] if len(m) > 0 else None
            out[col].fillna(fill, inplace=True)
        return out
