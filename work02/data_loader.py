# data_loader.py
import pandas as pd
from typing import Optional

AIRPORT_COLS = [
    "airport_id", "name", "city", "country", "iata", "icao",
    "latitude", "longitude", "altitude", "timezone", "dst",
    "tz_db", "type", "source"
]

class DataLoader:
    """Загрузка airports.dat (OpenFlights) из URL или локального пути.
       Автоматически конвертирует '\\N' в NaN и приводит колонки к удобным типам.
    """

    @staticmethod
    def load_airports(path_or_url: str) -> pd.DataFrame:
        if not isinstance(path_or_url, str):
            raise TypeError("path_or_url должен быть строкой (URL или локальный путь).")
        try:
            # файл не содержит заголовка, поэтому передаём names
            df = pd.read_csv(
                path_or_url,
                header=None,
                names=AIRPORT_COLS,
                na_values=["\\N"],
                keep_default_na=True,
                dtype=str,  # сначала всё как строки — потом приведём нужное
                encoding='utf-8'
            )
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке: {e}")

        # Приводим числовые поля
        for col in ["latitude", "longitude", "altitude", "timezone"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df