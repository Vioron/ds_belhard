# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualization:
    def __init__(self):
        self.plots = []

    def histogram(self, df: pd.DataFrame, column: str, bins: int = 30, sample: int = None):
        if column not in df.columns:
            raise ValueError(f"Нет столбца {column}")
        data = df[column].dropna()
        if sample and sample < len(data):
            data = data.sample(sample, random_state=42)
        if data.empty:
            raise ValueError("Нет данных для гистограммы")
        plt.figure(figsize=(8,5))
        plt.hist(data, bins=bins)
        plt.title(f"Гистограмма: {column}")
        plt.xlabel(column)
        plt.ylabel("Частота")
        plt.grid(True)
        plt.show()
        self.plots.append(f"hist_{column}")

    def scatter_world(self, df: pd.DataFrame, lon_col='longitude', lat_col='latitude', color_by: str = None, sample: int = 5000):
        if lon_col not in df.columns or lat_col not in df.columns:
            raise ValueError("Нет координатных столбцов")
        tmp = df[[lon_col, lat_col]].dropna()
        if color_by and color_by in df.columns:
            tmp[color_by] = df.loc[tmp.index, color_by]
        if sample and len(tmp) > sample:
            tmp = tmp.sample(sample, random_state=42)
        plt.figure(figsize=(12,6))
        if color_by and color_by in tmp.columns:
            # если категориальная, ограничим уникальные значения
            sns.scatterplot(x=lon_col, y=lat_col, hue=color_by, data=tmp, alpha=0.7, s=10, legend='brief')
        else:
            plt.scatter(tmp[lon_col], tmp[lat_col], s=8, alpha=0.6)
        plt.title("Карта (проекция): широта/долгота аэропортов (сырой scatter)")
        plt.xlabel("Долгота")
        plt.ylabel("Широта")
        plt.grid(True)
        plt.show()
        self.plots.append("scatter_world")

    def countplot_top_countries(self, df: pd.DataFrame, country_col='country', top_n=10):
        if country_col not in df.columns:
            raise ValueError("Нет столбца country")
        top = df[country_col].value_counts().nlargest(top_n)
        plt.figure(figsize=(10,5))
        sns.barplot(x=top.index, y=top.values)
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Топ {top_n} стран по числу объектов в датасете")
        plt.ylabel("Количество")
        plt.xlabel("Страна")
        plt.show()
        self.plots.append("count_countries")
