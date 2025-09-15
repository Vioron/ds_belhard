# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_features_for_iata(df: pd.DataFrame, top_countries: int = 10) -> (pd.DataFrame, pd.Series):
    """
    Создаёт целевую переменную has_iata (1 если iata не пустой), простую фичу-таблицу:
     - numeric: latitude, longitude, altitude
     - country: топ N как one-hot, остальные -> 'Other'
    """
    if 'iata' not in df.columns:
        raise ValueError("Столбца 'iata' нет в DataFrame")

    df2 = df.copy()
    df2['has_iata'] = (~df2['iata'].isnull()) & (df2['iata'].str.strip() != "")
    y = df2['has_iata'].astype(int)

    # numeric
    X = df2[['latitude', 'longitude', 'altitude']].copy()

    # country -> top N
    if 'country' in df2.columns:
        top = df2['country'].value_counts().nlargest(top_countries).index
        df2['country_top'] = df2['country'].where(df2['country'].isin(top), other='Other')
        country_dummies = pd.get_dummies(df2['country_top'], prefix='country', drop_first=True)
        X = pd.concat([X, country_dummies], axis=1)

    # fill numeric NaNs with median
    for col in X.select_dtypes(include='number').columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)

    return X, y

def train_and_evaluate_has_iata(df, test_size=0.25, random_state=42):
    X, y = prepare_features_for_iata(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['no_iata','has_iata'], yticklabels=['no_iata','has_iata'])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.show()
    return clf
