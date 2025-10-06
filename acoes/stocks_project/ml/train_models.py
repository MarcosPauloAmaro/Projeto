import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump
import os
import re
from pathlib import Path

# Configurações
DB_PATH = 'database/stocks.db'
MODEL_DIR = 'ml/models'
os.makedirs(MODEL_DIR, exist_ok=True)
TICKERS = ["VALE3_SA", "PETR4_SA"]

def extract_tuple_column(df, col_name):
    """Extrai valores de colunas no formato ('Nome', 'Ticker')"""
    for col in df.columns:
        if isinstance(col, str) and col_name.lower() in col.lower():
            return df[col]
    return None

def load_and_prepare_data(ticker):
    """Carrega e prepara os dados com tratamento para formato de tupla"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT * FROM {ticker}"
        df = pd.read_sql(query, conn)
        conn.close()

        print(f"\nColunas brutas em {ticker}: {df.columns.tolist()}")

        # Extrai colunas do formato especial
        df_clean = pd.DataFrame()
        df_clean['open'] = extract_tuple_column(df, 'Open')
        df_clean['high'] = extract_tuple_column(df, 'High')
        df_clean['low'] = extract_tuple_column(df, 'Low')
        df_clean['close'] = extract_tuple_column(df, 'Close')
        df_clean['volume'] = extract_tuple_column(df, 'Volume')

        # Verifica se todas as colunas essenciais existem
        if df_clean['close'] is None:
            raise ValueError(f"Coluna 'Close' não encontrada em {ticker}")

        # Preenche valores faltantes
        for col in ['open', 'high', 'low']:
            if df_clean[col] is None:
                df_clean[col] = df_clean['close']

        if df_clean['volume'] is None:
            df_clean['volume'] = 1  # Valor padrão

        return df_clean.dropna()

    except Exception as e:
        print(f"Erro ao processar {ticker}: {str(e)}")
        return None

def train_model(ticker):
    """Treina e salva o modelo"""
    print(f"\nProcessando {ticker}...")
    
    df = load_and_prepare_data(ticker)
    if df is None or df.empty:
        print(f"Dados insuficientes para {ticker}")
        return None

    X = df[['open', 'high', 'low', 'volume']]
    y = df['close']

    # Divisão treino-teste (sem shuffle para dados temporais)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Avaliação
    preds = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, preds),
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'r2': model.score(X_test, y_test)
    }

    # Salva modelo
    model_path = os.path.join(MODEL_DIR, f'model_{ticker}.joblib')
    dump(model, model_path)
    
    print(f"Modelo treinado para {ticker}")
    print(f"Métricas: {metrics}")
    print(f"Salvo em: {model_path}\n")
    return model_path

def main():
    """Executa o treinamento para todos os tickers"""
    print("Iniciando treinamento de modelos...")
    
    for ticker in TICKERS:
        train_model(ticker)

    print("\nProcesso concluído. Verifique os modelos em:", MODEL_DIR)

if __name__ == "__main__":
    main()