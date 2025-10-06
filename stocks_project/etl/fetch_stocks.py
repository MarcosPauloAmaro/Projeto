import yfinance as yf
import sqlite3
import os

# Configurações
DB_PATH = 'database/stocks.db'
TICKERS = ["VALE3.SA", "PETR4.SA"]
PERIOD = "1mo"
INTERVAL = "1d"

def create_database():
    """Cria o banco de dados sem estrutura fixa"""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        # Apenas cria o arquivo do banco de dados
        open(DB_PATH, 'a').close()
        print("Banco de dados pronto para receber dados.")
    except Exception as e:
        print(f"Erro ao criar banco de dados: {str(e)}")
        raise

def save_raw_data(df, ticker):
    """Salva os dados exatamente como vieram do yfinance"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        # Adiciona o ticker como coluna
        df['Ticker'] = ticker
        # Salva os dados brutos
        df.to_sql(ticker.replace('.', '_'), conn, if_exists='replace', index=True)
        print(f"Dados brutos de {ticker} salvos com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar dados de {ticker}: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def fetch_stock_data(ticker):
    """Busca dados do yfinance sem modificações"""
    try:
        print(f"Baixando dados brutos para {ticker}...")
        df = yf.download(
            ticker, 
            period=PERIOD, 
            interval=INTERVAL,
            progress=False
        )
        return df if not df.empty else None
    except Exception as e:
        print(f"Erro ao baixar dados de {ticker}: {str(e)}")
        return None

def main():
    # Prepara o banco de dados
    create_database()
    
    # Baixa e salva os dados para cada ticker
    for ticker in TICKERS:
        try:
            df = fetch_stock_data(ticker)
            if df is not None:
                save_raw_data(df, ticker)
        except Exception as e:
            print(f"Falha ao processar {ticker}: {str(e)}")
    
    print("Processo concluído. Dados salvos exatamente como vieram do yfinance.")

if __name__ == "__main__":
    main()