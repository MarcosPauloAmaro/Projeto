# Projeto Ações — VALE e PETROBRAS (Yahoo Finance)



## O que contém


- `etl/fetch_stocks.py` — coleta histórico (5 anos) das ações e salva em `database/stocks.db`.

- `api/main.py` — FastAPI para solicitar previsões baseadas em modelos treinados.

- `ml/train_models.py` — treina modelos simples (Regressão Linear) e salva em `ml/models/`.

- `dashboard/app.py` — Streamlit dashboard para visualizar preços e pedir previsões.


## Como usar


1. Crie e ative um ambiente virtual (recomendado):


   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```


2. Instale dependências:


   ```bash
   pip install -r requirements.txt
   ```


3. Baixe dados:


   ```bash
   python etl/fetch_stocks.py
   ```


4. Treine modelos:


   ```bash
   python ml/train_models.py
   ```


5. Rode a API (em outro terminal):


   ```bash
   uvicorn api.main:app --reload
   ```


6. Rode o dashboard:


   ```bash
   streamlit run dashboard/app.py
   ```


## Dependências


Veja `requirements.txt`.

