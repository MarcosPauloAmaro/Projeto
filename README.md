Documentação do Projeto de Machine 
Learning: Previsão de Ações 
Este documento descreve a estrutura, funcionamento e objetivo do projeto de Machine Learning 
desenvolvido para análise e previsão de preços de ações da VALE3 e PETR4. O sistema foi dividido 
em quatro módulos principais: coleta de dados, treinamento de modelo, API e dashboard interativo. 
1 Coleta e Armazenamento de Dados 
O primeiro script utiliza a biblioteca yfinance para coletar dados históricos das ações (VALE3 e 
PETR4) e armazená-los em um banco SQLite. As funções principais são: 
● create_database(): cria o arquivo stocks.db na pasta database/. 
● fetch_stock_data(ticker): baixa os dados do Yahoo Finance conforme período e intervalo 
definidos. 
● save_raw_data(df, ticker): salva os dados brutos em tabelas separadas por ticker no banco 
SQLite. 
● main(): orquestra o processo de coleta e armazenamento. 
Resultado: Banco de dados com os dados originais de cada ação. 
2 Treinamento do Modelo de Machine Learning 
O segundo módulo treina modelos de regressão Random Forest para prever o preço de fechamento 
('Close'). As principais funções são: 
● load_and_prepare_data(ticker): carrega os dados do banco, ajusta colunas e remove valores 
nulos. 
● train_model(ticker): divide o dataset, treina o modelo e calcula métricas (MAE, RMSE, R²). 
● main(): executa o treinamento para todos os tickers e salva os modelos em .joblib. 
Resultado: Modelos prontos para previsão armazenados em ml/models/. 
3 API com FastAPI 
O terceiro módulo implementa uma API usando o FastAPI. Ela carrega os modelos treinados na 
inicialização e disponibiliza endpoints para acessar as métricas de desempenho. 
Principais endpoints: 
● /model-metrics/{ticker}: retorna métricas como MAE, RMSE, R² e data do último 
treinamento. 
Resultado: API local executando em http://localhost:8000, pronta para integração com o dashboard. 
4 Dashboard com Streamlit 
O quarto módulo oferece uma interface interativa para visualização e previsão dos dados. 
Funcionalidades: 
● Seleção de ação e período histórico. 
● Visualização gráfica com Bandas de Bollinger. 
● Exibição de métricas do modelo. 
● Entrada manual para previsão via API. 
● Gráficos de comparação (real vs previsto) e projeção futura. 
Resultado: Interface visual completa para acompanhamento das ações e previsões. 
5 Fluxograma 
ENTRADA: Yahoo Finance API 
↓ 
COLETA: Script Python (yfinance) 
↓ 
ARMAZENAMENTO: Banco SQLite 
↓ 
PROCESSAMENTO: Machine Learning (80/20) 
↓ 
MODELO: Random Forest (MAE, RMSE, R²) 
↓ 
API: FastAPI (/predict, /metrics) 
↓ 
INTERFACE: Dashboard Streamlit 
↓ 
SAÍDA: Usuário (Previsões)
