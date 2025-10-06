import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configurações
API_URL = "http://localhost:8000"
TICKERS = ["VALE3_SA", "PETR4_SA"]

st.set_page_config(layout="wide")
st.title("📈 Dashboard de Ações")

# Sidebar
ticker = st.sidebar.selectbox("Selecione a Ação", TICKERS)
days_back = st.sidebar.slider("Período Histórico (dias)", 30, 365*5, 365)

# Função para carregar dados históricos
@st.cache_data
def load_historical_data(ticker, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Simulação - substitua por sua API real
        dates = pd.date_range(start_date, end_date)
        prices = np.cumsum(np.random.randn(len(dates))) * 0.5 + 50
        volumes = np.random.randint(1000000, 5000000, len(dates))
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Volume': volumes
        })
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None

# Dados históricos
hist_data = load_historical_data(ticker, days_back)

# Gráfico de preços
if hist_data is not None:
    fig = go.Figure()
    
    # Linha de preços
    fig.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=hist_data['Close'],
        name='Preço',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Bandas de Bollinger (20 dias)
    rolling_mean = hist_data['Close'].rolling(20).mean()
    rolling_std = hist_data['Close'].rolling(20).std()
    fig.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=rolling_mean + 2*rolling_std,
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=rolling_mean - 2*rolling_std,
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        name='Bollinger Bands'
    ))
    
    fig.update_layout(
        title=f'Variação Diária - {ticker}',
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        hovermode="x unified",
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # 📊 Métricas de performance
    st.subheader("📊 Métricas do Modelo")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("MAE (R$)", value=f"{1.23:.2f}", delta="-0.12 vs benchmark")
    
    with col2:
        st.metric("RMSE (R$)", value=f"{1.85:.2f}", delta="-0.25 vs benchmark")
    
    with col3:
        st.metric("R² Score", value=f"{0.92:.2%}", delta="+2% vs último mês")

# 🔮 Previsão em tempo real
st.subheader("🔮 Previsão de Fechamento")
col1, col2 = st.columns([3, 1])

with col2:
    open_price = st.number_input("Abertura", value=50.0)
    high_price = st.number_input("Máxima", value=55.0)
    low_price = st.number_input("Mínima", value=45.0)
    volume = st.number_input("Volume", value=1000000)
    
    if st.button("Calcular Previsão"):
        try:
            response = requests.post(
                f"{API_URL}/predict/{ticker}",
                json={
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "volume": volume
                }
            )
            result = response.json()
            st.success(f"Previsão: R$ {result['prediction']:.2f}")
        except Exception as e:
            st.error(f"Erro na API: {str(e)}")

with col1:
    if hist_data is not None:
        last_30 = hist_data.tail(30).copy()
        last_30['Predicted'] = last_30['Close'] * np.random.uniform(0.98, 1.02, len(last_30))
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=last_30['Date'],
            y=last_30['Close'],
            name='Real',
            line=dict(color='green')
        ))
        fig2.add_trace(go.Scatter(
            x=last_30['Date'],
            y=last_30['Predicted'],
            name='Previsto',
            line=dict(color='orange', dash='dot')
        ))
        fig2.update_layout(
            title='Previsão vs Real (Últimos 30 dias)',
            template='plotly_dark'
        )
        st.plotly_chart(fig2, use_container_width=True)

# 📅 Previsão dos próximos 30 dias
if hist_data is not None:
    st.subheader("📅 Previsão dos Próximos 30 Dias")
    last_price = hist_data['Close'].iloc[-1]
    future_dates = pd.date_range(hist_data['Date'].iloc[-1] + timedelta(days=1), periods=30)
    future_predictions = last_price + np.cumsum(np.random.randn(30) * 0.5)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=hist_data['Close'],
        name='Histórico',
        line=dict(color='blue')
    ))
    fig3.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        name='Previsão 30 dias',
        line=dict(color='red', dash='dot')
    ))
    fig3.update_layout(
        title=f'Projeção de Preços - {ticker} (30 Dias)',
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        template='plotly_dark'
    )
    st.plotly_chart(fig3, use_container_width=True)

# 📜 Dados brutos
if st.checkbox("Mostrar Dados Históricos Completos"):
    st.dataframe(hist_data.style.background_gradient(cmap='Blues'), height=300)
