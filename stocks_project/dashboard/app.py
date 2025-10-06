import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Optional, Dict, Any

# ======================================
# CONFIGURAÇÕES
# ======================================
API_URL = "http://localhost:8000"  # URL da sua API FastAPI
TICKERS = ["VALE3_SA", "PETR4_SA"]  # Tickers disponíveis
DEFAULT_VALUES = {
    "VALE3_SA": {"open": 68.50, "high": 70.20, "low": 67.80, "volume": 15000000},
    "PETR4_SA": {"open": 32.45, "high": 33.10, "low": 31.90, "volume": 25000000}
}

# ======================================
# FUNÇÕES AUXILIARES
# ======================================
@st.cache_data(ttl=300)  # Cache por 5 minutos
def get_models() -> Optional[Dict[str, Any]]:
    """Busca modelos disponíveis na API com tratamento de erros"""
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"🔴 Erro ao conectar com a API: {str(e)}")
        return None

def make_prediction(ticker: str, data: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Faz predição via API com tratamento robusto"""
    try:
        response = requests.post(
            f"{API_URL}/predict/{ticker}",
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"🔴 Erro na API: {response.text}")
        return None
    except Exception as e:
        st.error(f"🔴 Falha na conexão: {str(e)}")
        return None

# ======================================
# INTERFACE STREAMLIT
# ======================================
def main():
    # Configuração da página
    st.set_page_config(
        page_title="Análise Preditiva de Ações",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS customizado
    st.markdown("""
    <style>
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar - Controles
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Stock+AI", width=150)
        st.title("⚙️ Parâmetros")
        
        ticker = st.selectbox(
            "Selecione o Ativo",
            TICKERS,
            index=0,
            help="Escolha o ticker da ação para análise"
        )

        st.subheader("Dados de Entrada")
        default = DEFAULT_VALUES.get(ticker, {})
        col1, col2 = st.columns(2)
        with col1:
            open_price = st.number_input(
                "Abertura (R$)", 
                value=default.get("open", 50.0),
                step=0.01,
                min_value=0.01
            )
            high = st.number_input(
                "Máxima (R$)", 
                value=default.get("high", 52.0),
                step=0.01,
                min_value=0.01
            )
        with col2:
            low = st.number_input(
                "Mínima (R$)", 
                value=default.get("low", 49.0),
                step=0.01,
                min_value=0.01
            )
            volume = st.number_input(
                "Volume", 
                value=default.get("volume", 1000000),
                step=1000,
                min_value=1
            )

        predict_btn = st.button(
            "📊 Calcular Previsão",
            type="primary",
            help="Clique para obter a previsão de fechamento"
        )

    # Página principal
    st.title("📈 Análise Preditiva de Ações")
    st.markdown("""
    Previsão do preço de fechamento utilizando modelos de machine learning.
    **Configure os parâmetros na sidebar** →
    """)

    # Seção de Status
    with st.expander("🔍 Status do Sistema", expanded=True):
        if st.button("🔄 Verificar Conexão com a API"):
            models = get_models()
            if models:
                st.success("✅ API conectada com sucesso!")
                st.json(models)
            else:
                st.error("❌ Falha ao conectar com a API")

    # Seção de Predição
    if predict_btn:
        with st.spinner("🔮 Calculando previsão..."):
            prediction_data = {
                "open": open_price,
                "high": high,
                "low": low,
                "volume": volume
            }
            
            result = make_prediction(ticker, prediction_data)
            
            if result:
                # Layout de resultados
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h3 style='text-align: center;'>PREVISÃO DE FECHAMENTO</h3>
                        <h1 style='text-align: center; color: #2e86c1;'>
                            R$ {:.2f}
                        </h1>
                        <p style='text-align: center;'>
                            Confiança: {:.1f}%
                        </p>
                    </div>
                    """.format(
                        result['prediction'],
                        result.get('confidence', 0) * 100 if result.get('confidence') else 'N/A'
                    ), unsafe_allow_html=True)
                    
                    st.download_button(
                        label="📥 Exportar Resultados",
                        data=pd.DataFrame([result]).to_csv(index=False),
                        file_name=f"predicao_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    # Gráfico interativo
                    plot_data = pd.DataFrame({
                        "Tipo": ["Mínima", "Abertura", "Previsão", "Máxima"],
                        "Valor (R$)": [low, open_price, result['prediction'], high]
                    })
                    
                    fig = px.bar(
                        plot_data,
                        x="Tipo",
                        y="Valor (R$)",
                        color="Tipo",
                        text="Valor (R$)",
                        color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
                    )
                    fig.update_traces(
                        texttemplate='%{text:.2f}',
                        textposition='outside'
                    )
                    fig.update_layout(
                        title=f"Comparação de Preços - {ticker}",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Detalhes técnicos
                with st.expander("📊 Detalhes Técnicos"):
                    st.json({
                        "timestamp": datetime.now().isoformat(),
                        "ticker": ticker,
                        "model": result.get("model_path", "N/A"),
                        "metrics": result.get("metrics", {}),
                        "input_parameters": prediction_data
                    })
            else:
                st.error("Não foi possível obter a previsão. Verifique os parâmetros e a conexão com a API.")

    # Seção Histórico (simulada)
    with st.expander("📅 Histórico de Previsões", expanded=False):
        st.warning("Esta funcionalidade requer integração com banco de dados.")
        if st.button("Carregar Dados Simulados"):
            dummy_data = pd.DataFrame({
                "Data": pd.date_range(end=datetime.now(), periods=30),
                "Previsão": [DEFAULT_VALUES[ticker]["open"] + i * 0.5 for i in range(30)],
                "Real": [DEFAULT_VALUES[ticker]["open"] + i * 0.55 for i in range(30)]
            })
            st.line_chart(
                dummy_data.set_index("Data"),
                color=["#2ecc71", "#3498db"]
            )

    # Rodapé
    st.markdown("---")
    st.caption(f"© {datetime.now().year} Stock AI | Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

if __name__ == "__main__":
    main()