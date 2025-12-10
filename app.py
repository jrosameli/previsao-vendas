import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Configuração da Página ---
st.set_page_config(page_title="Chronos Forecaster", layout="wide")

st.title("📈 Chronos Retail Forecaster")
st.markdown("""
Esta aplicação utiliza um modelo **SARIMA (Sazonal Auto-Regressivo)** para prever vendas.
O modelo busca padrões semanais (sazonalidade de 7 dias) nos seus dados.
""")

# --- Barra Lateral para Upload ---
st.sidebar.header("1. Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Arraste seu arquivo CSV aqui", type=["csv"])
st.sidebar.markdown("**Formato esperado:** Duas colunas (Data, Vendas).")

# --- Lógica Principal ---
if uploaded_file is not None:
    try:
        # Leitura dos dados
        df = pd.read_csv(uploaded_file)
        
        # Tentativa de identificar colunas automaticamente
        # Assume que a 1ª é data e a 2ª é valor, se não tiver nomes específicos
        col_data = df.columns[0]
        col_valor = df.columns[1]
        
        df[col_data] = pd.to_datetime(df[col_data])
        df = df.set_index(col_data).sort_index()
        
        # Garante frequência diária (Preenche buracos se houver)
        df = df.asfreq('D')
        df[col_valor] = df[col_valor].ffill() # Preenche vazios com valor anterior
        
        st.subheader("Visualização dos Dados Históricos")
        st.line_chart(df[col_valor])

        # --- Parâmetros de Previsão ---
        st.sidebar.header("2. Configuração")
        dias_previsao = st.sidebar.slider("Dias para prever:", min_value=7, max_value=90, value=30)
        
        if st.sidebar.button("🚀 Gerar Previsão"):
            with st.spinner('Treinando modelo SARIMA (pode levar alguns segundos)...'):
                
                # --- O Motor Matemático (Mesma lógica do RetailForecaster) ---
                # SARIMA(1,1,1)(1,1,0,7)
                model = SARIMAX(
                    df[col_valor],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 0, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                
                # Previsão
                forecast = results.get_forecast(steps=dias_previsao)
                pred_mean = forecast.predicted_mean
                conf_int = forecast.conf_int(alpha=0.05)
                
                # --- Exibição dos Resultados ---
                st.success("Previsão concluída!")
                
                # Gráfico com Matplotlib para controle total
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plota histórico recente (últimos 90 dias para não poluir)
                historico_recente = df[col_valor].tail(90)
                ax.plot(historico_recente.index, historico_recente, label='Histórico Recente', color='black')
                
                # Plota Previsão
                ax.plot(pred_mean.index, pred_mean, label='Previsão', color='blue', linestyle='--')
                
                # Plota Intervalo de Confiança (Sombra)
                ax.fill_between(conf_int.index, 
                                conf_int.iloc[:, 0], 
                                conf_int.iloc[:, 1], 
                                color='blue', alpha=0.1, label='Intervalo de Confiança (95%)')
                
                ax.set_title(f"Previsão de Vendas - Próximos {dias_previsao} dias")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Tabela de dados
                st.subheader("Dados Previstos")
                result_df = pd.DataFrame({
                    "Previsão": pred_mean,
                    "Mínimo Esperado": conf_int.iloc[:, 0],
                    "Máximo Esperado": conf_int.iloc[:, 1]
                })
                st.dataframe(result_df)
                
                # Botão de Download
                csv = result_df.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Baixar Previsão em CSV",
                    csv,
                    "previsao_vendas.csv",
                    "text/csv",
                    key='download-csv'
                )

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}. Verifique se o CSV tem datas válidas e números.")

else:
    st.info("Aguardando upload do arquivo CSV na barra lateral...")