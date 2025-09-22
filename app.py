import os
import streamlit as st

# -----------------------
# Configuración interna de tema
streamlit_folder = os.path.join(os.path.dirname(__file__), ".streamlit")
os.makedirs(streamlit_folder, exist_ok=True)

config_file = os.path.join(streamlit_folder, "config.toml")
if not os.path.exists(config_file):
    with open(config_file, "w") as f:
        f.write("""
[theme]
base="light"
primaryColor="#1f77b4"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f0f2f6"
textColor="#000000"
font="sans serif"
""")

# -----------------------
# Tu app de Streamlit
st.title("Mi Primera App de Streamlit")
st.write("¡Hola! Esta es tu app en modo claro.")

# Subida de archivo CSV
uploaded_file = st.file_uploader("Elige un CSV", type="csv")
if uploaded_file:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Leer CSV
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    # Selección de columna
    columna = st.selectbox("Selecciona columna para graficar", df.columns)

    # Gráfica
    plt.figure(figsize=(6, 4))
    sns.histplot(df[columna], kde=True)
    st.pyplot(plt)