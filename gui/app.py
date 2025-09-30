import streamlit as st

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