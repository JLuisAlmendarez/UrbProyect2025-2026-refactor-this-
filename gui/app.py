import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from script.lib import CorrSystem, DataTreatments, StatHypothesisTest

# -----------------------
st.set_page_config(
    page_title="Mi Reporte Q",
    page_icon="⚙️",
    #layout="wide"
)

# -----------------------
st.title("Reporte Q")
df = pd.read_excel("data/table/BasedeDatosTransectosJoseLuis.xlsx")


# -----------------------
df_corr = df.copy()
df_corr = df_corr.drop(columns=["Ageb Manzana", "Sexo", "Edad", "Escolaridad", "¿Cuántos años tiene viviendo en este lugar?", "Ponderador"])
df_corr = DataTreatments.corr_data_handler(df_corr)
st.markdown("---")
st.header("Correlation Analysis Frame")

st.markdown("**Interpretación:**")

st.markdown("""
Este es un párrafo largo en Streamlit. 
Puedes escribir varias líneas aquí, y Streamlit mantendrá los saltos de línea si usas dobles saltos de línea entre párrafos.
""")

st.latex(r"""
E = mc^2
\quad \text{y también:} \quad
\frac{a}{b} = c
""")

CorrSystem.do(df_corr)


# -----------------------
st.markdown("---")
df_hyp = df.copy()
df_hyp = df_hyp.drop(columns=["Ageb Manzana", "Sexo", "Edad", "Escolaridad", "¿Cuántos años tiene viviendo en este lugar?", "Ponderador"])
df_hyp = DataTreatments.hypothesis_data_handler(df_hyp)

st.header("Hypothesis Testing Frame")

st.markdown("**Interpretación:**")

st.markdown("""
Este es un párrafo largo en Streamlit. 
Puedes escribir varias líneas aquí, y Streamlit mantendrá los saltos de línea si usas dobles saltos de línea entre párrafos.
""")

st.latex(r"""
E = mc^2
\quad \text{y también:} \quad
\frac{a}{b} = c
""")

questions = [col for col in df_hyp.columns if col != "Transecto"]
selected_question = st.selectbox("Selecciona una pregunta", questions)

transect_groups, group_names, groups = StatHypothesisTest.get_transect_groups(df_hyp, selected_question)
st.write(f"Se generaron {len(groups)} grupos para la pregunta: **{selected_question}**")

if st.button("Ejecutar Comparaciones"):
    results = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            g1_name, g2_name = group_names[i], group_names[j]
            key = f"{g1_name} vs {g2_name}"

            st.markdown(f"### Comparando {g1_name} vs {g2_name}")

            # Normalidad
            d1, d2, transform_type, params, is_normal = StatHypothesisTest.analize_distribution_normality(g1, g2)

            # Homogeneidad
            stat_levene, p_levene, is_homogeneous = StatHypothesisTest.analize_distributions_homogeneity(d1, d2)

            # Test estadístico
            test_results = StatHypothesisTest.execute_statistical_test(
                d1, d2, is_normal, is_homogeneous,
                transformacion=transform_type, params=params
            )

            results[key] = (d1, d2, transform_type, test_results)
