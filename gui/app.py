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

st.subheader("**Interpretación:**")

st.markdown("""
La correlación mide la relación lineal entre dos variables. Indica si cuando una variable aumenta, la otra tiende a aumentar (correlación positiva), disminuir (correlación negativa), o no hay relación aparente (correlación cercana a cero).
\n
⚠️ __Advertencia Crítica__:
\n
* La correlación NO implica causalidad.
\n
__Dos variables pueden estar correlacionadas por:__
\n
* Causalidad directa: A causa B
\n
* Causalidad inversa: B causa A
\n
* Variable confusora: C causa tanto A como B
\n
* Coincidencia: Relación espuria sin conexión real
\n

_Ejemplo_: Puede haber correlación entre "número de ventiladores" y "contagios COVID", pero esto no significa que los ventiladores causen contagios. Ambas variables podrían estar relacionadas con una tercera: el hacinamiento.

\n
__Formula:__
\n
""")

st.latex(r"""
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
""")

CorrSystem.do(df_corr)


# -----------------------
st.markdown("---")
df_hyp = df.copy()
df_hyp = df_hyp.drop(columns=["Ageb Manzana", "Sexo", "Edad", "Escolaridad", "¿Cuántos años tiene viviendo en este lugar?", "Ponderador"])
df_hyp = DataTreatments.hypothesis_data_handler(df_hyp)

st.header("Hypothesis Testing Frame")

st.subheader("**Interpretación:**")
st.markdown("#### Supuestos:")
st.markdown("###### Normalidad")
st.markdown("""
Evalúa si los datos siguen una **distribución normal** (campana de Gauss). Utilizamos la **prueba de Shapiro-Wilk**:
- **H₀ (hipótesis nula)**: Los datos provienen de una distribución normal
- **H₁ (hipótesis alternativa)**: Los datos NO provienen de una distribución normal
- **Decisión**: Si p-value > 0.05 → Los datos son normales
Si los datos **no son normales**, se aplican transformaciones (log, Box-Cox) o se usan pruebas no paramétricas.
\n
""")
st.latex(r"""
W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}
""")
st.markdown("###### Varianza")
st.markdown("""
Evalúa si las varianzas de los grupos son similares. Utilizamos la **prueba de Levene**:

- **H₀**: Las varianzas de los grupos son iguales
- **H₁**: Las varianzas de los grupos son diferentes
- **Decisión**: Si p-value > 0.05 → Las varianzas son homogéneas

Este supuesto determina qué versión de la prueba t usar.
""")

st.latex(r"""
W = \frac{(N-k)}{(k-1)} \frac{\sum_{i=1}^{k} n_i(Z_{i.} - Z_{..})^2}{\sum_{i=1}^{k}\sum_{j=1}^{n_i}(Z_{ij} - Z_{i.})^2}
""")

st.markdown("#### Pruebas:")
st.markdown("###### Prueba t-test o t student")
st.markdown("""
Cuándo se usa: Datos normales + varianzas homogéneas

Compara las medias de dos grupos asumiendo:
- Distribución normal
- Varianzas iguales
- Muestras independientes

**Interpretación del p-value**:
- p < 0.05 → Diferencia estadísticamente significativa
- p ≥ 0.05 → No hay diferencia significativa
""")

st.latex(r"""
t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
""")

st.markdown(r"""
Donde $s_p$ es la desviación estándar combinada:
""")

st.latex(r"""
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
""")

st.markdown("###### Prueba Welch's t-test")
st.markdown("""
Cuándo se usa: Datos normales + varianzas heterogéneas

Versión modificada de la prueba t que **no asume** varianzas iguales. Más robusta cuando las varianzas difieren significativamente entre grupos.
""")

st.latex(r"""
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
""")

st.markdown("###### Mann-Whitney U Test")
st.markdown("""
Cuándo se usa: Datos NO normales (prueba no paramétrica)

Compara las **medianas** en lugar de las medias. No requiere normalidad. Ordena todos los datos y compara los rangos entre grupos.

- **Hipótesis**: Las distribuciones de los dos grupos son diferentes
- **Ventaja**: Robusta ante outliers y distribuciones asimétricas
""")

st.latex(r"""
U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1
""")

st.markdown(r"""
Donde $R_1$ es la suma de rangos del grupo 1.
""")


st.markdown("#### Diferencias:")
st.markdown("###### Estimador d de Cohen")# Cambiar a H de cohen
st.markdown("""

Mide la diferencia entre medias en unidades de desviación estándar.

**Interpretación**:
- |d| < 0.2 → Efecto trivial
- 0.2 ≤ |d| < 0.5 → Efecto pequeño
- 0.5 ≤ |d| < 0.8 → Efecto mediano
- |d| ≥ 0.8 → Efecto grande
""")

st.latex(r"""
d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}
""")

st.markdown("###### Estimador r de Rosenthal")
st.markdown("""

Convierte el estadístico U en una correlación. Interpretación similar al coeficiente de correlación.

**Interpretación**:
- |r| < 0.1 → Efecto trivial
- 0.1 ≤ |r| < 0.3 → Efecto pequeño
- 0.3 ≤ |r| < 0.5 → Efecto mediano
- |r| ≥ 0.5 → Efecto grande
""")

st.latex(r"""
r = \frac{Z}{\sqrt{N}}
""")


st.markdown("###### Estimador Hodges-Lehmann")
st.markdown("""

Estima la **diferencia mediana** entre los dos grupos. Es una medida robusta de la diferencia de localización.

- Representa la mediana de todas las diferencias posibles entre pares de observaciones de los dos grupos
- Se interpreta en las unidades originales de medición
""")

st.latex(r"""
\Delta = \text{mediana}\{x_i - y_j : i=1,\ldots,n_1; \, j=1,\ldots,n_2\}
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
