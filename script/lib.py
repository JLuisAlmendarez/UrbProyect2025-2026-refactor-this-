"""
    @staticmethod
    def get_transect_groups(df, question_col):
        transect_groups = {
            t: pd.to_numeric(df[df["Transecto"] == t][question_col], errors="coerce").dropna()
            for t in sorted(df["Transecto"].unique())
        }
        group_names = [f"Transecto {t}" for t in transect_groups.keys()]
        groups = list(transect_groups.values())
        return transect_groups, group_names, groups
"""
from scipy.stats import normaltest
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import boxcox
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import scipy.stats as stats
from mord import LogisticAT
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CorrSystem:
    @staticmethod
    def do(df):
        # Asegurar que solo tenemos columnas numéricas
        df_num = df.select_dtypes(include=["int64", "float64"])

        if df_num.empty:
            st.error("❌ No hay columnas numéricas disponibles para correlación.")
            return

        # ⭐ AGRUPAR COLUMNAS POR PREGUNTA BASE
        question_groups = CorrSystem._group_columns_by_question(df_num.columns)

        st.markdown("### 🎯 Selección de Variables")

        # ⭐ SELECTOR DE MODO
        st.markdown("#### 🔧 Modo de Análisis")

        analysis_mode = st.radio(
            "Selecciona el modo de correlación",
            options=['individual', 'multiple'],
            format_func=lambda x: {
                'individual': '🎯 Correlación Individual (una pregunta vs todas)',
                'multiple': '📊 Correlación Múltiple (varias preguntas entre sí)'
            }[x],
            horizontal=True,
            help="Múltiple: analiza correlaciones entre varias preguntas. Individual: enfoca en una pregunta vs todas las demás."
        )


        if analysis_mode == 'multiple':
            # ════════════════════════════════════════════
            # MODO MÚLTIPLE (código original)
            # ════════════════════════════════════════════
            CorrSystem._handle_multiple_mode(df_num, question_groups)

        else:
            # ════════════════════════════════════════════
            # MODO INDIVIDUAL (nuevo)
            # ════════════════════════════════════════════
            CorrSystem._handle_individual_mode(df_num, question_groups)

    @staticmethod
    def _handle_multiple_mode(df_num, question_groups):
        """Modo múltiple - código original sin cambios"""

        # Mostrar información de agrupación
        col1, col2 = st.columns([3, 1])

        with col2:
            st.metric("Preguntas disponibles", len(question_groups))
            st.metric("Variables totales", len(df_num.columns))

        with col1:
            # Selector por pregunta
            selected_questions = st.multiselect(
                "Selecciona las preguntas a analizar",
                options=list(question_groups.keys()),
                default=list(question_groups.keys())[:min(5, len(question_groups))],
                help="Cada pregunta incluye automáticamente todas sus categorías de respuesta"
            )

        # Validación
        if len(selected_questions) < 2:
            st.warning("⚠️ Selecciona al menos 2 preguntas para calcular correlaciones.")
            return

        # Expandir preguntas a variables
        selected_columns = []
        for question in selected_questions:
            selected_columns.extend(question_groups[question])

        # Mostrar resumen
        with st.expander("📋 Ver detalle de variables seleccionadas", expanded=False):
            for question in selected_questions:
                st.markdown(f"**{question}**")
                vars_list = question_groups[question]
                st.markdown(f"- {len(vars_list)} categorías: {', '.join([v.split('_')[-1] for v in vars_list])}")

        st.info(f"✅ Total de variables en análisis: **{len(selected_columns)}**")

        if len(selected_columns) > 100:
            st.warning("⚠️ Demasiadas variables. Considera seleccionar menos preguntas para mejor visualización.")
            return

        # Calcular correlación
        corr = df_num[selected_columns].corr()

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Heatmap", "🔝 Top Correlaciones", "📈 Detalles", "🔍 Entre Preguntas"])

        with tab1:
            CorrSystem._plot_heatmap(corr, selected_columns, question_groups)

        with tab2:
            CorrSystem._show_top_correlations(corr)

        with tab3:
            CorrSystem._show_correlation_table(corr)

        with tab4:
            CorrSystem._show_cross_question_correlations(corr, selected_questions, question_groups)

    @staticmethod
    def _handle_individual_mode(df_num, question_groups):
        """Modo individual - una pregunta vs todas"""

        st.markdown("#### 🎯 Correlación Individual")

        # Información general
        col1, col2 = st.columns([3, 1])

        with col2:
            st.metric("Preguntas disponibles", len(question_groups))
            st.metric("Variables totales", len(df_num.columns))

        with col1:
            # Selector de pregunta de referencia
            reference_question = st.selectbox(
                "Selecciona la pregunta de referencia",
                options=list(question_groups.keys()),
                help="Se calcularán las correlaciones de TODAS las categorías de esta pregunta contra todas las demás variables"
            )

        # Mostrar categorías de la pregunta de referencia
        reference_vars = question_groups[reference_question]
        st.info(
            f"📋 Pregunta de referencia: **{reference_question}** ({len(reference_vars)} categorías: {', '.join([v.split('_')[-1] for v in reference_vars])})")

        # Configuración simplificada
        st.markdown("#### ⚙️ Configuración")

        col1, col2 = st.columns(2)

        with col1:
            top_n = st.slider(
                "Número de correlaciones a mostrar",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Muestra las N correlaciones más fuertes (ordenadas por valor absoluto)"
            )

        with col2:
            sort_by = st.radio(
                "Ordenar por",
                options=['abs', 'positive', 'negative'],
                format_func=lambda x: {
                    'abs': 'Valor Absoluto (más fuerte)',
                    'positive': 'Más Positivas',
                    'negative': 'Más Negativas'
                }[x],
                horizontal=False
            )

        # Calcular correlaciones
        with st.spinner("Calculando correlaciones..."):
            all_correlations = CorrSystem._calculate_all_individual_correlations(
                df_num,
                reference_question,
                reference_vars,
                question_groups
            )

        if not all_correlations:
            st.warning("⚠️ No se encontraron correlaciones.")
            return

        # Crear DataFrame
        df_corr = pd.DataFrame(all_correlations)

        # Ordenar según selección
        if sort_by == 'abs':
            df_corr = df_corr.sort_values('Correlacion_Abs', ascending=False)
        elif sort_by == 'positive':
            df_corr = df_corr.sort_values('Correlacion', ascending=False)
        else:  # negative
            df_corr = df_corr.sort_values('Correlacion', ascending=True)

        # Tomar top N
        df_display = df_corr.head(top_n)

        # Mostrar tabla
        st.markdown(f"### 📊 Top {top_n} Correlaciones")
        st.markdown(
            f"Ordenadas por: **{ {'abs': 'Valor Absoluto', 'positive': 'Más Positivas', 'negative': 'Más Negativas'}[sort_by] }**")

        # Formatear nombres de columnas
        df_display_formatted = df_display[[
            'Pregunta_Destino',
            'Categoria_Referencia',
            'Categoria_Destino',
            'Correlacion'
        ]].copy()

        df_display_formatted.columns = ['Pregunta', 'Cat. Referencia', 'Cat. Destino', 'Correlación']

        # Mostrar con estilo
        st.dataframe(
            df_display_formatted.style.format({'Correlación': '{:.4f}'})
            .background_gradient(subset=['Correlación'], cmap='RdBu_r', vmin=-1, vmax=1),
            use_container_width=True,
            height=600
        )

        # Estadísticas rápidas
        col1, col2, col4 = st.columns(3)

        with col1:
            st.metric("Correlación Máxima", f"{df_corr['Correlacion'].max():.3f}")
        with col2:
            st.metric("Correlación Mínima", f"{df_corr['Correlacion'].min():.3f}")
        with col4:
            st.metric("Total Correlaciones", len(df_corr))

        # Exportar (simplificado)
        st.markdown("### 📥 Exportar")

        csv = df_corr.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Todas las Correlaciones (CSV)",
            data=csv,
            file_name=f"correlaciones_{reference_question.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    @staticmethod
    def _calculate_all_individual_correlations(df_num, reference_question, reference_vars, question_groups):
        """
        Calcula TODAS las correlaciones individuales y las retorna en una lista plana
        """
        # Calcular matriz de correlación completa
        corr_matrix = df_num.corr()

        all_correlations = []

        for target_question, target_vars in question_groups.items():
            # Saltar la misma pregunta de referencia
            if target_question == reference_question:
                continue

            # Calcular todas las correlaciones entre ref y target
            for ref_var in reference_vars:
                for target_var in target_vars:
                    if ref_var in corr_matrix.columns and target_var in corr_matrix.columns:
                        corr_value = corr_matrix.loc[ref_var, target_var]

                        all_correlations.append({
                            'Pregunta_Referencia': reference_question,
                            'Variable_Referencia': ref_var,
                            'Categoria_Referencia': ref_var.split('_')[-1],
                            'Pregunta_Destino': target_question,
                            'Variable_Destino': target_var,
                            'Categoria_Destino': target_var.split('_')[-1],
                            'Correlacion': corr_value,
                            'Correlacion_Abs': abs(corr_value)
                        })

        return all_correlations

    @staticmethod
    def _show_individual_ranking(top_results, reference_question, show_details):
        """Muestra el ranking de preguntas con mayor correlación"""

        st.markdown("### 🏆 Ranking de Preguntas")
        st.markdown(f"Correlaciones con respecto a: **{reference_question}**")

        for rank, (question, data) in enumerate(top_results, 1):
            # Card por pregunta
            with st.container():
                st.markdown(f"#### #{rank} - {question}")

                # Métricas
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Promedio", f"{data['avg_corr']:.3f}")
                with col2:
                    st.metric("Promedio (abs)", f"{data['avg_corr_abs']:.3f}")
                with col3:
                    st.metric("Máxima (abs)", f"{data['max_corr_abs']:.3f}")
                with col4:
                    st.metric("N° Correlaciones", data['n_correlations'])

                # Detalles expandibles
                if show_details:
                    with st.expander(f"📋 Ver {len(data['correlations'])} correlaciones individuales"):
                        # Crear DataFrame
                        df_details = pd.DataFrame(data['correlations'])

                        # Ordenar por correlación absoluta
                        df_details['abs_corr'] = df_details['correlation'].abs()
                        df_details = df_details.sort_values('abs_corr', ascending=False)

                        # Formatear para mostrar
                        df_display = df_details[[
                            'var_ref_category',
                            'var_target_category',
                            'correlation'
                        ]].copy()
                        df_display.columns = ['Categoría Referencia', 'Categoría Destino', 'Correlación']

                        # Mostrar con formato
                        st.dataframe(
                            df_display.style.format({'Correlación': '{:.4f}'})
                            .background_gradient(subset=['Correlación'], cmap='RdBu_r', vmin=-1, vmax=1),
                            use_container_width=True,
                            height=min(400, len(df_display) * 35 + 38)
                        )

                        # Destacar máximos
                        max_positive = df_details.loc[df_details['correlation'].idxmax()]
                        max_negative = df_details.loc[df_details['correlation'].idxmin()]

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.success(f"""
                            **Mayor correlación positiva:**  
                            `{max_positive['var_ref_category']}` ↔ `{max_positive['var_target_category']}`  
                            **r = {max_positive['correlation']:.4f}**
                            """)

                        with col_b:
                            if max_negative['correlation'] < 0:
                                st.error(f"""
                                **Mayor correlación negativa:**  
                                `{max_negative['var_ref_category']}` ↔ `{max_negative['var_target_category']}`  
                                **r = {max_negative['correlation']:.4f}**
                                """)

                st.markdown("---")

    @staticmethod
    def _show_individual_export(results, reference_question):
        """Opciones de exportación para modo individual"""

        st.markdown("### 📥 Exportar Resultados")

        # Crear DataFrame completo
        export_data = []

        for question, data in sorted(results.items(), key=lambda x: x[1]['avg_corr_abs'], reverse=True):
            for corr in data['correlations']:
                export_data.append({
                    'Pregunta_Referencia': reference_question,
                    'Variable_Referencia': corr['var_ref'],
                    'Categoria_Referencia': corr['var_ref_category'],
                    'Pregunta_Destino': question,
                    'Variable_Destino': corr['var_target'],
                    'Categoria_Destino': corr['var_target_category'],
                    'Correlacion': corr['correlation'],
                    'Correlacion_Abs': abs(corr['correlation'])
                })

        df_export = pd.DataFrame(export_data)

        # Mostrar preview
        st.markdown("#### Vista Previa")
        st.dataframe(
            df_export.head(20).style.format({'Correlacion': '{:.4f}', 'Correlacion_Abs': '{:.4f}'}),
            use_container_width=True
        )

        st.info(f"📊 Total de correlaciones: **{len(df_export)}**")

        # Botón de descarga
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Todas las Correlaciones (CSV)",
            data=csv,
            file_name=f"correlaciones_individuales_{reference_question.replace(' ', '_')}.csv",
            mime="text/csv"
        )

        # Resumen por pregunta
        st.markdown("---")
        st.markdown("#### Resumen por Pregunta")

        summary_data = []
        for question, data in sorted(results.items(), key=lambda x: x[1]['avg_corr_abs'], reverse=True):
            summary_data.append({
                'Pregunta': question,
                'Correlacion_Promedio': data['avg_corr'],
                'Correlacion_Promedio_Abs': data['avg_corr_abs'],
                'Correlacion_Maxima': data['max_corr'],
                'Correlacion_Maxima_Abs': data['max_corr_abs'],
                'N_Correlaciones': data['n_correlations']
            })

        df_summary = pd.DataFrame(summary_data)

        st.dataframe(
            df_summary.style.format({
                'Correlacion_Promedio': '{:.4f}',
                'Correlacion_Promedio_Abs': '{:.4f}',
                'Correlacion_Maxima': '{:.4f}',
                'Correlacion_Maxima_Abs': '{:.4f}'
            }),
            use_container_width=True
        )

        csv_summary = df_summary.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Resumen por Pregunta (CSV)",
            data=csv_summary,
            file_name=f"resumen_correlaciones_{reference_question.replace(' ', '_')}.csv",
            mime="text/csv"
        )

    # ════════════════════════════════════════════════════════
    # Métodos auxiliares (sin cambios)
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _group_columns_by_question(columns):
        """Agrupa las columnas por su pregunta base"""
        question_groups = {}

        for col in columns:
            if '_' in col:
                parts = col.rsplit('_', 1)
                question_base = parts[0]
                question_base = CorrSystem._clean_question_name(question_base)

                if question_base not in question_groups:
                    question_groups[question_base] = []
                question_groups[question_base].append(col)
            else:
                if "Sin categoría" not in question_groups:
                    question_groups["Sin categoría"] = []
                question_groups["Sin categoría"].append(col)

        return dict(sorted(question_groups.items()))

    @staticmethod
    def _clean_question_name(question):
        """Limpia y acorta el nombre de la pregunta"""
        replacements = {
            "¿Cuánto paga mensualmente de ": "",
            "¿Cuánto paga de ": "",
            "¿Cuántos ": "",
            "¿Tiene ": "",
            "¿": "",
            "?": "",
            "mensualmente": "(mensual)",
        }

        cleaned = question
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        cleaned = cleaned.strip()
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    @staticmethod
    def _plot_heatmap(corr, selected_columns, question_groups):
        """Heatmap para modo múltiple"""
        st.markdown("#### Matriz de Correlación")

        display_names = [col.split('_')[-1] for col in selected_columns]
        n_vars = len(selected_columns)
        height = max(500, min(n_vars * 25, 1200))

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            labels=dict(color="Correlación"),
            x=display_names,
            y=display_names
        )

        fig.update_xaxes(side="bottom", tickangle=45, showticklabels=True)
        fig.update_yaxes(showticklabels=True)
        fig.update_layout(height=height, title="Matriz de Correlaciones (Pearson)", title_x=0.5)

        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Interpretación:**
        - 🔵 **Azul** (≈1): Correlación positiva fuerte
        - ⚪ **Blanco** (≈0): Sin correlación
        - 🔴 **Rojo** (≈-1): Correlación negativa fuerte
        """)

    @staticmethod
    def _show_top_correlations(corr):
        """Top correlaciones para modo múltiple"""
        st.markdown("#### Top Correlaciones más Fuertes")

        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                var1 = corr.columns[i]
                var2 = corr.columns[j]

                question1 = var1.rsplit('_', 1)[0] if '_' in var1 else var1
                question2 = var2.rsplit('_', 1)[0] if '_' in var2 else var2

                if question1 != question2:
                    corr_pairs.append({
                        'Variable 1': var1,
                        'Variable 2': var2,
                        'Correlación': corr.iloc[i, j],
                        'Abs_Corr': abs(corr.iloc[i, j])
                    })

        if not corr_pairs:
            st.warning("⚠️ No hay correlaciones entre diferentes preguntas.")
            return

        df_pairs = pd.DataFrame(corr_pairs)
        df_pairs = df_pairs.sort_values('Abs_Corr', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🔝 Top 15 Correlaciones Positivas")
            top_positive = df_pairs[df_pairs['Correlación'] > 0].head(15)
            if not top_positive.empty:
                st.dataframe(
                    top_positive[['Variable 1', 'Variable 2', 'Correlación']].style.format({'Correlación': '{:.3f}'}),
                    hide_index=True,
                    height=400
                )
            else:
                st.info("No hay correlaciones positivas significativas")

        with col2:
            st.markdown("##### 🔻 Top 15 Correlaciones Negativas")
            top_negative = df_pairs[df_pairs['Correlación'] < 0].head(15)
            if not top_negative.empty:
                st.dataframe(
                    top_negative[['Variable 1', 'Variable 2', 'Correlación']].style.format({'Correlación': '{:.3f}'}),
                    hide_index=True,
                    height=400
                )
            else:
                st.info("No hay correlaciones negativas significativas")

        st.markdown("##### Distribución de Correlaciones")
        fig = px.histogram(
            df_pairs,
            x='Correlación',
            nbins=50,
            title="Distribución de correlaciones entre diferentes preguntas"
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _show_correlation_table(corr):
        """Tabla completa para modo múltiple"""
        st.markdown("#### Matriz de Correlación Completa")

        st.dataframe(
            corr.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format('{:.3f}'),
            use_container_width=True,
            height=600
        )

        csv = corr.to_csv(index=True)
        st.download_button(
            label="📥 Descargar Matriz de Correlación (CSV)",
            data=csv,
            file_name="matriz_correlacion.csv",
            mime="text/csv"
        )

        st.markdown("#### Estadísticas de Correlación")

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_values = corr.where(mask).values.flatten()
        corr_values = corr_values[~np.isnan(corr_values)]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Media", f"{corr_values.mean():.3f}")
        with col2:
            st.metric("Máxima", f"{corr_values.max():.3f}")
        with col3:
            st.metric("Mínima", f"{corr_values.min():.3f}")
        with col4:
            st.metric("Desv. Std", f"{corr_values.std():.3f}")

    @staticmethod
    def _show_cross_question_correlations(corr, selected_questions, question_groups):
        """Correlaciones entre preguntas para modo múltiple"""
        st.markdown("#### Correlaciones Promedio Entre Preguntas")

        question_corr_data = []

        for i, q1 in enumerate(selected_questions):
            for j, q2 in enumerate(selected_questions):
                if i < j:
                    vars_q1 = question_groups[q1]
                    vars_q2 = question_groups[q2]

                    correlations = []
                    for v1 in vars_q1:
                        for v2 in vars_q2:
                            if v1 in corr.columns and v2 in corr.columns:
                                correlations.append(corr.loc[v1, v2])

                    if correlations:
                        avg_corr = np.mean(correlations)
                        max_corr = np.max(np.abs(correlations))

                        question_corr_data.append({
                            'Pregunta 1': q1,
                            'Pregunta 2': q2,
                            'Correlación Promedio': avg_corr,
                            'Correlación Máxima (abs)': max_corr,
                            'N° Comparaciones': len(correlations)
                        })

        if not question_corr_data:
            st.info("No hay suficientes preguntas para comparar.")
            return

        df_question_corr = pd.DataFrame(question_corr_data)
        df_question_corr = df_question_corr.sort_values('Correlación Máxima (abs)', ascending=False)

        st.dataframe(
            df_question_corr.style.format({
                'Correlación Promedio': '{:.3f}',
                'Correlación Máxima (abs)': '{:.3f}'
            }),
            hide_index=True,
            use_container_width=True,
            height=400
        )

        if len(selected_questions) > 2:
            st.markdown("##### Mapa de Correlaciones Entre Preguntas")

            question_corr_matrix = pd.DataFrame(
                np.zeros((len(selected_questions), len(selected_questions))),
                index=selected_questions,
                columns=selected_questions
            )

            for _, row in df_question_corr.iterrows():
                q1 = row['Pregunta 1']
                q2 = row['Pregunta 2']
                corr_val = row['Correlación Promedio']
                question_corr_matrix.loc[q1, q2] = corr_val
                question_corr_matrix.loc[q2, q1] = corr_val

            np.fill_diagonal(question_corr_matrix.values, 1)

            fig = px.imshow(
                question_corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                labels=dict(color="Correlación Promedio")
            )

            fig.update_layout(height=600, title="Correlación Promedio entre Preguntas")

            st.plotly_chart(fig, use_container_width=True)

class RegressionSystem:
    REGRESSION_TYPES = {
        'ols': {
            'name': 'Regresión Lineal Múltiple (OLS)',
            'description': 'Para variables continuas. Predice valores numéricos.',
            'best_for': 'Variable Y continua (ej: gastos, ingresos, edad)',
            'y_type': 'continuous'
        },
        'logistic': {
            'name': 'Regresión Logística',
            'description': 'Para variables binarias. Predice probabilidades.',
            'best_for': 'Variable Y binaria (ej: Sí/No, Tiene/No tiene)',
            'y_type': 'binary'
        },
        'ordinal': {
            'name': 'Regresión Ordinal',
            'description': 'Para variables categóricas ordenadas.',
            'best_for': 'Variable Y ordinal (ej: Bajo < Medio < Alto)',
            'y_type': 'ordinal'
        }
    }

    @staticmethod
    def do(df):
        """Orquestador principal del sistema de regresión"""

        st.markdown("### 📊 Análisis de Regresión")

        # Preparar datos
        with st.spinner("Preparando datos para regresión..."):
            df_processed, category_mappings = DataTreatments.regression_data_handler(df)

        st.success(f"✅ Datos procesados: {df_processed.shape[1]} variables disponibles")

        # Paso 1: Selección de variable dependiente (Y)
        st.markdown("#### 🎯 Paso 1: Variable Dependiente (Y)")

        selected_y = st.selectbox(
            "Selecciona la variable que quieres predecir (Y)",
            options=df_processed.columns.tolist(),
            help="Esta es la variable objetivo que el modelo intentará predecir"
        )

        # Análisis de la variable Y
        y_series = df_processed[selected_y]
        y_info = RegressionSystem._analyze_y_variable(y_series)

        # Mostrar información de Y
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valores únicos", y_info['n_unique'])
        with col2:
            st.metric("Valores faltantes", y_info['n_missing'])
        with col3:
            st.metric("Tipo sugerido", y_info['suggested_type'])

        # Visualización de Y
        with st.expander("📊 Ver distribución de Y", expanded=False):
            RegressionSystem._plot_y_distribution(y_series, y_info)

        # Paso 2: Selección del tipo de regresión
        st.markdown("#### 🔧 Paso 2: Tipo de Regresión")

        regression_type = st.selectbox(
            "Selecciona el tipo de regresión",
            options=['ols', 'logistic', 'ordinal'],
            format_func=lambda x: RegressionSystem.REGRESSION_TYPES[x]['name'],
            index=['ols', 'logistic', 'ordinal'].index(y_info['suggested_model']),
            help="El tipo sugerido se basa en el análisis de tu variable Y"
        )

        # Información del modelo
        model_info = RegressionSystem.REGRESSION_TYPES[regression_type]
        st.info(f"""
        **{model_info['name']}**

        📝 {model_info['description']}

        ✅ **Mejor para:** {model_info['best_for']}
        """)

        # Validación de compatibilidad
        if not RegressionSystem._validate_y_for_model(y_info, regression_type):
            st.error(f"⚠️ La variable Y no es apropiada para {model_info['name']}. "
                     f"Se recomienda usar {y_info['suggested_model'].upper()}.")
            return

        # Paso 3: Selección de variables independientes (X)
        st.markdown("#### 🔬 Paso 3: Variables Independientes (X)")

        available_x = [col for col in df_processed.columns if col != selected_y]

        # Agrupar variables por pregunta
        question_groups = RegressionSystem._group_columns_by_question(available_x)

        col1, col2 = st.columns([3, 1])

        with col1:
            selection_method = st.radio(
                "Método de selección",
                options=['manual', 'by_correlation'],
                format_func=lambda x: {
                    'manual': '📝 Selección manual (variables individuales)',
                    'by_correlation': '🔗 Por correlación con Y (top N)'
                }[x],
                horizontal=True
            )

        with col2:
            st.metric("Variables disponibles", len(available_x))

        # Selección según método
        if selection_method == 'manual':
            selected_x = st.multiselect(
                "Selecciona variables predictoras",
                options=available_x,
                default=available_x[:min(5, len(available_x))],
                help="Puedes seleccionar múltiples variables"
            )

        elif selection_method == 'by_question':
            selected_questions = st.multiselect(
                "Selecciona preguntas (incluye todas sus categorías)",
                options=list(question_groups.keys()),
                default=list(question_groups.keys())[:min(3, len(question_groups))],
                help="Cada pregunta incluye automáticamente todas sus categorías"
            )

            # Expandir preguntas a variables
            selected_x = []
            for question in selected_questions:
                selected_x.extend(question_groups[question])

            with st.expander("📋 Variables incluidas", expanded=False):
                for question in selected_questions:
                    st.markdown(f"**{question}**")
                    vars_list = [v.split('_')[-1] for v in question_groups[question]]
                    st.markdown(f"- {len(vars_list)} categorías: {', '.join(vars_list)}")

        else:  # by_correlation
            top_n = st.slider("Número de variables más correlacionadas", 5, 50, 15)

            # Calcular correlaciones
            correlations = df_processed[available_x].corrwith(y_series).abs().sort_values(ascending=False)
            selected_x = correlations.head(top_n).index.tolist()

            st.dataframe(
                pd.DataFrame({
                    'Variable': selected_x,
                    'Correlación (abs)': [correlations[v] for v in selected_x]
                }).style.format({'Correlación (abs)': '{:.3f}'}),
                hide_index=True,
                height=300
            )

        if len(selected_x) < 1:
            st.warning("⚠️ Selecciona al menos 1 variable independiente.")
            return

        st.info(f"✅ Total de variables predictoras: **{len(selected_x)}**")

        # Paso 4: Configuración del modelo
        st.markdown("#### ⚙️ Paso 4: Configuración")

        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider(
                "% de datos para prueba",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Porcentaje de datos reservados para validar el modelo"
            )

        with col2:
            random_state = st.number_input(
                "Semilla aleatoria",
                min_value=0,
                max_value=9999,
                value=42,
                help="Para reproducibilidad de resultados"
            )

        # Botón de entrenamiento
        if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):

            with st.spinner("Entrenando modelo..."):
                # Preparar datos
                X = df_processed[selected_x].copy()
                y = y_series.copy()

                # Eliminar filas con NaN
                valid_idx = ~(X.isna().any(axis=1) | y.isna())
                X = X[valid_idx]
                y = y[valid_idx]

                if len(X) < 10:
                    st.error("❌ No hay suficientes datos válidos para entrenar (mínimo 10 observaciones)")
                    return

                # Entrenar modelo
                results = RegressionSystem._train_model(
                    X, y,
                    regression_type=regression_type,
                    test_size=test_size / 100,
                    random_state=random_state
                )

                if results is None:
                    st.error("❌ Error al entrenar el modelo")
                    return

                # Guardar en session_state
                st.session_state['regression_results'] = results
                st.session_state['regression_config'] = {
                    'y_name': selected_y,
                    'x_names': selected_x,
                    'regression_type': regression_type
                }

            st.success("✅ Modelo entrenado exitosamente!")
            st.rerun()

        # Mostrar resultados si existen
        if 'regression_results' in st.session_state:
            st.markdown("## 📈 Resultados del Modelo")

            results = st.session_state['regression_results']
            config = st.session_state['regression_config']

            # Tabs de resultados
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Métricas",
                "📉 Visualizaciones",
                "🎯 Coeficientes",
                "📥 Exportar"
            ])

            with tab1:
                RegressionSystem._show_metrics(results, config['regression_type'])

            with tab2:
                RegressionSystem._show_visualizations(results, config)

            with tab3:
                RegressionSystem._show_coefficients(results, config)

            with tab4:
                RegressionSystem._show_export_options(results, config)

    @staticmethod
    def _analyze_y_variable(y_series):
        """Analiza la variable Y y sugiere el tipo de regresión apropiado"""

        n_unique = y_series.nunique()
        n_missing = y_series.isna().sum()

        # Determinar tipo sugerido
        if n_unique == 2:
            suggested_type = 'Binaria'
            suggested_model = 'logistic'
        elif n_unique <= 10:
            # Verificar si parece ordinal
            unique_vals = sorted(y_series.dropna().unique())
            if all(isinstance(v, (int, float)) for v in unique_vals):
                suggested_type = 'Ordinal/Discreta'
                suggested_model = 'ordinal'
            else:
                suggested_type = 'Categórica'
                suggested_model = 'ordinal'
        else:
            suggested_type = 'Continua'
            suggested_model = 'ols'

        return {
            'n_unique': n_unique,
            'n_missing': n_missing,
            'suggested_type': suggested_type,
            'suggested_model': suggested_model,
            'min': y_series.min(),
            'max': y_series.max(),
            'mean': y_series.mean() if n_unique > 2 else None,
            'std': y_series.std() if n_unique > 2 else None
        }

    @staticmethod
    def _plot_y_distribution(y_series, y_info):
        """Visualiza la distribución de Y"""

        if y_info['n_unique'] <= 20:
            # Gráfico de barras para variables discretas
            value_counts = y_series.value_counts().sort_index()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={'x': 'Valor', 'y': 'Frecuencia'},
                title=f"Distribución de la Variable Y (n={len(y_series)})"
            )
        else:
            # Histograma para variables continuas
            fig = px.histogram(
                y_series,
                nbins=50,
                labels={'value': 'Valor', 'count': 'Frecuencia'},
                title=f"Distribución de la Variable Y (n={len(y_series)})"
            )

        st.plotly_chart(fig, use_container_width=True)

        # Estadísticas descriptivas
        if y_info['mean'] is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mínimo", f"{y_info['min']:.2f}")
            with col2:
                st.metric("Media", f"{y_info['mean']:.2f}")
            with col3:
                st.metric("Máximo", f"{y_info['max']:.2f}")
            with col4:
                st.metric("Desv. Std", f"{y_info['std']:.2f}")

    @staticmethod
    def _validate_y_for_model(y_info, regression_type):
        """Valida que Y sea apropiada para el modelo seleccionado"""

        if regression_type == 'logistic' and y_info['n_unique'] != 2:
            return False

        if regression_type == 'ordinal' and y_info['n_unique'] < 3:
            return False

        if regression_type == 'ols' and y_info['n_unique'] < 10:
            # Advertencia pero no bloquear
            st.warning("⚠️ Y tiene pocos valores únicos. Considera usar regresión ordinal.")

        return True

    @staticmethod
    def _analyze_y_variable(y_series):
        """Analiza la variable Y y sugiere el tipo de regresión apropiado"""

        n_unique = y_series.nunique()
        n_missing = y_series.isna().sum()

        # Determinar tipo sugerido
        if n_unique == 2:
            suggested_type = 'Binaria'
            suggested_model = 'logistic'
        elif n_unique <= 10:
            # Verificar si parece ordinal
            unique_vals = sorted(y_series.dropna().unique())
            if all(isinstance(v, (int, float)) for v in unique_vals):
                suggested_type = 'Ordinal/Discreta'
                suggested_model = 'ordinal'
            else:
                suggested_type = 'Categórica'
                suggested_model = 'ordinal'
        else:
            suggested_type = 'Continua'
            suggested_model = 'ols'

        return {
            'n_unique': n_unique,
            'n_missing': n_missing,
            'suggested_type': suggested_type,
            'suggested_model': suggested_model,
            'min': y_series.min(),
            'max': y_series.max(),
            'mean': y_series.mean() if n_unique > 2 else None,
            'std': y_series.std() if n_unique > 2 else None
        }

    @staticmethod
    def _plot_y_distribution(y_series, y_info):
        """Visualiza la distribución de Y"""

        if y_info['n_unique'] <= 20:
            # Gráfico de barras para variables discretas
            value_counts = y_series.value_counts().sort_index()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={'x': 'Valor', 'y': 'Frecuencia'},
                title=f"Distribución de la Variable Y (n={len(y_series)})"
            )
        else:
            # Histograma para variables continuas
            fig = px.histogram(
                y_series,
                nbins=50,
                labels={'value': 'Valor', 'count': 'Frecuencia'},
                title=f"Distribución de la Variable Y (n={len(y_series)})"
            )

        st.plotly_chart(fig, use_container_width=True)

        # Estadísticas descriptivas
        if y_info['mean'] is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mínimo", f"{y_info['min']:.2f}")
            with col2:
                st.metric("Media", f"{y_info['mean']:.2f}")
            with col3:
                st.metric("Máximo", f"{y_info['max']:.2f}")
            with col4:
                st.metric("Desv. Std", f"{y_info['std']:.2f}")

    @staticmethod
    def _validate_y_for_model(y_info, regression_type):
        """Valida que Y sea apropiada para el modelo seleccionado"""

        if regression_type == 'logistic' and y_info['n_unique'] != 2:
            return False

        if regression_type == 'ordinal' and y_info['n_unique'] < 3:
            return False

        if regression_type == 'ols' and y_info['n_unique'] < 10:
            # Advertencia pero no bloquear
            st.warning("⚠️ Y tiene pocos valores únicos. Considera usar regresión ordinal.")

        return True

    @staticmethod
    def _group_columns_by_question(columns):
        """Agrupa columnas por pregunta base"""
        question_groups = {}

        for col in columns:
            if '_' in col:
                parts = col.rsplit('_', 1)
                question_base = parts[0]

                # Limpiar nombre
                question_base = question_base.replace('¿', '').replace('?', '').strip()

                if question_base not in question_groups:
                    question_groups[question_base] = []
                question_groups[question_base].append(col)
            else:
                if "Sin categoría" not in question_groups:
                    question_groups["Sin categoría"] = []
                question_groups["Sin categoría"].append(col)

        return dict(sorted(question_groups.items()))

    @staticmethod
    def _train_model(X, y, regression_type, test_size=0.2, random_state=42):
        """
        Entrena el modelo seleccionado y retorna los resultados
        """
        try:
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Estandarizar X (importante para regresión)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Convertir de vuelta a DataFrame para mantener nombres
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

            # Entrenar según tipo
            if regression_type == 'ols':
                model, metrics = RegressionSystem._train_ols(
                    X_train_scaled, X_test_scaled, y_train, y_test
                )

            elif regression_type == 'logistic':
                model, metrics = RegressionSystem._train_logistic(
                    X_train_scaled, X_test_scaled, y_train, y_test
                )

            elif regression_type == 'ordinal':
                model, metrics = RegressionSystem._train_ordinal(
                    X_train_scaled, X_test_scaled, y_train, y_test
                )

            else:
                return None

            # Predicciones
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Empaquetar resultados
            results = {
                'model': model,
                'scaler': scaler,
                'X_train': X_train,
                'X_test': X_test,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'metrics': metrics,
                'feature_names': X.columns.tolist()
            }

            return results

        except Exception as e:
            st.error(f"Error al entrenar el modelo: {str(e)}")
            return None

    @staticmethod
    def _train_ols(X_train, X_test, y_train, y_test):
        """Entrena regresión lineal OLS"""
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predicciones
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Métricas
        metrics = {
            'train': {
                'r2': r2_score(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mae': mean_absolute_error(y_train, y_train_pred)
            },
            'test': {
                'r2': r2_score(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mae': mean_absolute_error(y_test, y_test_pred)
            }
        }

        # R² ajustado
        n = len(y_train)
        p = X_train.shape[1]
        r2_train = metrics['train']['r2']
        metrics['train']['r2_adjusted'] = 1 - (1 - r2_train) * (n - 1) / (n - p - 1)

        n_test = len(y_test)
        r2_test = metrics['test']['r2']
        metrics['test']['r2_adjusted'] = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)

        return model, metrics

    @staticmethod
    def _train_logistic(X_train, X_test, y_train, y_test):
        """Entrena regresión logística"""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Predicciones
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilidades
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            'train': {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred, zero_division=0),
                'recall': recall_score(y_train, y_train_pred, zero_division=0),
                'f1': f1_score(y_train, y_train_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_train, y_train_proba)
            },
            'test': {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred, zero_division=0),
                'recall': recall_score(y_test, y_test_pred, zero_division=0),
                'f1': f1_score(y_test, y_test_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_test_proba)
            },
            'probabilities': {
                'train': y_train_proba,
                'test': y_test_proba
            }
        }

        return model, metrics

    @staticmethod
    def _train_ordinal(X_train, X_test, y_train, y_test):
        """Entrena regresión ordinal"""
        try:
            from mord import LogisticAT
            model = LogisticAT()
        except ImportError:
            st.warning("⚠️ Librería 'mord' no disponible. Usando regresión lineal como alternativa.")
            # Fallback a OLS si mord no está disponible
            return RegressionSystem._train_ols(X_train, X_test, y_train, y_test)

        model.fit(X_train, y_train)

        # Predicciones
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Métricas
        from sklearn.metrics import accuracy_score, mean_absolute_error

        metrics = {
            'train': {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'mae': mean_absolute_error(y_train, y_train_pred),
                'mae_baseline': mean_absolute_error(y_train, [y_train.mode()[0]] * len(y_train))
            },
            'test': {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'mae': mean_absolute_error(y_test, y_test_pred),
                'mae_baseline': mean_absolute_error(y_test, [y_train.mode()[0]] * len(y_test))
            }
        }

        return model, metrics

    @staticmethod
    def _show_metrics(results, regression_type):
        """Muestra las métricas del modelo"""

        st.markdown("### 📊 Métricas de Desempeño")

        metrics = results['metrics']

        if regression_type == 'ols':
            # Métricas para regresión lineal
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🎓 Conjunto de Entrenamiento")
                st.metric("R² Score", f"{metrics['train']['r2']:.4f}")
                st.metric("R² Ajustado", f"{metrics['train']['r2_adjusted']:.4f}")
                st.metric("RMSE", f"{metrics['train']['rmse']:.4f}")
                st.metric("MAE", f"{metrics['train']['mae']:.4f}")

            with col2:
                st.markdown("#### 🧪 Conjunto de Prueba")
                st.metric("R² Score", f"{metrics['test']['r2']:.4f}")
                st.metric("R² Ajustado", f"{metrics['test']['r2_adjusted']:.4f}")
                st.metric("RMSE", f"{metrics['test']['rmse']:.4f}")
                st.metric("MAE", f"{metrics['test']['mae']:.4f}")

            # Interpretación
            st.markdown("#### 📖 Interpretación")

            r2_test = metrics['test']['r2']

            if r2_test >= 0.8:
                interpretation = "🟢 **Excelente ajuste:** El modelo explica más del 80% de la variabilidad en Y."
            elif r2_test >= 0.6:
                interpretation = "🟡 **Buen ajuste:** El modelo explica entre 60-80% de la variabilidad en Y."
            elif r2_test >= 0.4:
                interpretation = "🟠 **Ajuste moderado:** El modelo explica entre 40-60% de la variabilidad en Y."
            else:
                interpretation = "🔴 **Ajuste pobre:** El modelo explica menos del 40% de la variabilidad en Y."

            st.info(f"""
                {interpretation}

                **R² = {r2_test:.3f}** significa que el modelo explica el {r2_test * 100:.1f}% de la varianza en la variable dependiente.

                **RMSE = {metrics['test']['rmse']:.3f}** indica que, en promedio, las predicciones difieren de los valores reales en ±{metrics['test']['rmse']:.3f} unidades.
                """)

            # Comparar train vs test
            diff_r2 = abs(metrics['train']['r2'] - metrics['test']['r2'])
            if diff_r2 > 0.1:
                st.warning(
                    f"⚠️ **Posible overfitting:** La diferencia entre R² de entrenamiento ({metrics['train']['r2']:.3f}) "
                    f"y prueba ({metrics['test']['r2']:.3f}) es significativa ({diff_r2:.3f}).")

        elif regression_type == 'logistic':
            # Métricas para regresión logística
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🎓 Conjunto de Entrenamiento")
                st.metric("Accuracy", f"{metrics['train']['accuracy']:.4f}")
                st.metric("Precision", f"{metrics['train']['precision']:.4f}")
                st.metric("Recall", f"{metrics['train']['recall']:.4f}")
                st.metric("F1-Score", f"{metrics['train']['f1']:.4f}")
                st.metric("ROC-AUC", f"{metrics['train']['roc_auc']:.4f}")

            with col2:
                st.markdown("#### 🧪 Conjunto de Prueba")
                st.metric("Accuracy", f"{metrics['test']['accuracy']:.4f}")
                st.metric("Precision", f"{metrics['test']['precision']:.4f}")
                st.metric("Recall", f"{metrics['test']['recall']:.4f}")
                st.metric("F1-Score", f"{metrics['test']['f1']:.4f}")
                st.metric("ROC-AUC", f"{metrics['test']['roc_auc']:.4f}")

            # Matriz de confusión
            st.markdown("#### 🎯 Matriz de Confusión (Prueba)")

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(results['y_test'], results['y_test_pred'])

            fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicción", y="Real", color="Cantidad"),
                x=['Clase 0', 'Clase 1'],
                y=['Clase 0', 'Clase 1'],
                color_continuous_scale='Blues'
            )
            fig.update_layout(title="Matriz de Confusión")
            st.plotly_chart(fig, use_container_width=True)

            # Interpretación
            st.markdown("#### 📖 Interpretación")
            acc = metrics['test']['accuracy']

            st.info(f"""
                **Accuracy = {acc:.3f}**: El modelo clasifica correctamente el {acc * 100:.1f}% de las observaciones.

                **Precision = {metrics['test']['precision']:.3f}**: De las predicciones positivas, el {metrics['test']['precision'] * 100:.1f}% son correctas.

                **Recall = {metrics['test']['recall']:.3f}**: El modelo detecta el {metrics['test']['recall'] * 100:.1f}% de los casos positivos reales.

                **ROC-AUC = {metrics['test']['roc_auc']:.3f}**: Capacidad del modelo para distinguir entre clases (1.0 = perfecto).
                """)

        elif regression_type == 'ordinal':
            # Métricas para regresión ordinal
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🎓 Conjunto de Entrenamiento")
                st.metric("Accuracy", f"{metrics['train']['accuracy']:.4f}")
                st.metric("MAE", f"{metrics['train']['mae']:.4f}")
                st.metric("MAE Baseline", f"{metrics['train']['mae_baseline']:.4f}")

            with col2:
                st.markdown("#### 🧪 Conjunto de Prueba")
                st.metric("Accuracy", f"{metrics['test']['accuracy']:.4f}")
                st.metric("MAE", f"{metrics['test']['mae']:.4f}")
                st.metric("MAE Baseline", f"{metrics['test']['mae_baseline']:.4f}")

            # Interpretación
            st.markdown("#### 📖 Interpretación")

            improvement = (metrics['test']['mae_baseline'] - metrics['test']['mae']) / metrics['test']['mae_baseline']

            st.info(f"""
                **Accuracy = {metrics['test']['accuracy']:.3f}**: El modelo predice la categoría exacta correctamente el {metrics['test']['accuracy'] * 100:.1f}% del tiempo.

                **MAE = {metrics['test']['mae']:.3f}**: En promedio, las predicciones difieren en {metrics['test']['mae']:.2f} categorías de los valores reales.

                **Mejora sobre baseline:** {improvement * 100:.1f}% mejor que predecir siempre la categoría más frecuente.
                """)

    @staticmethod
    def _show_visualizations(results, config):
        """Muestra las visualizaciones del modelo"""

        regression_type = config['regression_type']

        if regression_type == 'ols':
            RegressionSystem._plot_ols_visualizations(results, config)
        elif regression_type == 'logistic':
            RegressionSystem._plot_logistic_visualizations(results, config)
        elif regression_type == 'ordinal':
            RegressionSystem._plot_ordinal_visualizations(results, config)

    @staticmethod
    def _plot_ols_visualizations(results, config):
        """Visualizaciones para regresión lineal OLS"""

        # 1. Predicciones vs Reales
        st.markdown("### 📊 Predicciones vs Valores Reales")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Entrenamiento")
            fig = go.Figure()

            # Scatter plot
            fig.add_trace(go.Scatter(
                x=results['y_train'],
                y=results['y_train_pred'],
                mode='markers',
                name='Predicciones',
                marker=dict(color='blue', size=6, opacity=0.6)
            ))

            # Línea perfecta
            min_val = min(results['y_train'].min(), results['y_train_pred'].min())
            max_val = max(results['y_train'].max(), results['y_train_pred'].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predicción perfecta',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                xaxis_title="Valores Reales",
                yaxis_title="Valores Predichos",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Prueba")
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=results['y_test'],
                y=results['y_test_pred'],
                mode='markers',
                name='Predicciones',
                marker=dict(color='green', size=6, opacity=0.6)
            ))

            min_val = min(results['y_test'].min(), results['y_test_pred'].min())
            max_val = max(results['y_test'].max(), results['y_test_pred'].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predicción perfecta',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                xaxis_title="Valores Reales",
                yaxis_title="Valores Predichos",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # 2. Residuos vs Predicciones
        st.markdown("### 📉 Análisis de Residuos")

        residuals_test = results['y_test'] - results['y_test_pred']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Residuos vs Predicciones")
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=results['y_test_pred'],
                y=residuals_test,
                mode='markers',
                marker=dict(color='purple', size=6, opacity=0.6)
            ))

            # Línea en cero
            fig.add_hline(y=0, line_dash="dash", line_color="red")

            fig.update_layout(
                xaxis_title="Valores Predichos",
                yaxis_title="Residuos",
                height=400,
                title="Los residuos deben distribuirse aleatoriamente alrededor de 0"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("""
                **¿Qué buscar?**
                - ✅ Residuos distribuidos aleatoriamente alrededor de 0
                - ❌ Patrones sistemáticos (curvas, abanico) indican problemas
                """)

        with col2:
            st.markdown("#### Distribución de Residuos")
            fig = px.histogram(
                residuals_test,
                nbins=30,
                labels={'value': 'Residuos', 'count': 'Frecuencia'},
                title="Los residuos deben seguir una distribución normal"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("""
                **¿Qué buscar?**
                - ✅ Forma de campana (distribución normal)
                - ❌ Asimetría fuerte indica problemas
                """)

        # 3. Q-Q Plot
        st.markdown("### 📐 Q-Q Plot (Normalidad de Residuos)")

        fig = go.Figure()

        # Calcular quantiles teóricos y empíricos
        residuals_sorted = np.sort(residuals_test)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_sorted)))

        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=residuals_sorted,
            mode='markers',
            name='Quantiles observados',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))

        # Línea de referencia
        fig.add_trace(go.Scatter(
            x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
            y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
            mode='lines',
            name='Distribución normal perfecta',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            xaxis_title="Quantiles Teóricos",
            yaxis_title="Quantiles Observados",
            height=500,
            title="Q-Q Plot: Normalidad de Residuos"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
            **¿Qué buscar?**
            - ✅ Puntos alineados con la línea roja = residuos normales
            - ❌ Desviaciones sistemáticas = residuos no normales
            - Desviaciones en los extremos son comunes y aceptables
            """)

    @staticmethod
    def _plot_logistic_visualizations(results, config):
        """Visualizaciones para regresión logística"""

        # 1. Curva ROC
        st.markdown("### 📈 Curva ROC")

        from sklearn.metrics import roc_curve, auc

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Entrenamiento")
            fpr, tpr, _ = roc_curve(results['y_train'], results['metrics']['probabilities']['train'])
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {roc_auc:.3f})',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Aleatorio',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                xaxis_title="Tasa de Falsos Positivos",
                yaxis_title="Tasa de Verdaderos Positivos",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Prueba")
            fpr, tpr, _ = roc_curve(results['y_test'], results['metrics']['probabilities']['test'])
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {roc_auc:.3f})',
                line=dict(color='green', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Aleatorio',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                xaxis_title="Tasa de Falsos Positivos",
                yaxis_title="Tasa de Verdaderos Positivos",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # 2. Distribución de probabilidades predichas
        st.markdown("### 📊 Distribución de Probabilidades Predichas")

        fig = go.Figure()

        # Clase 0
        probs_class0 = results['metrics']['probabilities']['test'][results['y_test'] == 0]
        fig.add_trace(go.Histogram(
            x=probs_class0,
            name='Clase 0 (real)',
            opacity=0.7,
            nbinsx=20
        ))

        # Clase 1
        probs_class1 = results['metrics']['probabilities']['test'][results['y_test'] == 1]
        fig.add_trace(go.Histogram(
            x=probs_class1,
            name='Clase 1 (real)',
            opacity=0.7,
            nbinsx=20
        ))

        fig.update_layout(
            xaxis_title="Probabilidad Predicha",
            yaxis_title="Frecuencia",
            barmode='overlay',
            height=400,
            title="Separación de clases por probabilidad"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
            **¿Qué buscar?**
            - ✅ Buena separación entre histogramas = modelo discrimina bien
            - ❌ Mucha superposición = modelo tiene dificultad para separar clases
            """)

    @staticmethod
    def _plot_ordinal_visualizations(results, config):
        """Visualizaciones para regresión ordinal"""

        # 1. Matriz de confusión detallada
        st.markdown("### 🎯 Matriz de Confusión")

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(results['y_test'], results['y_test_pred'])

        # Obtener labels únicos
        unique_labels = sorted(results['y_test'].unique())
        label_names = [f"Categoría {int(l)}" for l in unique_labels]

        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicción", y="Real", color="Cantidad"),
            x=label_names,
            y=label_names,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            title="Matriz de Confusión - Regresión Ordinal",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # 2. Distribución de predicciones vs reales
        st.markdown("### 📊 Distribución de Predicciones")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Valores Reales")
            real_counts = results['y_test'].value_counts().sort_index()
            fig = px.bar(
                x=real_counts.index,
                y=real_counts.values,
                labels={'x': 'Categoría', 'y': 'Frecuencia'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Valores Predichos")
            pred_counts = pd.Series(results['y_test_pred']).value_counts().sort_index()
            fig = px.bar(
                x=pred_counts.index,
                y=pred_counts.values,
                labels={'x': 'Categoría', 'y': 'Frecuencia'},
                color_discrete_sequence=['green']
            )
            st.plotly_chart(fig, use_container_width=True)

        # 3. Errores de predicción
        st.markdown("### 📉 Análisis de Errores")

        errors = results['y_test'].values - results['y_test_pred']

        fig = px.histogram(
            errors,
            nbins=len(unique_labels) * 2,
            labels={'value': 'Error (Real - Predicho)', 'count': 'Frecuencia'},
            title="Distribución de Errores"
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
            **¿Qué buscar?**
            - ✅ Errores centrados en 0 = buenas predicciones
            - ❌ Sesgo sistemático hacia un lado = modelo subestima o sobreestima
            """)

    @staticmethod
    def _show_coefficients(results, config):
        """Muestra los coeficientes del modelo y su importancia"""

        st.markdown("### 🎯 Coeficientes del Modelo")

        model = results['model']
        feature_names = results['feature_names']
        regression_type = config['regression_type']

        if regression_type == 'ols':
            # Coeficientes de regresión lineal
            coefficients = model.coef_
            intercept = model.intercept_

            # Crear DataFrame de coeficientes
            coef_df = pd.DataFrame({
                'Variable': feature_names,
                'Coeficiente': coefficients,
                'Abs_Coeficiente': np.abs(coefficients)
            }).sort_values('Abs_Coeficiente', ascending=False)

            # Métricas generales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Intercepto (β₀)", f"{intercept:.4f}")
            with col2:
                st.metric("Variables", len(feature_names))
            with col3:
                st.metric("Coef. más alto", f"{coef_df.iloc[0]['Coeficiente']:.4f}")

            # Visualización de coeficientes
            st.markdown("#### 📊 Importancia de Variables")

            # Top N variables
            top_n = st.slider("Mostrar top N variables", 5, min(50, len(feature_names)), min(15, len(feature_names)))

            top_coef = coef_df.head(top_n)

            fig = px.bar(
                top_coef,
                x='Coeficiente',
                y='Variable',
                orientation='h',
                title=f"Top {top_n} Variables por Importancia (Valor Absoluto)",
                labels={'Coeficiente': 'Coeficiente', 'Variable': 'Variable'},
                color='Coeficiente',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0
            )
            fig.update_layout(height=max(400, top_n * 25))
            st.plotly_chart(fig, use_container_width=True)

            # Interpretación
            st.markdown("#### 📖 Interpretación de Coeficientes")

            st.info(f"""
                **Intercepto (β₀) = {intercept:.4f}**: Valor predicho de Y cuando todas las variables X son 0.

                **Coeficientes positivos**: Cuando la variable aumenta, Y tiende a aumentar.

                **Coeficientes negativos**: Cuando la variable aumenta, Y tiende a disminuir.

                **Magnitud**: El valor absoluto indica la fuerza del efecto.

                ⚠️ **Nota**: Los coeficientes están en escala estandarizada, por lo que son comparables entre sí.
                """)

            # Tabla completa
            with st.expander("📋 Ver tabla completa de coeficientes", expanded=False):
                st.dataframe(
                    coef_df.style.format({
                        'Coeficiente': '{:.6f}',
                        'Abs_Coeficiente': '{:.6f}'
                    }).background_gradient(subset=['Coeficiente'], cmap='RdBu_r',
                                           vmin=-coef_df['Abs_Coeficiente'].max(),
                                           vmax=coef_df['Abs_Coeficiente'].max()),
                    use_container_width=True,
                    height=400
                )

            # Análisis adicional
            st.markdown("#### 🔍 Análisis Adicional")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Variables con Mayor Impacto Positivo")
                top_positive = coef_df[coef_df['Coeficiente'] > 0].head(5)
                if not top_positive.empty:
                    for idx, row in top_positive.iterrows():
                        st.markdown(f"- **{row['Variable']}**: +{row['Coeficiente']:.4f}")
                else:
                    st.info("No hay variables con impacto positivo")

            with col2:
                st.markdown("##### Variables con Mayor Impacto Negativo")
                top_negative = coef_df[coef_df['Coeficiente'] < 0].head(5)
                if not top_negative.empty:
                    for idx, row in top_negative.iterrows():
                        st.markdown(f"- **{row['Variable']}**: {row['Coeficiente']:.4f}")
                else:
                    st.info("No hay variables con impacto negativo")

        elif regression_type == 'logistic':
            # Coeficientes de regresión logística
            coefficients = model.coef_[0]  # Para clasificación binaria
            intercept = model.intercept_[0]

            # Calcular odds ratios
            odds_ratios = np.exp(coefficients)

            # Crear DataFrame
            coef_df = pd.DataFrame({
                'Variable': feature_names,
                'Coeficiente': coefficients,
                'Odds Ratio': odds_ratios,
                'Abs_Coeficiente': np.abs(coefficients)
            }).sort_values('Abs_Coeficiente', ascending=False)

            # Métricas generales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Intercepto (β₀)", f"{intercept:.4f}")
            with col2:
                st.metric("Variables", len(feature_names))
            with col3:
                st.metric("Mayor Odds Ratio", f"{coef_df.iloc[0]['Odds Ratio']:.4f}")

            # Visualización
            st.markdown("#### 📊 Importancia de Variables")

            top_n = st.slider("Mostrar top N variables", 5, min(50, len(feature_names)), min(15, len(feature_names)))
            top_coef = coef_df.head(top_n)

            # Gráfico de coeficientes
            fig = px.bar(
                top_coef,
                x='Coeficiente',
                y='Variable',
                orientation='h',
                title=f"Top {top_n} Variables por Importancia",
                color='Coeficiente',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0
            )
            fig.update_layout(height=max(400, top_n * 25))
            st.plotly_chart(fig, use_container_width=True)

            # Gráfico de Odds Ratios
            st.markdown("#### 📈 Odds Ratios")

            fig = px.bar(
                top_coef,
                x='Odds Ratio',
                y='Variable',
                orientation='h',
                title=f"Odds Ratios - Top {top_n} Variables",
                color='Odds Ratio',
                color_continuous_scale='Viridis'
            )
            fig.add_vline(x=1, line_dash="dash", line_color="red",
                          annotation_text="OR = 1 (sin efecto)")
            fig.update_layout(height=max(400, top_n * 25))
            st.plotly_chart(fig, use_container_width=True)

            # Interpretación
            st.markdown("#### 📖 Interpretación")

            st.info("""
                **Coeficientes (β)**:
                - Coeficiente positivo: La variable aumenta la probabilidad de Y=1
                - Coeficiente negativo: La variable disminuye la probabilidad de Y=1

                **Odds Ratios (OR)**:
                - OR > 1: La variable aumenta las probabilidades de Y=1
                - OR < 1: La variable disminuye las probabilidades de Y=1
                - OR = 1: La variable no tiene efecto

                **Ejemplo**: Si OR = 2.5, significa que por cada unidad de aumento en X, 
                las probabilidades de Y=1 se multiplican por 2.5 (aumentan 150%).
                """)

            # Tabla completa
            with st.expander("📋 Ver tabla completa de coeficientes", expanded=False):
                st.dataframe(
                    coef_df.style.format({
                        'Coeficiente': '{:.6f}',
                        'Odds Ratio': '{:.4f}',
                        'Abs_Coeficiente': '{:.6f}'
                    }),
                    use_container_width=True,
                    height=400
                )

        elif regression_type == 'ordinal':
            # Coeficientes de regresión ordinal
            try:
                coefficients = model.coef_

                # Crear DataFrame
                coef_df = pd.DataFrame({
                    'Variable': feature_names,
                    'Coeficiente': coefficients,
                    'Abs_Coeficiente': np.abs(coefficients)
                }).sort_values('Abs_Coeficiente', ascending=False)

                # Métricas generales
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Variables", len(feature_names))
                with col2:
                    st.metric("Coef. más alto", f"{coef_df.iloc[0]['Coeficiente']:.4f}")

                # Visualización
                st.markdown("#### 📊 Importancia de Variables")

                top_n = st.slider("Mostrar top N variables", 5, min(50, len(feature_names)),
                                  min(15, len(feature_names)))
                top_coef = coef_df.head(top_n)

                fig = px.bar(
                    top_coef,
                    x='Coeficiente',
                    y='Variable',
                    orientation='h',
                    title=f"Top {top_n} Variables por Importancia",
                    color='Coeficiente',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0
                )
                fig.update_layout(height=max(400, top_n * 25))
                st.plotly_chart(fig, use_container_width=True)

                # Interpretación
                st.markdown("#### 📖 Interpretación")

                st.info("""
                    **Coeficientes positivos**: La variable aumenta la probabilidad de categorías más altas.

                    **Coeficientes negativos**: La variable aumenta la probabilidad de categorías más bajas.

                    **Magnitud**: El valor absoluto indica la fuerza del efecto.
                    """)

                # Tabla completa
                with st.expander("📋 Ver tabla completa de coeficientes", expanded=False):
                    st.dataframe(
                        coef_df.style.format({
                            'Coeficiente': '{:.6f}',
                            'Abs_Coeficiente': '{:.6f}'
                        }),
                        use_container_width=True,
                        height=400
                    )

            except Exception as e:
                st.warning(f"No se pueden mostrar coeficientes para este modelo ordinal: {str(e)}")

    @staticmethod
    def _show_export_options(results, config):
        """Opciones para exportar resultados"""

        st.markdown("### 📥 Exportar Resultados")

        regression_type = config['regression_type']

        # 1. Exportar predicciones
        st.markdown("#### 🔢 Predicciones")

        # Crear DataFrame de predicciones
        predictions_df = pd.DataFrame({
            'Índice': results['y_test'].index,
            'Valor_Real': results['y_test'].values,
            'Valor_Predicho': results['y_test_pred']
        })

        if regression_type == 'ols':
            predictions_df['Error'] = predictions_df['Valor_Real'] - predictions_df['Valor_Predicho']
            predictions_df['Error_Absoluto'] = np.abs(predictions_df['Error'])

        st.dataframe(predictions_df.head(10), use_container_width=True)

        # Botón de descarga
        csv_predictions = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Predicciones (CSV)",
            data=csv_predictions,
            file_name=f"predicciones_{regression_type}_{config['y_name']}.csv",
            mime="text/csv"
        )

        # 2. Exportar coeficientes
        st.markdown("#### 📊 Coeficientes")

        model = results['model']
        feature_names = results['feature_names']

        if regression_type == 'ols':
            coef_df = pd.DataFrame({
                'Variable': ['Intercepto'] + feature_names,
                'Coeficiente': [model.intercept_] + list(model.coef_)
            })

        elif regression_type == 'logistic':
            odds_ratios = np.exp(model.coef_[0])
            coef_df = pd.DataFrame({
                'Variable': ['Intercepto'] + feature_names,
                'Coeficiente': [model.intercept_[0]] + list(model.coef_[0]),
                'Odds_Ratio': [np.exp(model.intercept_[0])] + list(odds_ratios)
            })

        elif regression_type == 'ordinal':
            try:
                coef_df = pd.DataFrame({
                    'Variable': feature_names,
                    'Coeficiente': model.coef_
                })
            except:
                coef_df = pd.DataFrame({
                    'Variable': feature_names,
                    'Coeficiente': [0] * len(feature_names)
                })

        st.dataframe(coef_df, use_container_width=True)

        csv_coef = coef_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Coeficientes (CSV)",
            data=csv_coef,
            file_name=f"coeficientes_{regression_type}_{config['y_name']}.csv",
            mime="text/csv"
        )

        # 3. Exportar métricas
        st.markdown("#### 📈 Métricas de Desempeño")

        metrics = results['metrics']

        if regression_type == 'ols':
            metrics_df = pd.DataFrame({
                'Métrica': ['R²', 'R² Ajustado', 'RMSE', 'MAE'],
                'Entrenamiento': [
                    metrics['train']['r2'],
                    metrics['train']['r2_adjusted'],
                    metrics['train']['rmse'],
                    metrics['train']['mae']
                ],
                'Prueba': [
                    metrics['test']['r2'],
                    metrics['test']['r2_adjusted'],
                    metrics['test']['rmse'],
                    metrics['test']['mae']
                ]
            })

        elif regression_type == 'logistic':
            metrics_df = pd.DataFrame({
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Entrenamiento': [
                    metrics['train']['accuracy'],
                    metrics['train']['precision'],
                    metrics['train']['recall'],
                    metrics['train']['f1'],
                    metrics['train']['roc_auc']
                ],
                'Prueba': [
                    metrics['test']['accuracy'],
                    metrics['test']['precision'],
                    metrics['test']['recall'],
                    metrics['test']['f1'],
                    metrics['test']['roc_auc']
                ]
            })

        elif regression_type == 'ordinal':
            metrics_df = pd.DataFrame({
                'Métrica': ['Accuracy', 'MAE', 'MAE Baseline'],
                'Entrenamiento': [
                    metrics['train']['accuracy'],
                    metrics['train']['mae'],
                    metrics['train']['mae_baseline']
                ],
                'Prueba': [
                    metrics['test']['accuracy'],
                    metrics['test']['mae'],
                    metrics['test']['mae_baseline']
                ]
            })

        st.dataframe(metrics_df, use_container_width=True)

        csv_metrics = metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Métricas (CSV)",
            data=csv_metrics,
            file_name=f"metricas_{regression_type}_{config['y_name']}.csv",
            mime="text/csv"
        )

        # 4. Reporte completo
        st.markdown("#### 📄 Reporte Completo")

        report = f"""
    # Reporte de Regresión - {RegressionSystem.REGRESSION_TYPES[regression_type]['name']}

    ## Configuración del Modelo
    - **Variable Dependiente (Y):** {config['y_name']}
    - **Variables Independientes (X):** {len(config['x_names'])}
    - **Tipo de Regresión:** {regression_type.upper()}
    - **Tamaño de Prueba:** {len(results['y_test'])} observaciones
    - **Tamaño de Entrenamiento:** {len(results['y_train'])} observaciones

    ## Métricas de Desempeño

    ### Conjunto de Prueba
    """

        if regression_type == 'ols':
            report += f"""
    - **R² Score:** {metrics['test']['r2']:.4f}
    - **R² Ajustado:** {metrics['test']['r2_adjusted']:.4f}
    - **RMSE:** {metrics['test']['rmse']:.4f}
    - **MAE:** {metrics['test']['mae']:.4f}
    """
        elif regression_type == 'logistic':
            report += f"""
    - **Accuracy:** {metrics['test']['accuracy']:.4f}
    - **Precision:** {metrics['test']['precision']:.4f}
    - **Recall:** {metrics['test']['recall']:.4f}
    - **F1-Score:** {metrics['test']['f1']:.4f}
    - **ROC-AUC:** {metrics['test']['roc_auc']:.4f}
    """
        elif regression_type == 'ordinal':
            report += f"""
    - **Accuracy:** {metrics['test']['accuracy']:.4f}
    - **MAE:** {metrics['test']['mae']:.4f}
    - **MAE Baseline:** {metrics['test']['mae_baseline']:.4f}
    """

        report += f"""

    ## Variables Más Importantes

    """

        # Agregar top 10 coeficientes
        coef_df_sorted = coef_df.sort_values('Coeficiente', key=abs, ascending=False).head(10)
        for idx, row in coef_df_sorted.iterrows():
            if 'Variable' in row:
                report += f"- **{row['Variable']}:** {row['Coeficiente']:.6f}\n"

        report += f"""

    ## Variables Seleccionadas

    """
        for var in config['x_names']:
            report += f"- {var}\n"

        st.download_button(
            label="📥 Descargar Reporte Completo (TXT)",
            data=report,
            file_name=f"reporte_{regression_type}_{config['y_name']}.txt",
            mime="text/plain"
        )

        # 5. Guardar modelo (opcional)
        st.markdown("#### 💾 Guardar Modelo")

        st.info("""
            ℹ️ Para guardar el modelo entrenado y reutilizarlo más tarde, 
            puedes usar la librería `pickle` o `joblib` de Python.

            Esta funcionalidad se puede implementar si necesitas hacer predicciones 
            sobre nuevos datos en el futuro.
            """)

        if st.button("🔧 Ver código para guardar modelo"):
            st.code("""
    import pickle

    # Guardar el modelo
    with open('modelo_regresion.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Guardar el scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Cargar modelo más tarde
    with open('modelo_regresion.pkl', 'rb') as f:
        modelo_cargado = pickle.load(f)

    # Hacer predicciones
    nuevas_predicciones = modelo_cargado.predict(nuevos_datos)
                """, language='python')

class StatHypothesisTest:

    @staticmethod
    def do(df):
        """Orquestador principal del sistema de pruebas de hipótesis"""

        st.markdown("### 🧪 Pruebas de Hipótesis Estadísticas")

        st.markdown("""
        Este módulo permite comparar dos grupos usando pruebas estadísticas apropiadas.
        El sistema automáticamente:
        1. 🔍 Evalúa normalidad (con transformaciones si es necesario)
        2. 📊 Verifica homogeneidad de varianzas
        3. ✅ Aplica la prueba estadística correcta
        4. 📈 Calcula tamaños de efecto
        """)

        # Verificar que existe columna Transecto
        if "Transecto" not in df.columns:
            st.error("❌ El dataset debe contener una columna 'Transecto' para agrupar los datos.")
            return

        # Paso 1: Selección de variable
        st.markdown("---")
        st.markdown("#### 📋 Paso 1: Seleccionar Variable a Analizar")

        # Obtener columnas numéricas (excluyendo Transecto)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Transecto' in numeric_cols:
            numeric_cols.remove('Transecto')

        if not numeric_cols:
            st.error("❌ No hay columnas numéricas disponibles para análisis.")
            return

        selected_variable = st.selectbox(
            "Selecciona la variable a comparar entre transectos",
            options=numeric_cols,
            help="Variable numérica que se comparará entre los diferentes transectos"
        )

        # Información de la variable
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Variable seleccionada", selected_variable)
        with col2:
            st.metric("Transectos disponibles", df['Transecto'].nunique())
        with col3:
            st.metric("Observaciones totales", len(df[selected_variable].dropna()))

        # Paso 2: Selección de transectos
        st.markdown("---")
        st.markdown("#### 🎯 Paso 2: Seleccionar Transectos a Comparar")

        transects = sorted(df['Transecto'].unique())

        if len(transects) < 2:
            st.error("❌ Se necesitan al menos 2 transectos para comparar.")
            return

        col1, col2 = st.columns(2)

        with col1:
            transect1 = st.selectbox(
                "Transecto 1 (Grupo de referencia)",
                options=transects,
                index=0
            )

        with col2:
            available_transects2 = [t for t in transects if t != transect1]
            transect2 = st.selectbox(
                "Transecto 2 (Grupo de comparación)",
                options=available_transects2,
                index=0 if available_transects2 else None
            )

        # Extraer grupos
        group1 = pd.to_numeric(df[df["Transecto"] == transect1][selected_variable], errors="coerce").dropna()
        group2 = pd.to_numeric(df[df["Transecto"] == transect2][selected_variable], errors="coerce").dropna()

        # Validar tamaños
        if len(group1) < 3 or len(group2) < 3:
            st.error("❌ Cada grupo debe tener al menos 3 observaciones válidas.")
            return

        # Mostrar información de grupos
        st.markdown("##### 📊 Información de los Grupos")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Transecto {transect1}**")
            st.metric("N", len(group1))
            st.metric("Media", f"{group1.mean():.2f}")
            st.metric("Desv. Std", f"{group1.std():.2f}")

        with col2:
            st.markdown(f"**Transecto {transect2}**")
            st.metric("N", len(group2))
            st.metric("Media", f"{group2.mean():.2f}")
            st.metric("Desv. Std", f"{group2.std():.2f}")

        # Visualización previa
        with st.expander("📊 Vista Previa de los Datos", expanded=False):
            fig = go.Figure()

            fig.add_trace(go.Box(
                y=group1,
                name=f'Transecto {transect1}',
                marker_color='lightblue'
            ))

            fig.add_trace(go.Box(
                y=group2,
                name=f'Transecto {transect2}',
                marker_color='lightcoral'
            ))

            fig.update_layout(
                title=f"Distribución de {selected_variable}",
                yaxis_title=selected_variable,
                showlegend=True,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Paso 3: Configuración
        st.markdown("---")
        st.markdown("#### ⚙️ Paso 3: Configuración")

        col1, col2 = st.columns(2)

        with col1:
            alpha = st.slider(
                "Nivel de significancia (α)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                help="Probabilidad de rechazar H0 cuando es verdadera (típicamente 0.05)"
            )

        with col2:
            show_plots = st.checkbox(
                "Mostrar visualizaciones detalladas",
                value=True,
                help="Incluye gráficos de normalidad, homogeneidad y efectos"
            )

        # Botón de análisis
        st.markdown("---")
        if st.button("🚀 Ejecutar Análisis Completo", type="primary", use_container_width=True):

            with st.spinner("Ejecutando análisis estadístico..."):

                # ═══════════════════════════════════════
                # FASE 1: ANÁLISIS DE NORMALIDAD
                # ═══════════════════════════════════════
                st.markdown("---")
                st.markdown("## 📊 Fase 1: Análisis de Normalidad")

                with st.expander("ℹ️ ¿Qué es la prueba de normalidad?", expanded=False):
                    st.markdown("""
                    La **Prueba de D'Agostino-Pearson** evalúa si los datos siguen una distribución normal.

                    **¿Por qué es importante?**
                    - Las pruebas paramétricas (t-test) asumen normalidad
                    - Si los datos no son normales, usamos pruebas no paramétricas (Mann-Whitney U)

                    **Transformaciones disponibles:**
                    1. **Original**: Datos sin transformar
                    2. **Logarítmica**: Para datos con sesgo positivo
                    3. **Raíz cuadrada**: Para datos con varianza proporcional a la media
                    4. **Box-Cox**: Transformación óptima automática
                    """)

                g1_transformed, g2_transformed, transformation, norm_params, is_normal = \
                    StatHypothesisTest.analize_distribution_normality(
                        group1.values, group2.values, alpha=alpha, plot=show_plots
                    )

                # Mostrar resultado de normalidad
                if is_normal:
                    st.success(f"""
                    ✅ **Normalidad Alcanzada**

                    - Transformación utilizada: **{transformation.upper()}**
                    - p-valor Grupo 1: {norm_params['p1']:.6f}
                    - p-valor Grupo 2: {norm_params['p2']:.6f}

                    Ambos grupos cumplen con la asunción de normalidad (p > {alpha})
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **Normalidad NO Alcanzada**

                    - Ninguna transformación logró normalidad
                    - Se usarán datos originales
                    - Se aplicará prueba **no paramétrica** (Mann-Whitney U)
                    """)

                # ═══════════════════════════════════════
                # FASE 2: HOMOGENEIDAD DE VARIANZAS
                # ═══════════════════════════════════════
                st.markdown("---")
                st.markdown("## 📊 Fase 2: Homogeneidad de Varianzas")

                with st.expander("ℹ️ ¿Qué es la prueba de homogeneidad?", expanded=False):
                    st.markdown("""
                    La **Prueba de Levene** evalúa si las varianzas de ambos grupos son iguales.

                    **¿Por qué es importante?**
                    - El t-test estándar asume varianzas iguales
                    - Si las varianzas son diferentes, usamos el test de Welch

                    **Interpretación:**
                    - p > α: Varianzas homogéneas → t-test estándar
                    - p < α: Varianzas heterogéneas → Welch's t-test
                    """)

                stat_levene, p_levene, is_homogeneous = \
                    StatHypothesisTest.analize_distributions_homogeneity(
                        g1_transformed, g2_transformed, alpha=alpha, plot=show_plots
                    )

                # Mostrar resultado de homogeneidad
                if is_homogeneous:
                    st.success(f"""
                    ✅ **Varianzas Homogéneas**

                    - Estadístico de Levene: {stat_levene:.4f}
                    - p-valor: {p_levene:.6f}

                    Las varianzas son similares entre grupos (p > {alpha})
                    """)
                else:
                    st.info(f"""
                    ℹ️ **Varianzas Heterogéneas**

                    - Estadístico de Levene: {stat_levene:.4f}
                    - p-valor: {p_levene:.6f}

                    Las varianzas difieren significativamente (p < {alpha})
                    Se ajustará la prueba estadística
                    """)

                # ═══════════════════════════════════════
                # FASE 3: PRUEBA ESTADÍSTICA
                # ═══════════════════════════════════════
                st.markdown("---")
                st.markdown("## 📊 Fase 3: Prueba de Hipótesis")

                # Determinar prueba a usar
                if is_normal and is_homogeneous:
                    test_name = "t-test Independiente"
                    test_description = "Prueba paramétrica para comparar medias con varianzas homogéneas"
                elif is_normal and not is_homogeneous:
                    test_name = "Welch's t-test"
                    test_description = "Prueba paramétrica para comparar medias con varianzas heterogéneas"
                else:
                    test_name = "Mann-Whitney U"
                    test_description = "Prueba no paramétrica para comparar distribuciones"

                st.info(f"""
                **Prueba seleccionada:** {test_name}

                {test_description}
                """)

                with st.expander("ℹ️ Interpretación de la prueba", expanded=False):
                    st.markdown(f"""
                    **Hipótesis:**
                    - H₀: No hay diferencia entre los grupos
                    - H₁: Existe diferencia significativa entre los grupos

                    **Criterio de decisión:**
                    - Si p-valor < {alpha}: Rechazamos H₀ (hay diferencia significativa)
                    - Si p-valor ≥ {alpha}: No rechazamos H₀ (no hay evidencia de diferencia)
                    """)

                test_results = StatHypothesisTest.execute_statistical_test(
                    g1_transformed, g2_transformed,
                    normalidad_positiva=1 if is_normal else 0,
                    homogeneidad_positiva=1 if is_homogeneous else 0,
                    transformacion=transformation,
                    params=norm_params,
                    alpha=alpha,
                    plot=show_plots
                )

                # ═══════════════════════════════════════
                # RESULTADOS FINALES
                # ═══════════════════════════════════════
                st.markdown("---")
                st.markdown("## 🎯 Resultados Finales")

                # Decisión estadística
                is_significant = test_results['p_value'] < alpha

                if is_significant:
                    st.success(f"""
                    ### ✅ Diferencia Estadísticamente Significativa

                    **Conclusión:** Existe evidencia suficiente para afirmar que hay una diferencia 
                    significativa en {selected_variable} entre Transecto {transect1} y Transecto {transect2}.

                    **p-valor:** {test_results['p_value']:.6f} < {alpha}
                    """)
                else:
                    st.warning(f"""
                    ### ❌ Sin Diferencia Estadísticamente Significativa

                    **Conclusión:** No hay evidencia suficiente para afirmar que existe una diferencia 
                    en {selected_variable} entre Transecto {transect1} y Transecto {transect2}.

                    **p-valor:** {test_results['p_value']:.6f} ≥ {alpha}
                    """)

                # Tabla de resultados
                st.markdown("---")
                st.markdown("### 📋 Resumen de Estadísticos")

                results_df = pd.DataFrame({
                    'Métrica': [
                        'Estadístico de Prueba',
                        'p-valor',
                        'Hodges-Lehmann',
                        'IC 95% (inferior)',
                        'IC 95% (superior)',
                        "Cohen's d",
                        "Rosenthal's r"
                    ],
                    'Valor': [
                        f"{test_results['stat']:.4f}",
                        f"{test_results['p_value']:.6f}",
                        f"{test_results['hodges_lehmann']:.4f}",
                        f"{test_results['hl_confidence_interval'][0]:.4f}",
                        f"{test_results['hl_confidence_interval'][1]:.4f}",
                        f"{test_results['cohens_d']:.4f}",
                        f"{test_results['rosenthal_r']:.4f}"
                    ],
                    'Interpretación': [
                        f"Valor del test {test_name}",
                        "Significativo" if is_significant else "No significativo",
                        "Diferencia mediana estimada",
                        "Límite inferior del IC",
                        "Límite superior del IC",
                        StatHypothesisTest._interpret_cohens_d(test_results['cohens_d']),
                        StatHypothesisTest._interpret_rosenthal_r(test_results['rosenthal_r'])
                    ]
                })

                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # Interpretación de tamaños de efecto
                st.markdown("---")
                st.markdown("### 📏 Tamaños de Efecto")

                col1, col2 = st.columns(2)

                with col1:
                    cohens_interpretation = StatHypothesisTest._interpret_cohens_d(test_results['cohens_d'])
                    st.metric(
                        "Cohen's d",
                        f"{test_results['cohens_d']:.3f}",
                        cohens_interpretation
                    )
                    st.caption("Mide la diferencia estandarizada entre medias")

                with col2:
                    rosenthal_interpretation = StatHypothesisTest._interpret_rosenthal_r(test_results['rosenthal_r'])
                    st.metric(
                        "Rosenthal's r",
                        f"{test_results['rosenthal_r']:.3f}",
                        rosenthal_interpretation
                    )
                    st.caption("Correlación entre grupo y variable")

                # Exportar resultados
                st.markdown("---")
                st.markdown("### 📥 Exportar Resultados")

                export_data = {
                    'Variable': selected_variable,
                    'Transecto_1': transect1,
                    'Transecto_2': transect2,
                    'N_Transecto_1': len(group1),
                    'N_Transecto_2': len(group2),
                    'Media_Transecto_1': group1.mean(),
                    'Media_Transecto_2': group2.mean(),
                    'Transformacion': transformation,
                    'Normalidad': 'Sí' if is_normal else 'No',
                    'Homogeneidad': 'Sí' if is_homogeneous else 'No',
                    'Prueba_Utilizada': test_name,
                    'Estadistico': test_results['stat'],
                    'p_valor': test_results['p_value'],
                    'Significativo': 'Sí' if is_significant else 'No',
                    'Hodges_Lehmann': test_results['hodges_lehmann'],
                    'IC_95_inferior': test_results['hl_confidence_interval'][0],
                    'IC_95_superior': test_results['hl_confidence_interval'][1],
                    'Cohens_d': test_results['cohens_d'],
                    'Rosenthal_r': test_results['rosenthal_r'],
                    'Alpha': alpha
                }

                export_df = pd.DataFrame([export_data])
                csv = export_df.to_csv(index=False)

                st.download_button(
                    label="📥 Descargar Resultados (CSV)",
                    data=csv,
                    file_name=f"prueba_hipotesis_{selected_variable}_{transect1}_vs_{transect2}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.success("✅ Análisis completado exitosamente!")

    # ═══════════════════════════════════════════════════════════
    # MÉTODOS AUXILIARES
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _interpret_cohens_d(d):
        """Interpreta el tamaño de efecto de Cohen's d"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Muy pequeño"
        elif abs_d < 0.5:
            return "Pequeño"
        elif abs_d < 0.8:
            return "Mediano"
        else:
            return "Grande"

    @staticmethod
    def _interpret_rosenthal_r(r):
        """Interpreta el tamaño de efecto de Rosenthal's r"""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "Muy pequeño"
        elif abs_r < 0.3:
            return "Pequeño"
        elif abs_r < 0.5:
            return "Mediano"
        else:
            return "Grande"

    @staticmethod
    def get_transect_groups(df, question_col):
        transect_groups = {
            t: pd.to_numeric(df[df["Transecto"] == t][question_col], errors="coerce").dropna()
            for t in sorted(df["Transecto"].unique())
        }
        group_names = [f"Transecto {t}" for t in transect_groups.keys()]
        groups = list(transect_groups.values())
        return transect_groups, group_names, groups

    @staticmethod
    def analize_distribution_normality(group1, group2, alpha=0.05, plot=True):
        """Análisis de normalidad con transformaciones"""

        # Contenedor para logs
        with st.expander("📝 Log Detallado de Normalidad", expanded=False):
            log_container = st.empty()
            logs = []

            def add_log(message):
                logs.append(message)
                log_container.code('\n'.join(logs))

            add_log("=" * 55)
            add_log("BÚSQUEDA DE NORMALIDAD")
            add_log("=" * 55)

            # 1. Datos originales
            add_log("\n1. Análisis de datos originales:")
            _, p1_orig = normaltest(group1)
            _, p2_orig = normaltest(group2)
            normal1_orig = p1_orig > alpha
            normal2_orig = p2_orig > alpha

            add_log(f"   Grupo 1: p={p1_orig:.6f} {'✅' if normal1_orig else '❌'}")
            add_log(f"   Grupo 2: p={p2_orig:.6f} {'✅' if normal2_orig else '❌'}")

            if normal1_orig and normal2_orig:
                add_log("\n✅ Ambos grupos cumplen normalidad - Datos originales")
                if plot:
                    StatHypothesisTest._plot_all_normality_tests(group1, group2, alpha)
                    all_results = {'Original': (p1_orig, p2_orig, True)}
                    StatHypothesisTest._plot_normality_summary(all_results, alpha)
                return group1, group2, "original", {"p1": p1_orig, "p2": p2_orig}, True

            # 2. Transformación logarítmica
            add_log("\n2. Transformación logarítmica:")
            try:
                min_val = min(np.min(group1), np.min(group2))
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    g1_log = np.log(group1 + shift)
                    g2_log = np.log(group2 + shift)
                    add_log(f"   Constante añadida: +{shift} (para evitar log ≤ 0)")
                else:
                    g1_log = np.log(group1)
                    g2_log = np.log(group2)
                    shift = 0

                _, p1_log = normaltest(g1_log)
                _, p2_log = normaltest(g2_log)
                normal1_log = p1_log > alpha
                normal2_log = p2_log > alpha

                add_log(f"   Grupo 1: p={p1_log:.6f} {'✅' if normal1_log else '❌'}")
                add_log(f"   Grupo 2: p={p2_log:.6f} {'✅' if normal2_log else '❌'}")

                if normal1_log and normal2_log:
                    add_log("\n✅ Ambos grupos cumplen normalidad - Transformación logarítmica")
                    if plot:
                        StatHypothesisTest._plot_all_normality_tests(group1, group2, alpha)
                        all_results = {
                            'Original': (p1_orig, p2_orig, normal1_orig and normal2_orig),
                            'Log': (p1_log, p2_log, True)
                        }
                        StatHypothesisTest._plot_normality_summary(all_results, alpha)
                    return g1_log, g2_log, "log", {"p1": p1_log, "p2": p2_log, "shift": shift}, True

            except Exception as e:
                add_log(f"   Error en transformación logarítmica: {e}")
                p1_log, p2_log = np.nan, np.nan

            # 3. Transformación raíz cuadrada
            add_log("\n3. Transformación raíz cuadrada:")
            try:
                min_val = min(np.min(group1), np.min(group2))
                if min_val < 0:
                    shift = abs(min_val)
                    g1_sqrt = np.sqrt(group1 + shift)
                    g2_sqrt = np.sqrt(group2 + shift)
                    add_log(f"   Constante añadida: +{shift} (para evitar √negativo)")
                else:
                    g1_sqrt = np.sqrt(group1)
                    g2_sqrt = np.sqrt(group2)
                    shift = 0

                _, p1_sqrt = normaltest(g1_sqrt)
                _, p2_sqrt = normaltest(g2_sqrt)
                normal1_sqrt = p1_sqrt > alpha
                normal2_sqrt = p2_sqrt > alpha

                add_log(f"   Grupo 1: p={p1_sqrt:.6f} {'✅' if normal1_sqrt else '❌'}")
                add_log(f"   Grupo 2: p={p2_sqrt:.6f} {'✅' if normal2_sqrt else '❌'}")

                if normal1_sqrt and normal2_sqrt:
                    add_log("\n✅ Ambos grupos cumplen normalidad - Transformación raíz cuadrada")
                    if plot:
                        StatHypothesisTest._plot_all_normality_tests(group1, group2, alpha)
                        all_results = {
                            'Original': (p1_orig, p2_orig, normal1_orig and normal2_orig),
                            'Log': (p1_log if not np.isnan(p1_log) else 0, p2_log if not np.isnan(p2_log) else 0,
                                    not np.isnan(p1_log) and not np.isnan(
                                        p2_log) and p1_log > alpha and p2_log > alpha),
                            'Sqrt': (p1_sqrt, p2_sqrt, True)
                        }
                        StatHypothesisTest._plot_normality_summary(all_results, alpha)
                    return g1_sqrt, g2_sqrt, "sqrt", {"p1": p1_sqrt, "p2": p2_sqrt, "shift": shift}, True

            except Exception as e:
                add_log(f"   Error en transformación raíz cuadrada: {e}")
                p1_sqrt, p2_sqrt = np.nan, np.nan

            # 4. Transformación Box-Cox
            add_log("\n4. Transformación Box-Cox:")
            try:
                min_val = min(np.min(group1), np.min(group2))
                if min_val <= 0:
                    shift = abs(min_val) + 0.1
                    g1_shifted = group1 + shift
                    g2_shifted = group2 + shift
                    add_log(f"   Constante añadida: +{shift} (Box-Cox requiere valores > 0)")
                else:
                    g1_shifted = group1
                    g2_shifted = group2
                    shift = 0

                g1_boxcox, lambda1 = boxcox(g1_shifted)
                g2_boxcox, lambda2 = boxcox(g2_shifted)

                add_log(f"   Lambda Grupo 1: {lambda1:.4f}")
                add_log(f"   Lambda Grupo 2: {lambda2:.4f}")

                _, p1_boxcox = normaltest(g1_boxcox)
                _, p2_boxcox = normaltest(g2_boxcox)
                normal1_boxcox = p1_boxcox > alpha
                normal2_boxcox = p2_boxcox > alpha

                add_log(f"   Grupo 1: p={p1_boxcox:.6f} {'✅' if normal1_boxcox else '❌'}")
                add_log(f"   Grupo 2: p={p2_boxcox:.6f} {'✅' if normal2_boxcox else '❌'}")

                if normal1_boxcox and normal2_boxcox:
                    add_log("\n✅ Ambos grupos cumplen normalidad - Transformación Box-Cox")
                    if plot:
                        StatHypothesisTest._plot_all_normality_tests(group1, group2, alpha)
                        all_results = {
                            'Original': (p1_orig, p2_orig, normal1_orig and normal2_orig),
                            'Log': (p1_log if not np.isnan(p1_log) else 0, p2_log if not np.isnan(p2_log) else 0,
                                    not np.isnan(p1_log) and not np.isnan(
                                        p2_log) and p1_log > alpha and p2_log > alpha),
                            'Sqrt': (p1_sqrt if not np.isnan(p1_sqrt) else 0, p2_sqrt if not np.isnan(p2_sqrt) else 0,
                                     not np.isnan(p1_sqrt) and not np.isnan(
                                         p2_sqrt) and p1_sqrt > alpha and p2_sqrt > alpha),
                            'Box-Cox': (p1_boxcox, p2_boxcox, True)
                        }
                        StatHypothesisTest._plot_normality_summary(all_results, alpha)
                    return g1_boxcox, g2_boxcox, "boxcox", {
                        "p1": p1_boxcox, "p2": p2_boxcox,
                        "lambda1": lambda1, "lambda2": lambda2, "shift": shift
                    }, True

            except Exception as e:
                add_log(f"   Error en transformación Box-Cox: {e}")
                p1_boxcox, p2_boxcox = np.nan, np.nan

            # Ninguna transformación válida
            add_log("\n❌ Ninguna transformación logró normalidad")
            add_log("\nResumen:")
            add_log(f"   Original:  G1={'✅' if normal1_orig else '❌'} G2={'✅' if normal2_orig else '❌'}")
            add_log(
                f"   Log:       G1={'✅' if not np.isnan(p1_log) and p1_log > alpha else '❌'} G2={'✅' if not np.isnan(p2_log) and p2_log > alpha else '❌'}")
            add_log(
                f"   Sqrt:      G1={'✅' if not np.isnan(p1_sqrt) and p1_sqrt > alpha else '❌'} G2={'✅' if not np.isnan(p2_sqrt) and p2_sqrt > alpha else '❌'}")
            add_log(
                f"   Box-Cox:   G1={'✅' if not np.isnan(p1_boxcox) and p1_boxcox > alpha else '❌'} G2={'✅' if not np.isnan(p2_boxcox) and p2_boxcox > alpha else '❌'}")
            add_log("\n➡️ Se usarán datos originales con Mann-Whitney U")

            if plot:
                StatHypothesisTest._plot_all_normality_tests(group1, group2, alpha)
                all_results = {
                    'Original': (p1_orig, p2_orig, normal1_orig and normal2_orig),
                    'Log': (p1_log if not np.isnan(p1_log) else 0, p2_log if not np.isnan(p2_log) else 0,
                            not np.isnan(p1_log) and not np.isnan(p2_log) and p1_log > alpha and p2_log > alpha),
                    'Sqrt': (p1_sqrt if not np.isnan(p1_sqrt) else 0, p2_sqrt if not np.isnan(p2_sqrt) else 0,
                             not np.isnan(p1_sqrt) and not np.isnan(p2_sqrt) and p1_sqrt > alpha and p2_sqrt > alpha),
                    'Box-Cox': (p1_boxcox if not np.isnan(p1_boxcox) else 0,
                                p2_boxcox if not np.isnan(p2_boxcox) else 0,
                                not np.isnan(p1_boxcox) and not np.isnan(
                                    p2_boxcox) and p1_boxcox > alpha and p2_boxcox > alpha)
                }
                StatHypothesisTest._plot_normality_summary(all_results, alpha)

        return group1, group2, "original", {"p1": p1_orig, "p2": p2_orig}, False

    @staticmethod
    def analize_distributions_homogeneity(group1, group2, alpha=0.05, plot=True):
        """Análisis de homogeneidad de varianzas"""

        stat_levene, p_levene = levene(group1, group2)
        homogeneous = p_levene > alpha

        if plot:
            StatHypothesisTest._plot_variance_homogeneity(group1, group2, stat_levene, p_levene, homogeneous)

        return stat_levene, p_levene, homogeneous

    @staticmethod
    def execute_statistical_test(group1, group2, normalidad_positiva, homogeneidad_positiva, transformacion, params,
                                 alpha=0.05, plot=True):
        """Ejecuta la prueba estadística apropiada y calcula métricas"""

        def inverse_transform(data, transform_type, params, group_num=1):
            """Revierte la transformación aplicada"""
            if transform_type == "original":
                return data

            elif transform_type == "log":
                shift = params.get("shift", 0)
                return np.exp(data) - shift

            elif transform_type == "sqrt":
                shift = params.get("shift", 0)
                return data ** 2 - shift

            elif transform_type == "boxcox":
                lambda_key = f"lambda{group_num}"
                lambda_val = params[lambda_key]
                shift = params.get("shift", 0)

                if lambda_val == 0:
                    original = np.exp(data)
                else:
                    original = np.power(lambda_val * data + 1, 1 / lambda_val)

                return original - shift

            else:
                raise ValueError(f"Unknown transformation type: {transform_type}")

        def statistical_difference_significance(group1, group2, normalidad_positiva, homogeneidad_positiva):
            """Ejecuta la prueba estadística apropiada"""

            with st.expander("📝 Log de Prueba Estadística", expanded=False):
                if normalidad_positiva == 0:
                    stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                    is_sig = p_val < alpha
                    st.code(f"""
Mann-Whitney U Test:
   U-statistic: {stat:.4f}
   p-value: {p_val:.6f}
   Resultado: {'✅ Diferencia significativa' if is_sig else '❌ Sin diferencia significativa'}
                    """)
                    return stat, p_val

                elif normalidad_positiva == 1 and homogeneidad_positiva == 1:
                    stat, p_val = ttest_ind(group1, group2)
                    is_sig = p_val < alpha
                    st.code(f"""
Independent t-test:
   t-statistic: {stat:.4f}
   p-value: {p_val:.6f}
   Resultado: {'✅ Diferencia significativa' if is_sig else '❌ Sin diferencia significativa'}
                    """)
                    return stat, p_val

                elif normalidad_positiva == 1 and homogeneidad_positiva == 0:
                    stat, p_val = ttest_ind(group1, group2, equal_var=False)
                    is_sig = p_val < alpha
                    st.code(f"""
Welch's t-test:
   t-statistic: {stat:.4f}
   p-value: {p_val:.6f}
   Resultado: {'✅ Diferencia significativa' if is_sig else '❌ Sin diferencia significativa'}
                    """)
                    return stat, p_val

        # Ejecutar prueba estadística
        stat, p_val = statistical_difference_significance(group1, group2, normalidad_positiva, homogeneidad_positiva)

        # Plot p-value distribution
        if plot:
            test_type = StatHypothesisTest._determine_test_type(normalidad_positiva, homogeneidad_positiva)
            StatHypothesisTest._plot_pvalue_distribution(stat, p_val, test_type, alpha, len(group1), len(group2))

        # Revertir a escala original
        if transformacion != "original":
            with st.expander("🔄 Reversión de Transformación", expanded=False):
                st.info(f"Revirtiendo transformación '{transformacion}' para calcular métricas en escala original...")
            group1_original = inverse_transform(group1, transformacion, params, group_num=1)
            group2_original = inverse_transform(group2, transformacion, params, group_num=2)
        else:
            group1_original = group1
            group2_original = group2

        def measure_difference(group1, group2):
            """Estimador de diferencia (Hodges-Lehmann)"""

            def hodges_lehmann(group1, group2):
                x = np.asarray(group1)
                y = np.asarray(group2)
                m, n = len(x), len(y)

                # Hodges-Lehmann estimator
                pairwise_diffs = np.subtract.outer(x, y).ravel()
                hl = np.median(pairwise_diffs)

                # 95% CI por bootstrap
                rng = np.random.default_rng(12345)
                n_boot = 5000
                boots = np.empty(n_boot)
                for i in range(n_boot):
                    bx = rng.choice(x, size=m, replace=True)
                    by = rng.choice(y, size=n, replace=True)
                    boots[i] = np.median(np.subtract.outer(bx, by).ravel())

                alpha_ci = 0.05
                ci_lower, ci_upper = np.percentile(boots, [100 * alpha_ci / 2, 100 * (1 - alpha_ci / 2)])

                with st.expander("📊 Hodges-Lehmann (Escala Original)", expanded=False):
                    st.code(f"""
Hodges–Lehmann = {hl:.4f}
Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]
                    """)

                return hl, (ci_lower, ci_upper)

            return hodges_lehmann(group1, group2)

        def measure_effect(group1, group2, stat):
            """Tamaño de efecto (Cohen's d, Rosenthal's r)"""
            n1, n2 = len(group1), len(group2)
            df = n1 + n2 - 2
            r_rosenthal = np.sqrt(stat ** 2 / (stat ** 2 + df))

            mean_diff = np.mean(group1) - np.mean(group2)
            s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
            cohens_d = mean_diff / pooled_std

            with st.expander("📊 Tamaños de Efecto (Escala Original)", expanded=False):
                st.code(f"""
Cohen's d = {cohens_d:.4f}
Rosenthal's r = {r_rosenthal:.4f}
                """)

            return cohens_d, r_rosenthal

        # Calcular métricas en escala original
        hl, hl_ci = measure_difference(group1_original, group2_original)
        cohens_d, r_rosenthal = measure_effect(group1_original, group2_original, stat)

        # Plot effect sizes y Hodges-Lehmann
        if plot:
            StatHypothesisTest._plot_effect_sizes(group1_original, group2_original, cohens_d, r_rosenthal)

        # Retornar resultados
        return {
            'stat': stat,
            'p_value': p_val,
            'hodges_lehmann': hl,
            'hl_confidence_interval': hl_ci,
            'cohens_d': cohens_d,
            'rosenthal_r': r_rosenthal
        }

    @staticmethod
    def _determine_test_type(normalidad_positiva, homogeneidad_positiva):
        """Determina qué prueba se usó"""
        if normalidad_positiva == 0:
            return "Mann-Whitney U"
        elif normalidad_positiva == 1 and homogeneidad_positiva == 1:
            return "Independent t-test"
        elif normalidad_positiva == 1 and homogeneidad_positiva == 0:
            return "Welch's t-test"

    # ═══════════════════════════════════════════════════════════
    # MÉTODOS DE VISUALIZACIÓN
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _plot_variance_homogeneity(group1, group2, stat_levene, p_levene, homogeneous):
        """Boxplot para comparar homogeneidad de varianzas"""
        fig, ax = plt.subplots(figsize=(10, 6))

        bp = plt.boxplot([group1, group2], labels=['Group 1', 'Group 2'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)

        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        max1, max2 = np.max(group1), np.max(group2)

        plt.text(1, max1 + (max1 - np.min(group1)) * 0.05, f'Var: {var1:.3f}',
                 ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.text(2, max2 + (max2 - np.min(group2)) * 0.05, f'Var: {var2:.3f}',
                 ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

        result_text = f"Levene's Test\nStatistic: {stat_levene:.4f}\np-value: {p_levene:.6f}\n{'✅ Homogeneous' if homogeneous else '❌ Heterogeneous'}"
        plt.text(1.5, plt.ylim()[1] * 0.9, result_text, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

        plt.title('Variance Homogeneity Test', fontsize=14, fontweight='bold')
        plt.ylabel('Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    @staticmethod
    def _plot_pvalue_distribution(stat, p_val, test_type, alpha, n1, n2):
        """Curva de campana con p-value para cada prueba"""
        fig = plt.figure(figsize=(12, 6))

        if test_type == "Mann-Whitney U":
            x = np.linspace(-4, 4, 1000)
            y = norm.pdf(x, 0, 1)

            mean_u = n1 * n2 / 2
            std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z_score = (stat - mean_u) / std_u

            plt.plot(x, y, 'b-', linewidth=2, label='Standard Normal')
            plt.axvline(z_score, color='red', linewidth=3, label=f'Observed z: {z_score:.3f}')

            z_crit = norm.ppf(1 - alpha / 2)
            plt.axvline(-z_crit, color='red', linestyle='--', alpha=0.7)
            plt.axvline(z_crit, color='red', linestyle='--', alpha=0.7)

            x_left = x[x <= -z_crit]
            x_right = x[x >= z_crit]
            plt.fill_between(x_left, norm.pdf(x_left, 0, 1), alpha=0.3, color='red')
            plt.fill_between(x_right, norm.pdf(x_right, 0, 1), alpha=0.3, color='red')

        else:  # t-tests
            df = n1 + n2 - 2
            x = np.linspace(-4, 4, 1000)
            y = t.pdf(x, df)

            plt.plot(x, y, 'b-', linewidth=2, label=f't-distribution (df={df})')
            plt.axvline(stat, color='red', linewidth=3, label=f'Observed t: {stat:.3f}')

            t_crit = t.ppf(1 - alpha / 2, df)
            plt.axvline(-t_crit, color='red', linestyle='--', alpha=0.7)
            plt.axvline(t_crit, color='red', linestyle='--', alpha=0.7)

            x_left = x[x <= -t_crit]
            x_right = x[x >= t_crit]
            plt.fill_between(x_left, t.pdf(x_left, df), alpha=0.3, color='red')
            plt.fill_between(x_right, t.pdf(x_right, df), alpha=0.3, color='red')

        conclusion = "✅ Diferencia significativa" if p_val < alpha else "❌ No hay Diferencia significativa"
        plt.title(f'{test_type}\np-value = {p_val:.5f}', fontsize=14, fontweight='bold')
        plt.xlabel('Test Statistic')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.text(0.05, 0.9, conclusion, transform=plt.gca().transAxes,
                 fontsize=12, fontweight='bold', color='green' if p_val < alpha else 'red')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    @staticmethod
    def _plot_all_normality_tests(group1, group2, alpha):
        """Plot detallado de pruebas de normalidad para todas las transformaciones intentadas"""

        transformations = {}

        # Original
        stat1_orig, p1_orig = normaltest(group1)
        stat2_orig, p2_orig = normaltest(group2)
        transformations['Original'] = {
            'data1': group1, 'data2': group2,
            'stat1': stat1_orig, 'stat2': stat2_orig,
            'p1': p1_orig, 'p2': p2_orig
        }

        # Log
        try:
            min_val = min(np.min(group1), np.min(group2))
            if min_val <= 0:
                shift = abs(min_val) + 1
                g1_log = np.log(group1 + shift)
                g2_log = np.log(group2 + shift)
            else:
                g1_log = np.log(group1)
                g2_log = np.log(group2)

            stat1_log, p1_log = normaltest(g1_log)
            stat2_log, p2_log = normaltest(g2_log)
            transformations['Log'] = {
                'data1': g1_log, 'data2': g2_log,
                'stat1': stat1_log, 'stat2': stat2_log,
                'p1': p1_log, 'p2': p2_log
            }
        except:
            transformations['Log'] = None

        # Sqrt
        try:
            min_val = min(np.min(group1), np.min(group2))
            if min_val < 0:
                shift = abs(min_val)
                g1_sqrt = np.sqrt(group1 + shift)
                g2_sqrt = np.sqrt(group2 + shift)
            else:
                g1_sqrt = np.sqrt(group1)
                g2_sqrt = np.sqrt(group2)

            stat1_sqrt, p1_sqrt = normaltest(g1_sqrt)
            stat2_sqrt, p2_sqrt = normaltest(g2_sqrt)
            transformations['Sqrt'] = {
                'data1': g1_sqrt, 'data2': g2_sqrt,
                'stat1': stat1_sqrt, 'stat2': stat2_sqrt,
                'p1': p1_sqrt, 'p2': p2_sqrt
            }
        except:
            transformations['Sqrt'] = None

        # Box-Cox
        try:
            min_val = min(np.min(group1), np.min(group2))
            if min_val <= 0:
                shift = abs(min_val) + 0.1
                g1_shifted = group1 + shift
                g2_shifted = group2 + shift
            else:
                g1_shifted = group1
                g2_shifted = group2

            g1_boxcox, _ = boxcox(g1_shifted)
            g2_boxcox, _ = boxcox(g2_shifted)

            stat1_boxcox, p1_boxcox = normaltest(g1_boxcox)
            stat2_boxcox, p2_boxcox = normaltest(g2_boxcox)
            transformations['Box-Cox'] = {
                'data1': g1_boxcox, 'data2': g2_boxcox,
                'stat1': stat1_boxcox, 'stat2': stat2_boxcox,
                'p1': p1_boxcox, 'p2': p2_boxcox
            }
        except:
            transformations['Box-Cox'] = None

        # Crear plots para cada transformación
        valid_transforms = {k: v for k, v in transformations.items() if v is not None}
        n_transforms = len(valid_transforms)

        fig, axes = plt.subplots(n_transforms, 2, figsize=(15, 5 * n_transforms))
        if n_transforms == 1:
            axes = axes.reshape(1, -1)

        # Distribución Chi-cuadrado
        x = np.linspace(0, 15, 1000)
        y = chi2.pdf(x, df=2)
        chi2_crit = chi2.ppf(1 - alpha, df=2)
        x_reject = x[x >= chi2_crit]

        for i, (name, data) in enumerate(valid_transforms.items()):
            # Grupo 1
            axes[i, 0].plot(x, y, 'b-', linewidth=2, label='Chi-square (df=2)')
            axes[i, 0].axvline(data['stat1'], color='red', linewidth=3, label=f'Observed: {data["stat1"]:.3f}')
            axes[i, 0].axvline(chi2_crit, color='red', linestyle='--', alpha=0.7, label=f'Critical: {chi2_crit:.3f}')
            axes[i, 0].fill_between(x_reject, chi2.pdf(x_reject, df=2), alpha=0.3, color='red')

            result1 = "✅ Normal" if data['p1'] > alpha else "❌ Non-normal"
            axes[i, 0].set_title(f'Group 1 - {name}\np-value = {data["p1"]:.6f}\n{result1}')
            axes[i, 0].set_xlabel('Test Statistic')
            axes[i, 0].set_ylabel('Density')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)

            # Grupo 2
            axes[i, 1].plot(x, y, 'b-', linewidth=2, label='Chi-square (df=2)')
            axes[i, 1].axvline(data['stat2'], color='red', linewidth=3, label=f'Observed: {data["stat2"]:.3f}')
            axes[i, 1].axvline(chi2_crit, color='red', linestyle='--', alpha=0.7, label=f'Critical: {chi2_crit:.3f}')
            axes[i, 1].fill_between(x_reject, chi2.pdf(x_reject, df=2), alpha=0.3, color='red')

            result2 = "✅ Normal" if data['p2'] > alpha else "❌ Non-normal"
            axes[i, 1].set_title(f'Group 2 - {name}\np-value = {data["p2"]:.6f}\n{result2}')
            axes[i, 1].set_xlabel('Test Statistic')
            axes[i, 1].set_ylabel('Density')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)

        plt.suptitle('D\'Agostino-Pearson Normality Tests - All Transformations',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    @staticmethod
    def _plot_normality_summary(all_results, alpha):
        """Resumen de transformaciones de normalidad"""
        transformations = list(all_results.keys())
        p_values_g1 = [result[0] for result in all_results.values()]
        p_values_g2 = [result[1] for result in all_results.values()]
        success = [result[2] for result in all_results.values()]

        fig = plt.figure(figsize=(12, 6))

        x = np.arange(len(transformations))
        width = 0.35

        bars1 = plt.bar(x - width / 2, p_values_g1, width, label='Group 1', alpha=0.8,
                        color=['green' if p > alpha else 'red' for p in p_values_g1])
        bars2 = plt.bar(x + width / 2, p_values_g2, width, label='Group 2', alpha=0.8,
                        color=['green' if p > alpha else 'red' for p in p_values_g2])

        plt.axhline(y=alpha, color='black', linestyle='--', linewidth=2, label=f'α = {alpha}')
        plt.xlabel('Transformation')
        plt.ylabel('p-value')
        plt.title('Normality Test Results - All Transformations')
        plt.xticks(x, transformations, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


    @staticmethod
    def _plot_effect_sizes(group1, group2, cohens_d, r_rosenthal):
        """Distribuciones superpuestas para Cohen's d y Rosenthal's r"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Cohen's d
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std_pooled = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)

        x = np.linspace(min(mean1, mean2) - 3 * std_pooled, max(mean1, mean2) + 3 * std_pooled, 1000)
        y1 = norm.pdf(x, mean1, std_pooled)
        y2 = norm.pdf(x, mean2, std_pooled)

        ax1.plot(x, y1, 'b-', linewidth=2, label='Group 1')
        ax1.plot(x, y2, 'r-', linewidth=2, label='Group 2')
        ax1.fill_between(x, y1, alpha=0.3, color='blue')
        ax1.fill_between(x, y2, alpha=0.3, color='red')

        overlap = 2 * norm.cdf(-abs(cohens_d) / 2)

        if abs(cohens_d) < 0.2:
            effect_size = "Very small"
        elif abs(cohens_d) < 0.5:
            effect_size = "Small"
        elif abs(cohens_d) < 0.8:
            effect_size = "Medium"
        else:
            effect_size = "Large"

        ax1.set_title(f"Cohen's d = {cohens_d:.3f}\n{effect_size} effect\nOverlap = {overlap * 100:.1f}%")
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Rosenthal's r
        if abs(r_rosenthal) < 1:
            d_equivalent = 2 * r_rosenthal / np.sqrt(1 - r_rosenthal ** 2)
        else:
            d_equivalent = cohens_d

        x = np.linspace(-3, 3, 1000)
        y1 = norm.pdf(x, -d_equivalent / 2, 1)
        y2 = norm.pdf(x, d_equivalent / 2, 1)

        ax2.plot(x, y1, 'b-', linewidth=2, label='Group 1')
        ax2.plot(x, y2, 'r-', linewidth=2, label='Group 2')
        ax2.fill_between(x, y1, alpha=0.3, color='blue')
        ax2.fill_between(x, y2, alpha=0.3, color='red')

        if abs(r_rosenthal) < 0.1:
            r_effect_size = "Very small"
        elif abs(r_rosenthal) < 0.3:
            r_effect_size = "Small"
        elif abs(r_rosenthal) < 0.5:
            r_effect_size = "Medium"
        else:
            r_effect_size = "Large"

        ax2.set_title(f"Rosenthal's r = {r_rosenthal:.3f}\n{r_effect_size} effect\n(Equivalent d = {d_equivalent:.3f})")
        ax2.set_xlabel('Standardized Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Effect Size Visualizations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()



class DataTreatments:
    @staticmethod
    def hypothesis_data_handler(df):
        new_df = df.copy()

        mapping_dict = {
            "Sí": 1,
            "No": 0,
            "9+": 9,
            "No sé": np.nan,
            "Ninguno": 0,
            "0 (No)": 0,
            "Más del 51%": 0.51,
            "No cambió": 0,
            "Cambio positivamente": 1,
            "Cambio negativamente": -1,
            "No se ha recuperado": np.nan,
            "36+": 36,
            'Buena (Pintura reciente, enjarre, vidrios y herrería completa)':3,
            'Mala (Deterioro evidente en varios componentes, grafiti, vandalismo desatendido)':1,
            'Regular (Algunos elementos faltantes)':2,
            'Entre 6 y 10 focos':8,
            'Entre 11 y 15 focos':13,
            'Entre 16 y 20 focos':18,
            '5 o menos focos':3,
            '21 focos o más':21,
            'No realizó modificaciones en este rubro':0,
            'Adaptaciones temporales(reversibles)':1,
            'Adaptaciones permanentes (fijas)':3,
            'Neutra':0,
            'Muy calurosa':2,
            'Fría':-1,
            'Calurosa':1,
            'Muy fría':-2,
            'Pésimo':-2,
            'Regular':0,
            'Bueno':1,
            'Malo':-1,
            'No hay':np.nan,
            'Excelente':2,
            'No recuerdo':np.nan,
            'No los usabamos':np.nan,
            'No hay espacios urbanos y/o áreas verdes en el barrio':np.nan,
            'No sé / no recuerdo':np.nan,
            'Dos veces por semana':2,
            'Una vez a la semana o menos':1,
            'Diario':7,


        }

        for col in new_df.columns:
            new_df[col] = new_df[col].apply(lambda x: mapping_dict.get(x, x))

        return new_df

    @staticmethod
    def corr_data_handler(df):
        new_df = df.copy()

        # Configuración centralizada
        EXCLUDE_COLS = ['Transecto', 'Ponderador']
        HIGH_CARDINALITY_THRESHOLD = 20
        INVALID_VALUES = [9999999, 999999]

        BINNING_RULES = {
            'luz': {'bins': [0, 200, 500, 1000, np.inf], 'labels': ['Bajo', 'Medio', 'Alto', 'Muy alto']},
            'agua': {'bins': [0, 150, 300, 600, np.inf], 'labels': ['Bajo', 'Medio', 'Alto', 'Muy alto']},
            'gas': {'bins': [0, 200, 400, 700, np.inf], 'labels': ['Bajo', 'Medio', 'Alto', 'Muy alto']}
        }

        columns_to_drop = []

        for col in new_df.columns:
            if col in EXCLUDE_COLS:
                continue

            # Limpiar valores inválidos
            new_df[col] = new_df[col].replace(INVALID_VALUES, np.nan)

            unique_values = new_df[col].dropna().unique()
            n_unique = len(unique_values)

            # Aplicar binning si es necesario
            if n_unique > HIGH_CARDINALITY_THRESHOLD:
                rule_applied = False

                for service, rule in BINNING_RULES.items():
                    if service in col.lower() and 'mensual' in col.lower():
                        new_df[col] = pd.cut(new_df[col], bins=rule['bins'],
                                             labels=rule['labels'], include_lowest=True)
                        unique_values = rule['labels']
                        rule_applied = True
                        break

                if not rule_applied:
                    # Binning genérico para otras variables
                    try:
                        new_df[col] = pd.qcut(new_df[col], q=5, labels=False, duplicates='drop')
                        unique_values = new_df[col].dropna().unique()
                    except:
                        pass

            # One-hot encoding (drop first para evitar multicolinealidad)
            for idx, value in enumerate(unique_values):
                new_col_name = f"{col}_{value}"
                new_df[new_col_name] = (new_df[col] == value).astype(int)

                new_col_name = f"{col}_{value}"
                new_df[new_col_name] = (new_df[col] == value).astype(int)

            columns_to_drop.append(col)

        # Limpieza final
        new_df = new_df.drop(columns=columns_to_drop)

        # Validación
        new_df = new_df.loc[:, new_df.nunique() > 1]  # Eliminar columnas sin varianza

        if new_df.shape[1] < 2:
            raise ValueError("Datos insuficientes para análisis de correlación")

        # ⭐ ESTANDARIZACIÓN ⭐
        scaler = StandardScaler()

        # Manejar NaN: StandardScaler no los acepta
        # Opción 1: Imputar con la mediana
        new_df_filled = new_df.fillna(new_df.median())

        # Aplicar estandarización
        scaled_data = scaler.fit_transform(new_df_filled)

        # Convertir de vuelta a DataFrame
        new_df_scaled = pd.DataFrame(
            scaled_data,
            columns=new_df.columns,
            index=new_df.index
        )

        return new_df_scaled

    @staticmethod
    def regression_data_handler(df):
        """
        Prepara datos específicamente para regresión.
        Similar a corr_data_handler pero con drop_first=True para evitar multicolinealidad.
        """
        new_df = df.copy()

        # Configuración centralizada
        EXCLUDE_COLS = ['Transecto', 'Ponderador']
        HIGH_CARDINALITY_THRESHOLD = 20
        INVALID_VALUES = [9999999, 999999]

        BINNING_RULES = {
            'luz': {'bins': [0, 200, 500, 1000, np.inf], 'labels': ['Bajo', 'Medio', 'Alto', 'Muy alto']},
            'agua': {'bins': [0, 150, 300, 600, np.inf], 'labels': ['Bajo', 'Medio', 'Alto', 'Muy alto']},
            'gas': {'bins': [0, 200, 400, 700, np.inf], 'labels': ['Bajo', 'Medio', 'Alto', 'Muy alto']}
        }

        columns_to_drop = []

        # Guardar mapeo de categorías para cada columna (útil para interpretación)
        category_mappings = {}

        for col in new_df.columns:
            if col in EXCLUDE_COLS:
                continue

            # Limpiar valores inválidos
            new_df[col] = new_df[col].replace(INVALID_VALUES, np.nan)

            unique_values = new_df[col].dropna().unique()
            n_unique = len(unique_values)

            # Aplicar binning si es necesario
            if n_unique > HIGH_CARDINALITY_THRESHOLD:
                rule_applied = False

                for service, rule in BINNING_RULES.items():
                    if service in col.lower() and 'mensual' in col.lower():
                        new_df[col] = pd.cut(new_df[col], bins=rule['bins'],
                                             labels=rule['labels'], include_lowest=True)
                        unique_values = rule['labels']
                        rule_applied = True
                        break

                if not rule_applied:
                    # Binning genérico para otras variables
                    try:
                        new_df[col] = pd.qcut(new_df[col], q=5, labels=False, duplicates='drop')
                        unique_values = new_df[col].dropna().unique()
                    except:
                        pass

            # Guardar mapeo de categorías
            category_mappings[col] = list(unique_values)

            # ⭐ ONE-HOT ENCODING con drop_first=True para evitar multicolinealidad
            for idx, value in enumerate(unique_values):
                if idx == 0:  # Saltar primera categoría (referencia)
                    continue

                new_col_name = f"{col}_{value}"
                new_df[new_col_name] = (new_df[col] == value).astype(int)

            columns_to_drop.append(col)

        # Limpieza final
        new_df = new_df.drop(columns=columns_to_drop)

        # Validación
        new_df = new_df.loc[:, new_df.nunique() > 1]  # Eliminar columnas sin varianza

        if new_df.shape[1] < 2:
            raise ValueError("Datos insuficientes para análisis de regresión")

        # ⭐ NO ESTANDARIZAR AQUÍ - Lo haremos después de separar X e Y
        # Porque Y no debe estandarizarse en todos los casos

        return new_df, category_mappings

    @staticmethod
    def except_categories():
        return [
            'La vivienda es… ',  # Agregar espacio
            'La vivienda es … ',
            'De las siguientes opciones, ¿Cuál es el tipo de material de la azotea de esta vivienda? ',  # Agregar espacio
            'De las siguientes opciones, ¿Cuál es el tipo de material de los muros de esta vivienda?',
            'De las siguientes opciones, ¿Cuál es el tipo de material del recubrimiento de la azotea de esta vivienda? ',  # Agregar espacio
            '¿Bajo qué esquema la compró?',
            '¿Bajo qué esquema se la rentan?',
            'En caso de haber hecho modificaciones, ¿Cómo se cubrieron los gastos de esas modificaciones? ',  # Agregar espacio
            '¿De dónde proviene este ruido o contaminación auditiva?',
            'Durante la pandemia por COVID-19, adquirieron los siguientes aparatos: | Computadora de escritorio',
            'Durante la pandemia por COVID-19, adquirieron los siguientes aparatos: | Laptop o computadora portátil',
            'Durante la pandemia por COVID-19, adquirieron los siguientes aparatos: | Tableta electrónica',
            'Durante la pandemia por COVID-19, adquirieron los siguientes aparatos: | Teléfono celular inteligente',
            '¿Cuál fue el impacto en la salud del miembro(s) del hogar contagiados de COVID-19? (Miembro 1)',
            '¿Cuál fue el impacto en la salud del miembro(s) del hogar contagiados de COVID-19? (Miembro 2)',
            '¿Cuál fue el impacto en la salud del miembro(s) del hogar contagiados de COVID-19? (Miembro 3)',
            '¿Cuál fue el impacto en la salud del miembro(s) del hogar contagiados de COVID-19? (Miembro 4)',
            '¿Cuál fue el impacto en la salud del miembro(s) del hogar contagiados de COVID-19? (Miembro 5)',
            '¿Cuál fue el impacto en la salud del miembro(s) del hogar contagiados de COVID-19? (Miembro 6)',
            '¿Cuál fue el impacto en la salud del miembro(s) del hogar contagiados de COVID-19? (Miembro 7)',
            '¿Cuál fue el impacto en la salud del miembro(s) del hogar contagiados de COVID-19? (Miembro 8)',
            '¿Cuál fue el impacto en la salud del miembro(s) del hogar contagiados de COVID-19? (Miembro 9)',
            '¿Cómo se financió el hogar ante la pérdida de empleo? ',
            '¿Cómo se financió la renta o pago de la vivienda? ',
            '¿Si es vivienda vertical, en qué piso se encuentra? ',
            '¿Cuál fue la causa por la que dejo o dejaron de trabajar?',
            '¿Qué miembro de la familia estuvo a cargo del cuidado de o los enfermos/contagiados?',
            'Durante las etapas fuertes del COVID-19, en su trabajo le permitieron:',
            'De las siguientes opciones, ¿Cuál es el tipo de material del recubrimiento de los muros de esta vivienda? ',
            '¿Qué acciones se tomaron para generar ingresos ante la pérdida de trabajo? ',
            '¿A qué distancia de su vivienda se encuentra el parque o espacio público que frecuenta más a menudo? ',
            '¿Cuál es el principal medio de transporte que utiliza la jefa o jefe de familia para ir a su trabajo? ',
            'A partir de la pandemia, ¿Qué iniciativas colectivas o vecinales surgieron? ',
            'Si el o la jefe(a) de familia utiliza el transporte público para ir a trabajar, ¿cuántas cuadras tiene que caminar para llegar a la parada o estación?',
            'Si el o la jefe(a) de familia utiliza el automóvil particular para ir a trabajar, ¿Cuánto tiempo le toma en promedio ir y regresar?',
            '¿Cuál de las siguientes frases describe mejor a la calidad del entorno urbano de su barrio o colonia? (espacio público, banquetas, arbolado, ciclovías)',
            'A partir de la pandemia, ¿Qué iniciativas colectivas o vecinales surgieron? ' ,
            '¿En relación a su experiencia durante la pandemia, Usted actualmente percibe que su barrio o colonia es…? ',
        ]


