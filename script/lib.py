
from scipy.stats import normaltest
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import boxcox
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

class StatHypothesisTest:
    @staticmethod
    def analize_distribution_normality(group1, group2, alpha=0.05):
        print("Normality test search")
        print("=" * 55)

        # Original Data analysis
        print("\n1. Default data analysis:")

        _, p1_orig = normaltest(group1)
        _, p2_orig = normaltest(group2)
        normal1_orig = p1_orig > alpha
        normal2_orig = p2_orig > alpha

        print(f"   Group 1: p={p1_orig:.6f} {'✅' if normal1_orig else '❌'}")
        print(f"   Group 2: p={p2_orig:.6f} {'✅' if normal2_orig else '❌'}")

        if normal1_orig and normal2_orig:
            print("Both groups fulfill normality - Default values 😄")
            return group1, group2, "original", {"p1": p1_orig, "p2": p2_orig}, True

        # Logarithmic transformation Data Analysis
        print("\n2. Log transformed data:")

        try:
            min_val = min(np.min(group1), np.min(group2))
            if min_val <= 0:
                shift = abs(min_val) + 1
                g1_log = np.log(group1 + shift)
                g2_log = np.log(group2 + shift)
                print(f"   Added constant +{shift} to avoid log values <= 0 (Fulfill logarithmic nature) 🔧")
            else:
                g1_log = np.log(group1)
                g2_log = np.log(group2)
                shift = 0

            _, p1_log = normaltest(g1_log)
            _, p2_log = normaltest(g2_log)
            normal1_log = p1_log > alpha
            normal2_log = p2_log > alpha

            print(f"   Group 1: p={p1_log:.6f} {'✅' if normal1_log else '❌'}")
            print(f"   Group 2: p={p2_log:.6f} {'✅' if normal2_log else '❌'}")

            if normal1_log and normal2_log:
                print("Both groups fulfill normality - Logarithmic values 😄")
                return g1_log, g2_log, "log", {"p1": p1_log, "p2": p2_log, "shift": shift}, True

        except Exception as e:
            print(f"Error in logarithmic transformation: {e} ⚠️")
            p1_log, p2_log = np.nan, np.nan

        # 3. Square root transformation analysis
        print("\n3. Square root transformed data:")

        try:
            # Handle negative values
            min_val = min(np.min(group1), np.min(group2))
            if min_val < 0:
                shift = abs(min_val)
                g1_sqrt = np.sqrt(group1 + shift)
                g2_sqrt = np.sqrt(group2 + shift)
                print(f"   Added constant +{shift} (To avoid square root negative values) 🔧")
            else:
                g1_sqrt = np.sqrt(group1)
                g2_sqrt = np.sqrt(group2)
                shift = 0

            _, p1_sqrt = normaltest(g1_sqrt)
            _, p2_sqrt = normaltest(g2_sqrt)
            normal1_sqrt = p1_sqrt > alpha
            normal2_sqrt = p2_sqrt > alpha

            print(f"   Group 1: p={p1_sqrt:.6f} {'✅' if normal1_sqrt else '❌'}")
            print(f"   Group 2: p={p2_sqrt:.6f} {'✅' if normal2_sqrt else '❌'}")

            if normal1_sqrt and normal2_sqrt:
                print("Both groups fulfill normality - Square root values 😄")
                return g1_sqrt, g2_sqrt, "sqrt", {"p1": p1_sqrt, "p2": p2_sqrt, "shift": shift}, True

        except Exception as e:
            print(f"Error in square root transformation: {e} ⚠️")
            p1_sqrt, p2_sqrt = np.nan, np.nan

        # Box-Cox transformation analysis
        print("\n4. Box-Cox transformed data:")

        try:
            # Box-Cox requires values > 0
            min_val = min(np.min(group1), np.min(group2))
            if min_val <= 0:
                shift = abs(min_val) + 0.1
                g1_shifted = group1 + shift
                g2_shifted = group2 + shift
                print(f"   Added constant +{shift} (Fulfill Box-Cox transformation nature) 🔧")
            else:
                g1_shifted = group1
                g2_shifted = group2
                shift = 0

            g1_boxcox, lambda1 = boxcox(g1_shifted)
            g2_boxcox, lambda2 = boxcox(g2_shifted)

            print(f"   Lambda Group 1: {lambda1:.4f} 🔢")
            print(f"   Lambda Group 2: {lambda2:.4f} 🔢")

            _, p1_boxcox = normaltest(g1_boxcox)
            _, p2_boxcox = normaltest(g2_boxcox)
            normal1_boxcox = p1_boxcox > alpha
            normal2_boxcox = p2_boxcox > alpha

            print(f"   Group 1: p={p1_boxcox:.6f} {'✅' if normal1_boxcox else '❌'}")
            print(f"   Group 2: p={p2_boxcox:.6f} {'✅' if normal2_boxcox else '❌'}")

            if normal1_boxcox and normal2_boxcox:
                print("Both groups fulfill normality - Box-Cox values 😄")
                return g1_boxcox, g2_boxcox, "boxcox", {
                    "p1": p1_boxcox, "p2": p2_boxcox,
                    "lambda1": lambda1, "lambda2": lambda2, "shift": shift
                }, True

        except Exception as e:
            print(f"Error in Box-Cox transformation: {e} ⚠️")
            p1_boxcox, p2_boxcox = np.nan, np.nan

        # No transformation is valid
        print(f"\nNo transformation is valid for normality test ❌")
        print(f"Summary:")
        print(f"   Original: G1={'✅' if normal1_orig else '❌'} G2={'✅' if normal2_orig else '❌'}")
        print(f"   Log:      G1={'✅' if not np.isnan(p1_log) and p1_log > alpha else '❌'} G2={'✅' if not np.isnan(p2_log) and p2_log > alpha else '❌'}")
        print(f"   Sqrt:     G1={'✅' if not np.isnan(p1_sqrt) and p1_sqrt > alpha else '❌'} G2={'✅' if not np.isnan(p2_sqrt) and p2_sqrt > alpha else '❌'}")
        print(f"   Box-Cox:  G1={'✅' if not np.isnan(p1_boxcox) and p1_boxcox > alpha else '❌'} G2={'✅' if not np.isnan(p2_boxcox) and p2_boxcox > alpha else '❌'}")
        print(f"\n➡️  Use original data with Mann-Whitney U")

        return group1, group2, "original", {"p1": p1_orig, "p2": p2_orig}, False

    @staticmethod
    def analize_distributions_homogeneity(group1, group2, alpha=0.05):
        print("Homogeneity between variances verification")
        print("=" * 55)

        # Levene Test
        stat_levene, p_levene = levene(group1, group2)
        homogeneous = p_levene > alpha

        print(f"Levene's Test:")
        print(f"   Statistic: {stat_levene:.4f} 📊")
        print(f"   p-value: {p_levene:.6f} 📈")
        print(f"   Result: {'✅ Homogeneous variances' if homogeneous else '❌ Heterogeneous variances'}")

        return stat_levene, p_levene, homogeneous

    @staticmethod
    def execute_statistical_test(group1, group2, normalidad_positiva, homogeneidad_positiva, transformacion, params,
                                 alpha=0.05):

        def inverse_transform(data, transform_type, params, group_num=1):
            if transform_type == "original":
                return data

            elif transform_type == "log":
                shift = params.get("shift", 0)
                return np.exp(data) - shift

            elif transform_type == "sqrt":
                shift = params.get("shift", 0)
                return data ** 2 - shift

            elif transform_type == "boxcox":
                # For Box-Cox, each group can have different lambda
                lambda_key = f"lambda{group_num}"
                lambda_val = params[lambda_key]
                shift = params.get("shift", 0)

                if lambda_val == 0:
                    # Special case: logarithmic transformation
                    original = np.exp(data)
                else:
                    # General case: y = (x^λ - 1) / λ  =>  x = (λ*y + 1)^(1/λ)
                    original = np.power(lambda_val * data + 1, 1 / lambda_val)

                return original - shift

            else:
                raise ValueError(f"Unknown transformation type: {transform_type}")

        def statistical_difference_significance(group1, group2, normalidad_positiva, homogeneidad_positiva):
            if normalidad_positiva == 0 and (
                    homogeneidad_positiva == 0 or homogeneidad_positiva == 1):  # Mann-Whitney U
                stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                is_sig = p_val < alpha
                print(f"Mann-Whitney U Test:")
                print(f"   U-statistic: {stat:.4f} 📊")
                print(f"   p-value: {p_val:.6f} 📈")
                if is_sig == False:
                    print("   The difference is not significant ❌")
                elif is_sig == True:
                    print("   The difference is significant ✅")
                return stat, p_val

            elif normalidad_positiva == 1 and homogeneidad_positiva == 1:  # Independent t-test
                stat, p_val = ttest_ind(group1, group2)
                is_sig = p_val < alpha
                print(f"Independent t-test:")
                print(f"   t-statistic: {stat:.4f} 📊")
                print(f"   p-value: {p_val:.6f} 📈")
                if is_sig == False:
                    print("   The difference is not significant ❌")
                elif is_sig == True:
                    print("   The difference is significant ✅")
                return stat, p_val

            elif normalidad_positiva == 1 and homogeneidad_positiva == 0:  # Welch's t-test
                stat, p_val = ttest_ind(group1, group2, equal_var=False)
                is_sig = p_val < alpha
                print(f"Welch's t-test:")
                print(f"   t-statistic: {stat:.4f} 📊")
                print(f"   p-value: {p_val:.6f} 📈")
                if is_sig == False:
                    print("   The difference is not significant ❌")
                elif is_sig == True:
                    print("   The difference is significant ✅")
                return stat, p_val

        # Execute statistical test
        stat, p_val = statistical_difference_significance(group1, group2, normalidad_positiva, homogeneidad_positiva)

        # Return to original scale
        if transformacion != "original":
            print(f"\n🔄 Reverting '{transformacion}' transformation to calculate metrics in original scale...")
            group1_original = inverse_transform(group1, transformacion, params, group_num=1)
            group2_original = inverse_transform(group2, transformacion, params, group_num=2)
        else:
            group1_original = group1
            group2_original = group2

        def measure_difference(group1, group2):
            """Difference estimator (Hodges-Lehmann)"""

            def hodges_lehmann(group1, group2):
                x = np.asarray(group1)
                y = np.asarray(group2)
                m, n = len(x), len(y)

                # 1) Hodges-Lehmann estimator (median of all differences x_i - y_j)
                pairwise_diffs = np.subtract.outer(x, y).ravel()  # creates m x n matrix and flattens it
                hl = np.median(pairwise_diffs)

                # 2) 95% CI by bootstrap (percentile bootstrap)
                rng = np.random.default_rng(12345)
                n_boot = 5000
                boots = np.empty(n_boot)
                for i in range(n_boot):
                    bx = rng.choice(x, size=m, replace=True)
                    by = rng.choice(y, size=n, replace=True)
                    boots[i] = np.median(np.subtract.outer(bx, by).ravel())

                alpha = 0.05
                ci_lower, ci_upper = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])

                print(f"📊 Hodges–Lehmann (original scale) = {hl:.4f}")
                print(f"📊 Bootstrap 95% CI for HL: [{ci_lower:.4f}, {ci_upper:.4f}]")
                return hl, (ci_lower, ci_upper)

            return hodges_lehmann(group1, group2)

        def measure_effect(group1, group2, stat):
            """Effect size (Cohen's d, Rosenthal's r)"""
            # Rosenthal's r
            n1, n2 = len(group1), len(group2)
            df = n1 + n2 - 2  # approximate degrees of freedom
            r_rosenthal = np.sqrt(stat ** 2 / (stat ** 2 + df))

            # Cohen's d
            mean_diff = np.mean(group1) - np.mean(group2)
            s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
            cohens_d = mean_diff / pooled_std

            print(f"📊 Cohen's d (original scale) = {cohens_d:.4f}")
            print(f"📊 Rosenthal's r = {r_rosenthal:.4f}")

            return cohens_d, r_rosenthal

        # Calculate metrics in original scale
        hl, hl_ci = measure_difference(group1_original, group2_original)
        cohens_d, r_rosenthal = measure_effect(group1_original, group2_original, stat)

        # Return results
        return {
            'stat': stat,
            'p_value': p_val,
            'hodges_lehmann': hl,
            'hl_confidence_interval': hl_ci,
            'cohens_d': cohens_d,
            'rosenthal_r': r_rosenthal
        }

class DataTreatments:
    @staticmethod
    def categorical_handler(df):
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

class CorrSystem:
    @staticmethod
    def do(df):

        df_num = df.select_dtypes(include=["int64", "float64"])

        col_selector = pn.widgets.MultiSelect(
            name='Columnas',
            options=df_num.columns.tolist(),
            size=10
        )

        @pn.depends(col_selector)
        def plot_corr(selected):
            if len(selected) < 2:
                return "⚠️ Selecciona al menos 2 columnas."
            corr = df_num[selected].corr()
            fig = px.imshow(corr, text_auto=True)
            fig.update_xaxes(showticklabels=True, title_text=None)
            fig.update_yaxes(showticklabels=True, title_text=None)
            return fig

        return pn.Column(
            "# 🔎 Sistema de Correlación",
            col_selector,
            plot_corr
        )