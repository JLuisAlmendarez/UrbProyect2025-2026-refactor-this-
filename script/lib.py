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

class CorrSystem:
    @staticmethod
    def do(df):
        # Solo variables numéricas
        df_num = df.select_dtypes(include=["int64", "float64"])

        # Selector de columnas en Streamlit
        selected = st.multiselect(
            "Selecciona columnas numéricas",
            options=df_num.columns.tolist(),
            default=df_num.columns.tolist()[:2]  # Preselecciona las primeras 2
        )

        # Mostrar correlación
        if len(selected) < 2:
            st.warning("⚠️ Selecciona al menos 2 columnas.")
        else:
            corr = df_num[selected].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Plasma")
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)


class StatHypothesisTest:
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
            if plot:
                # Show plots for all transformations attempted
                StatHypothesisTest._plot_all_normality_tests(group1, group2, alpha)
                # Show summary
                all_results = {
                    'Original': (p1_orig, p2_orig, True)
                }
                StatHypothesisTest._plot_normality_summary(all_results, alpha)
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
                if plot:
                    # Show plots for all transformations attempted so far
                    StatHypothesisTest._plot_all_normality_tests(group1, group2, alpha)
                    # Show summary
                    all_results = {
                        'Original': (p1_orig, p2_orig, normal1_orig and normal2_orig),
                        'Log': (p1_log, p2_log, True)
                    }
                    StatHypothesisTest._plot_normality_summary(all_results, alpha)
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
                if plot:
                    # Show plots for all transformations attempted so far
                    StatHypothesisTest._plot_all_normality_tests(group1, group2, alpha)
                    # Show summary
                    all_results = {
                        'Original': (p1_orig, p2_orig, normal1_orig and normal2_orig),
                        'Log': (p1_log if not np.isnan(p1_log) else 0, p2_log if not np.isnan(p2_log) else 0,
                                not np.isnan(p1_log) and not np.isnan(p2_log) and p1_log > alpha and p2_log > alpha),
                        'Sqrt': (p1_sqrt, p2_sqrt, True)
                    }
                    StatHypothesisTest._plot_normality_summary(all_results, alpha)
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
                if plot:
                    # Show plots for all transformations attempted so far
                    StatHypothesisTest._plot_all_normality_tests(group1, group2, alpha)
                    # Show summary
                    all_results = {
                        'Original': (p1_orig, p2_orig, normal1_orig and normal2_orig),
                        'Log': (p1_log if not np.isnan(p1_log) else 0, p2_log if not np.isnan(p2_log) else 0,
                                not np.isnan(p1_log) and not np.isnan(p2_log) and p1_log > alpha and p2_log > alpha),
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
            print(f"Error in Box-Cox transformation: {e} ⚠️")
            p1_boxcox, p2_boxcox = np.nan, np.nan

        # No transformation is valid
        print(f"\nNo transformation is valid for normality test ❌")
        print(f"Summary:")
        print(f"   Original: G1={'✅' if normal1_orig else '❌'} G2={'✅' if normal2_orig else '❌'}")
        print(
            f"   Log:      G1={'✅' if not np.isnan(p1_log) and p1_log > alpha else '❌'} G2={'✅' if not np.isnan(p2_log) and p2_log > alpha else '❌'}")
        print(
            f"   Sqrt:     G1={'✅' if not np.isnan(p1_sqrt) and p1_sqrt > alpha else '❌'} G2={'✅' if not np.isnan(p2_sqrt) and p2_sqrt > alpha else '❌'}")
        print(
            f"   Box-Cox:  G1={'✅' if not np.isnan(p1_boxcox) and p1_boxcox > alpha else '❌'} G2={'✅' if not np.isnan(p2_boxcox) and p2_boxcox > alpha else '❌'}")
        print(f"\n➡️  Use original data with Mann-Whitney U")

        # Show summary of all transformations tried
        if plot:
            # Show plots for all transformations attempted
            StatHypothesisTest._plot_all_normality_tests(group1, group2, alpha)
            # Show final summary
            all_results = {
                'Original': (p1_orig, p2_orig, normal1_orig and normal2_orig),
                'Log': (p1_log if not np.isnan(p1_log) else 0, p2_log if not np.isnan(p2_log) else 0,
                        not np.isnan(p1_log) and not np.isnan(p2_log) and p1_log > alpha and p2_log > alpha),
                'Sqrt': (p1_sqrt if not np.isnan(p1_sqrt) else 0, p2_sqrt if not np.isnan(p2_sqrt) else 0,
                         not np.isnan(p1_sqrt) and not np.isnan(p2_sqrt) and p1_sqrt > alpha and p2_sqrt > alpha),
                'Box-Cox': (p1_boxcox if not np.isnan(p1_boxcox) else 0, p2_boxcox if not np.isnan(p2_boxcox) else 0,
                            not np.isnan(p1_boxcox) and not np.isnan(
                                p2_boxcox) and p1_boxcox > alpha and p2_boxcox > alpha)
            }
            StatHypothesisTest._plot_normality_summary(all_results, alpha)

        return group1, group2, "original", {"p1": p1_orig, "p2": p2_orig}, False

    @staticmethod
    def analize_distributions_homogeneity(group1, group2, alpha=0.05, plot=True):
        print("Homogeneity between variances verification")
        print("=" * 55)

        # Levene Test
        stat_levene, p_levene = levene(group1, group2)
        homogeneous = p_levene > alpha

        print(f"Levene's Test:")
        print(f"   Statistic: {stat_levene:.4f} 📊")
        print(f"   p-value: {p_levene:.6f} 📈")
        print(f"   Result: {'✅ Homogeneous variances' if homogeneous else '❌ Heterogeneous variances'}")

        # Plot variance comparison
        if plot:
            StatHypothesisTest._plot_variance_homogeneity(group1, group2, stat_levene, p_levene, homogeneous)

        return stat_levene, p_levene, homogeneous

    @staticmethod
    def _plot_variance_homogeneity(group1, group2, stat_levene, p_levene, homogeneous):
        """Boxplot comparison for variance homogeneity"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create boxplots
        bp = plt.boxplot([group1, group2], labels=['Group 1', 'Group 2'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)

        # Add variance annotations
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        max1, max2 = np.max(group1), np.max(group2)

        plt.text(1, max1 + (max1 - np.min(group1)) * 0.05, f'Var: {var1:.3f}',
                 ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.text(2, max2 + (max2 - np.min(group2)) * 0.05, f'Var: {var2:.3f}',
                 ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

        # Levene test result
        result_text = f"Levene's Test\nStatistic: {stat_levene:.4f}\np-value: {p_levene:.6f}\n{'✅ Homogeneous' if homogeneous else '❌ Heterogeneous'}"
        plt.text(1.5, plt.ylim()[1] * 0.9, result_text, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

        plt.title('Variance Homogeneity Test', fontsize=14, fontweight='bold')
        plt.ylabel('Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def execute_statistical_test(group1, group2, normalidad_positiva, homogeneidad_positiva, transformacion, params,
                                 alpha=0.05, plot=True):

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

        # Plot p-value distribution
        if plot:
            test_type = StatHypothesisTest._determine_test_type(normalidad_positiva, homogeneidad_positiva)
            StatHypothesisTest._plot_pvalue_distribution(stat, p_val, test_type, alpha, len(group1), len(group2))

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

        # Plot effect sizes and Hodges-Lehmann
        if plot:
            StatHypothesisTest._plot_effect_sizes(group1_original, group2_original, cohens_d, r_rosenthal)
            StatHypothesisTest._plot_hodges_lehmann_bootstrap(group1_original, group2_original, hl, hl_ci)

        # Return results
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
        """Determine which test was used"""
        if normalidad_positiva == 0:
            return "Mann-Whitney U"
        elif normalidad_positiva == 1 and homogeneidad_positiva == 1:
            return "Independent t-test"
        elif normalidad_positiva == 1 and homogeneidad_positiva == 0:
            return "Welch's t-test"

    @staticmethod
    def _plot_pvalue_distribution(stat, p_val, test_type, alpha, n1, n2):
        """Bell curve with p-value for each test"""
        plt.figure(figsize=(12, 6))

        if test_type == "Mann-Whitney U":
            # Para Mann-Whitney, usar distribución normal aproximada
            x = np.linspace(-4, 4, 1000)
            y = norm.pdf(x, 0, 1)

            # Convertir U a z-score aproximado
            mean_u = n1 * n2 / 2
            std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z_score = (stat - mean_u) / std_u

            plt.plot(x, y, 'b-', linewidth=2, label='Standard Normal')
            plt.axvline(z_score, color='red', linewidth=3, label=f'Observed z: {z_score:.3f}')

            # Critical regions
            z_crit = norm.ppf(1 - alpha / 2)
            plt.axvline(-z_crit, color='red', linestyle='--', alpha=0.7)
            plt.axvline(z_crit, color='red', linestyle='--', alpha=0.7)

            # Shade rejection regions
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

            # Critical regions
            t_crit = t.ppf(1 - alpha / 2, df)
            plt.axvline(-t_crit, color='red', linestyle='--', alpha=0.7)
            plt.axvline(t_crit, color='red', linestyle='--', alpha=0.7)

            # Shade rejection regions
            x_left = x[x <= -t_crit]
            x_right = x[x >= t_crit]
            plt.fill_between(x_left, t.pdf(x_left, df), alpha=0.3, color='red')
            plt.fill_between(x_right, t.pdf(x_right, df), alpha=0.3, color='red')

        plt.title(f'{test_type}\np-value = {p_val:.5f}', fontsize=14, fontweight='bold')
        plt.xlabel('Test Statistic')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)

    @staticmethod
    def _plot_all_normality_tests(group1, group2, alpha):
        """Plot detailed normality tests for all transformations attempted"""

        # Prepare all transformations
        transformations = {}

        # Original data
        stat1_orig, p1_orig = normaltest(group1)
        stat2_orig, p2_orig = normaltest(group2)
        transformations['Original'] = {
            'data1': group1, 'data2': group2,
            'stat1': stat1_orig, 'stat2': stat2_orig,
            'p1': p1_orig, 'p2': p2_orig
        }

        # Log transformation
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

        # Sqrt transformation
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

        # Box-Cox transformation
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

        # Create plots for each transformation
        valid_transforms = {k: v for k, v in transformations.items() if v is not None}
        n_transforms = len(valid_transforms)

        fig, axes = plt.subplots(n_transforms, 2, figsize=(15, 5 * n_transforms))
        if n_transforms == 1:
            axes = axes.reshape(1, -1)

        # Chi-square distribution
        x = np.linspace(0, 15, 1000)
        y = chi2.pdf(x, df=2)
        chi2_crit = chi2.ppf(1 - alpha, df=2)
        x_reject = x[x >= chi2_crit]

        for i, (name, data) in enumerate(valid_transforms.items()):
            # Group 1 plot
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

            # Group 2 plot
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

    @staticmethod
    def _plot_normality_summary(all_results, alpha):
        """Summary plot of normality transformations (simplified without success panel)"""
        transformations = list(all_results.keys())
        p_values_g1 = [result[0] for result in all_results.values()]
        p_values_g2 = [result[1] for result in all_results.values()]
        success = [result[2] for result in all_results.values()]

        plt.figure(figsize=(12, 6))

        # Bar plot of p-values
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

    @staticmethod
    def _plot_hodges_lehmann_bootstrap(group1, group2, hl_estimate, hl_ci):
        """Enhanced Hodges-Lehmann visualization with pairwise differences and dense bootstrap"""

        # Convert to numpy arrays if they're pandas objects
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        # Calculate all pairwise differences
        pairwise_diffs = np.subtract.outer(group1, group2).ravel()

        # Generate dense bootstrap distribution
        rng = np.random.default_rng(12345)
        n_boot = 10000  # Increased iterations for better visualization
        m, n = len(group1), len(group2)
        boots = np.empty(n_boot)

        for i in range(n_boot):
            bx = rng.choice(group1, size=m, replace=True)
            by = rng.choice(group2, size=n, replace=True)
            boots[i] = np.median(np.subtract.outer(bx, by).ravel())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Panel 1: Distribution of all pairwise differences with Hodges-Lehmann marked
        ax1.hist(pairwise_diffs, bins=50, density=True, alpha=0.7, color='lightblue',
                 edgecolor='black', label=f'All pairwise differences (n={len(pairwise_diffs)})')
        ax1.axvline(hl_estimate, color='red', linewidth=3,
                    label=f'Hodges-Lehmann (median): {hl_estimate:.3f}')
        ax1.axvline(np.mean(pairwise_diffs), color='orange', linewidth=2, linestyle='--',
                    label=f'Mean difference: {np.mean(pairwise_diffs):.3f}')

        # Add normal distribution overlay for comparison
        diff_mean, diff_std = np.mean(pairwise_diffs), np.std(pairwise_diffs)
        x_norm = np.linspace(np.min(pairwise_diffs), np.max(pairwise_diffs), 1000)
        y_norm = norm.pdf(x_norm, diff_mean, diff_std)
        ax1.plot(x_norm, y_norm, 'k--', alpha=0.5, label='Normal approximation')

        ax1.set_xlabel('Pairwise Differences (Group1 - Group2)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of All Pairwise Differences\n(xi - yj for all i,j pairs)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Dense bootstrap distribution with comparison to theoretical
        ax2.hist(boots, bins=80, density=True, alpha=0.7, color='lightgreen',
                 edgecolor='black', label=f'Bootstrap distribution (n={n_boot})')
        ax2.axvline(hl_estimate, color='red', linewidth=3,
                    label=f'Original H-L: {hl_estimate:.3f}')
        ax2.axvline(hl_ci[0], color='orange', linestyle='--', linewidth=2,
                    label=f'95% CI: [{hl_ci[0]:.3f}, {hl_ci[1]:.3f}]')
        ax2.axvline(hl_ci[1], color='orange', linestyle='--', linewidth=2)

        # Fill CI area
        y_max = ax2.get_ylim()[1]
        ax2.fill_betweenx([0, y_max], hl_ci[0], hl_ci[1], alpha=0.2, color='orange')

        # Add theoretical normal approximation of bootstrap
        boot_mean, boot_std = np.mean(boots), np.std(boots)
        x_boot_norm = np.linspace(np.min(boots), np.max(boots), 1000)
        y_boot_norm = norm.pdf(x_boot_norm, boot_mean, boot_std)
        ax2.plot(x_boot_norm, y_boot_norm, 'k--', alpha=0.8, linewidth=2,
                 label='Normal approximation')

        ax2.set_xlabel('Hodges-Lehmann Estimator')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Bootstrap Distribution of H-L Estimator\n(Enhanced with {n_boot:,} iterations)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add interpretation box
        if hl_ci[0] > 0:
            interpretation = "Significant positive difference"
            interpretation_detail = f"Group 1 > Group 2 (p < 0.05)"
            color = 'green'
        elif hl_ci[1] < 0:
            interpretation = "Significant negative difference"
            interpretation_detail = f"Group 1 < Group 2 (p < 0.05)"
            color = 'green'
        else:
            interpretation = "No significant difference"
            interpretation_detail = f"CI includes 0 (p ≥ 0.05)"
            color = 'red'

        # Add interpretation text box
        textstr = f'{interpretation}\n{interpretation_detail}\nCI width: {hl_ci[1] - hl_ci[0]:.3f}'
        props = dict(boxstyle='round', facecolor=color, alpha=0.1, edgecolor=color)
        ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10, fontweight='bold',
                 verticalalignment='top', horizontalalignment='right', bbox=props, color=color)

        plt.suptitle('Hodges-Lehmann Estimator Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def _plot_effect_sizes(group1, group2, cohens_d, r_rosenthal):
        """Overlapping distributions for Cohen's d and Rosenthal's r"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Cohen's d visualization
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std_pooled = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)

        x = np.linspace(min(mean1, mean2) - 3 * std_pooled, max(mean1, mean2) + 3 * std_pooled, 1000)
        y1 = norm.pdf(x, mean1, std_pooled)
        y2 = norm.pdf(x, mean2, std_pooled)

        ax1.plot(x, y1, 'b-', linewidth=2, label='Group 1')
        ax1.plot(x, y2, 'r-', linewidth=2, label='Group 2')
        ax1.fill_between(x, y1, alpha=0.3, color='blue')
        ax1.fill_between(x, y2, alpha=0.3, color='red')

        # Calculate overlap
        overlap = 2 * norm.cdf(-abs(cohens_d) / 2)

        # Effect size interpretation
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

        # Rosenthal's r visualization
        # Convert r to Cohen's d equivalent for visualization: d = 2r/sqrt(1-r²)
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

        # r effect size interpretation
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

    def corr_data_handler(df):
        return None

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


