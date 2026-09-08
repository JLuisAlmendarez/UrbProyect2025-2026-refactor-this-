"""
Herramientas estadísticas y de visualización para Reporte.ipynb.

Principios del proyecto:
- Las variables y recodificaciones específicas de cada pregunta permanecen en el notebook.
- La librería encapsula mecánica repetitiva: pruebas, tamaños de efecto,
  Random Forest, tablas y gráficas reutilizables.
- Las pruebas estadísticas se ejecutan sobre las observaciones válidas de la muestra.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm

try:
    import dcor
except ImportError:
    dcor = None

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression


ALPHA = 0.05

__all__ = [
    "preparar_encuesta",
    "evaluar_supuestos_parametricos",
    "normalidad_y_levene_por_grupos",
    "pruebas_no_parametricas",
    "spearman",
    "kruskal_wallis",
    "chi_cuadrada",
    "mann_whitney_pares",
    "wilcoxon_pareado",
    "ajuste_gamma_agrupada",
    "regresion_cuantil",
    "resumen_grupos",
    "grafica_barras",
    "grafica_conteos",
    "grafica_resumen_grupos",
    "grafica_dispersion",
    "grafica_boxplot_por_categoria",
    "random_forest_importance",
    "screening_masivo",
]

# Exclusiones específicas de esta encuesta para evitar data leakage en RF.
# Son variables cuyo patrón de respuesta depende directamente del objetivo.
EXCLUSIONES_LEAKAGE_DEFAULT = {
    "La vivienda es… ": {
        "¿Si es vivienda vertical, en qué piso se encuentra? ",
    },
    "La vivienda es … ": {
        "¿Bajo qué esquema la compró?",
        "¿Bajo qué esquema se la rentan?",
        "¿El pago de la renta de la vivienda ha sufrido incrementos durante la pandemia?",
    },
}


# ---------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------

def _normalizar_texto(x):
    if pd.isna(x):
        return None
    texto = (
        unicodedata.normalize("NFKD", str(x))
        .encode("ascii", "ignore")
        .decode("utf-8")
        .strip()
        .upper()
    )
    return re.sub(r"\s+", " ", texto)


def _razon_missing_original(x) -> str:
    if pd.isna(x):
        return "__MISSING__"
    t = _normalizar_texto(x)
    if t in {"NO SE", "NO SABE", "NS", "N/S"}:
        return "__NO_SE__"
    if t in {"NO RECUERDO", "NO SE / NO RECUERDO", "NO SE/NO RECUERDO"}:
        return "__NO_RECUERDA__"
    if t in {"NO APLICA", "NO APLICA/ES VIVIENDA HORIZONTAL"}:
        return "__NO_APLICA__"
    return "__RESPONDIDO__"


def resumen_grupos(
    data: pd.DataFrame,
    grupo: str,
    valor: str,
    orden: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Media, mediana y n observados por grupo."""
    tmp = data[[grupo, valor]].dropna(subset=[grupo, valor]).copy()
    out = tmp.groupby(grupo)[valor].agg(n="count", media="mean", mediana="median")
    if orden is not None and len(out):
        out = out.reindex(orden)
    return out


# ---------------------------------------------------------------------
# Gráficas reutilizables
# ---------------------------------------------------------------------

def grafica_barras(
    serie: pd.Series,
    titulo: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize=(9, 5),
    rotacion=0,
    formato_etiqueta=None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    serie = pd.Series(serie)
    barras = ax.bar(serie.index.astype(str), serie.values)
    if formato_etiqueta is not None:
        etiquetas = [formato_etiqueta(v) for v in serie.values]
        ax.bar_label(barras, labels=etiquetas)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotacion)
    plt.tight_layout()
    return ax


def grafica_conteos(
    valores,
    titulo="",
    xlabel="",
    ylabel=None,
    normalize=False,
    figsize=(9, 5),
    rotacion=0,
    ax=None,
):
    serie = pd.Series(valores).value_counts(normalize=normalize)
    if ylabel is None:
        ylabel = "Proporción" if normalize else "Cantidad"
    formato = (lambda v: f"{v:.1%}") if normalize else (lambda v: f"{v:,.0f}")
    return grafica_barras(
        serie, titulo=titulo, xlabel=xlabel, ylabel=ylabel,
        figsize=figsize, rotacion=rotacion, formato_etiqueta=formato, ax=ax
    )


def grafica_resumen_grupos(
    data: pd.DataFrame,
    grupo: str,
    valor: str,
    estadistico: str = "mediana",
    orden: Optional[Sequence] = None,
    titulo: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize=(9, 5),
    rotacion=0,
    ax=None,
):
    r = resumen_grupos(data, grupo, valor, orden=orden)
    if r.empty:
        return ax
    return grafica_barras(
        r[estadistico],
        titulo=titulo,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        rotacion=rotacion,
        formato_etiqueta=lambda v: f"{v:.2f}",
        ax=ax,
    )


def grafica_dispersion(data, x, y, titulo="", xlabel=None, ylabel=None, figsize=(8, 6), ax=None):
    tmp = data[[x, y]].dropna()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.scatter(tmp[x], tmp[y], alpha=0.5)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    plt.tight_layout()
    return ax


def grafica_boxplot_por_categoria(
    data, categoria, valor, orden=None, etiquetas=None, titulo="", xlabel="", ylabel="",
    limite_iqr_visual=True, figsize=(8, 6), ax=None
):
    """Boxplot muestral por categoría."""
    tmp = data[[categoria, valor]].dropna(subset=[categoria, valor]).copy()
    if orden is None:
        orden = list(tmp[categoria].dropna().unique())
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    grupos = [
        pd.to_numeric(tmp.loc[tmp[categoria] == cat, valor], errors="coerce")
        .dropna().to_numpy()
        for cat in orden
    ]
    ax.boxplot(grupos, patch_artist=True)

    if limite_iqr_visual and len(tmp):
        y_num = pd.to_numeric(tmp[valor], errors="coerce").dropna()
        if len(y_num):
            q1, q3 = y_num.quantile([0.25, 0.75])
            lim = q3 + 1.5 * (q3 - q1)
            if np.isfinite(lim) and lim > 0:
                ax.set_ylim(0, lim * 1.05)

    ax.set_xticks(range(1, len(orden) + 1))
    ax.set_xticklabels(etiquetas if etiquetas is not None else [str(v) for v in orden])
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------
# Supuestos y pruebas ad hoc
# ---------------------------------------------------------------------

def evaluar_supuestos_parametricos(
    data: pd.DataFrame,
    x: str,
    y: str,
    alpha: float = ALPHA,
) -> dict:
    tmp = data[[x, y]].copy()
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    tmp = tmp.dropna()
    if len(tmp) < 3 or tmp[x].nunique() < 2 or tmp[y].nunique() < 2:
        return {
            "n": len(tmp), "normalidad": None, "homocedasticidad": None,
            "todos_cumplen": False, "estado": "No evaluable"
        }

    X = sm.add_constant(tmp[x])
    modelo = sm.OLS(tmp[y], X).fit()
    residuos = modelo.resid

    if np.ptp(np.asarray(residuos, dtype=float)) == 0:
        sh_stat = sh_p = np.nan
    else:
        sh_stat, sh_p = stats.shapiro(residuos)

    try:
        bp_stat, bp_p, bp_f, bp_fp = sm.stats.diagnostic.het_breuschpagan(
            residuos, modelo.model.exog
        )
    except Exception:
        bp_stat = bp_p = bp_f = bp_fp = np.nan

    return {
        "n": len(tmp),
        "normalidad": {
            "estadistico": float(sh_stat) if np.isfinite(sh_stat) else np.nan,
            "p_valor": float(sh_p) if np.isfinite(sh_p) else np.nan,
            "cumple": bool(sh_p > alpha) if np.isfinite(sh_p) else None,
        },
        "homocedasticidad": {
            "estadistico": float(bp_stat), "p_valor": float(bp_p),
            "estadistico_f": float(bp_f), "p_valor_f": float(bp_fp),
            "cumple": bool(bp_p > alpha) if np.isfinite(bp_p) else False
        },
        "todos_cumplen": bool(
            np.isfinite(sh_p) and sh_p > alpha
            and (bp_p > alpha if np.isfinite(bp_p) else False)
        ),
        "estado": "Válida",
        "modelo": modelo,
    }


def normalidad_y_levene_por_grupos(data, grupo, valor, alpha=ALPHA) -> dict:
    tmp = data[[grupo, valor]].dropna()
    normalidad = []
    grupos = []
    for cat, d in tmp.groupby(grupo):
        arr = pd.to_numeric(d[valor], errors="coerce").dropna().to_numpy()
        grupos.append(arr)
        if len(arr) < 3:
            normalidad.append({
                "grupo": cat, "n": len(arr), "W": np.nan, "p_valor": np.nan,
                "normal": None, "advertencia": "Menos de 3 observaciones"
            })
        elif np.ptp(arr) == 0:
            normalidad.append({
                "grupo": cat, "n": len(arr), "W": np.nan, "p_valor": np.nan,
                "normal": None, "advertencia": "Grupo constante"
            })
        else:
            stat, p = stats.shapiro(arr)
            normalidad.append({
                "grupo": cat, "n": len(arr), "W": stat, "p_valor": p,
                "normal": bool(p >= alpha), "advertencia": None
            })

    grupos_levene = [g for g in grupos if len(g) > 0]
    if len(grupos_levene) >= 2 and any(np.ptp(g) > 0 for g in grupos_levene):
        lev_stat, lev_p = stats.levene(*grupos_levene)
    else:
        lev_stat = lev_p = np.nan
    return {
        "normalidad": pd.DataFrame(normalidad),
        "levene": {
            "estadistico": lev_stat,
            "p_valor": lev_p,
            "homogeneidad": bool(lev_p >= alpha) if np.isfinite(lev_p) else None,
        },
    }


def _magnitud_spearman(v):
    a = abs(v)
    if a < 0.10: return "Trivial"
    if a < 0.30: return "Débil"
    if a < 0.50: return "Moderado"
    return "Fuerte"


def _magnitud_epsilon(v):
    if not np.isfinite(v): return "No evaluable"
    if v < 0.01: return "Trivial"
    if v < 0.06: return "Pequeño"
    if v < 0.14: return "Moderado"
    return "Grande"


def _magnitud_cramer(v):
    return _magnitud_spearman(v)


def _base_resultado(prueba, n):
    return {
        "prueba": prueba, "n": int(n), "estadistico": np.nan, "p_valor": np.nan,
        "effect_size": np.nan, "magnitud": "No evaluable", "estado": "No evaluable",
        "advertencia": None, "significativo": "No evaluable",
    }


def _finalizar_significancia(res, alpha=ALPHA):
    if res["estado"] == "No evaluable":
        res["significativo"] = "No evaluable"
    elif np.isfinite(res["p_valor"]):
        res["significativo"] = "Sí" if res["p_valor"] < alpha else "No"
    else:
        res["significativo"] = "No evaluable"
    return res


def spearman(data, x, y, alpha=ALPHA) -> dict:
    tmp = data[[x, y]].copy()
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    tmp = tmp.dropna()
    n = len(tmp)
    res = _base_resultado("Spearman", n)
    if n < 2 or tmp[x].nunique() < 2 or tmp[y].nunique() < 2:
        res["advertencia"] = "Variable constante o insuficientes pares válidos"
        return res

    rho, p = stats.spearmanr(tmp[x], tmp[y])
    res.update(
        estadistico=float(rho), p_valor=float(p), effect_size=float(rho),
        magnitud=_magnitud_spearman(rho), estado="Válida"
    )
    if n < 10:
        res["estado"] = "Válida con advertencia"
        res["advertencia"] = f"Spearman con n pequeño (n={n})"
    return _finalizar_significancia(res, alpha)


def kruskal_wallis(data, grupo, valor, alpha=ALPHA) -> dict:
    tmp = data[[grupo, valor]].copy()
    tmp[valor] = pd.to_numeric(tmp[valor], errors="coerce")
    tmp = tmp.dropna()
    n = len(tmp)
    res = _base_resultado("Kruskal-Wallis", n)
    if n == 0 or tmp[grupo].nunique() < 2 or tmp[valor].nunique() < 2:
        res["advertencia"] = "Variable constante o menos de dos grupos válidos"
        return res

    counts = tmp.groupby(grupo).size()
    grupo_min = int(counts.min())
    grupos = [d[valor].to_numpy() for _, d in tmp.groupby(grupo)]
    H, p = stats.kruskal(*grupos)
    eps = max(0.0, float(H) / (n - 1)) if n > 1 else np.nan
    res.update(
        estadistico=float(H), p_valor=float(p), effect_size=eps,
        magnitud=_magnitud_epsilon(eps), estado="Válida"
    )
    if grupo_min < 5:
        res["estado"] = "Válida con advertencia"
        res["advertencia"] = f"grupo menor n={grupo_min}"
    res["grupo_min_n"] = grupo_min
    return _finalizar_significancia(res, alpha)


def chi_cuadrada(data, x, y, alpha=ALPHA) -> dict:
    tmp = data[[x, y]].dropna().copy()
    n = len(tmp)
    res = _base_resultado("Chi-cuadrada", n)
    if n == 0 or tmp[x].nunique() < 2 or tmp[y].nunique() < 2:
        res["advertencia"] = "Variable constante o tabla sin variación suficiente"
        return res

    tabla = pd.crosstab(tmp[x], tmp[y])
    chi2, p, dof, esperado = stats.chi2_contingency(tabla, correction=False)
    denom = n * max(1, min(tabla.shape[0] - 1, tabla.shape[1] - 1))
    v = math.sqrt(max(0.0, chi2) / denom) if denom > 0 else np.nan
    pct_menor5 = float((esperado < 5).mean() * 100)
    min_esp = float(np.min(esperado))

    res.update(
        estadistico=float(chi2), p_valor=float(p), effect_size=float(v),
        magnitud=_magnitud_cramer(v), estado="Válida"
    )
    if pct_menor5 > 0:
        res["estado"] = "No evaluable"
        res["advertencia"] = (
            f"Chi² no fiable: {pct_menor5:.1f}% de celdas esperadas < 5; "
            f"mínimo esperado={min_esp:.2f}"
        )
    res["esperadas_pct_menor_5"] = pct_menor5
    res["esperada_min"] = min_esp
    res["gl"] = int(dof)
    return _finalizar_significancia(res, alpha)


def pruebas_no_parametricas(data, x, y, col_categorica=None) -> dict:
    """Spearman, métricas exploratorias y, opcionalmente, Kruskal-Wallis."""
    resultados = {"spearman": spearman(data, x, y)}
    tmp = data[[x, y]].dropna().copy()
    try:
        xv = pd.to_numeric(tmp[x], errors="coerce")
        yv = pd.to_numeric(tmp[y], errors="coerce")
        m = xv.notna() & yv.notna()
        if dcor is not None and m.sum() > 1:
            resultados["distance_correlation"] = float(
                dcor.distance_correlation(
                    xv[m].to_numpy(dtype=float),
                    yv[m].to_numpy(dtype=float)
                )
            )
        if m.sum() > 3:
            resultados["mutual_information"] = float(
                mutual_info_regression(xv[m].to_frame(), yv[m], random_state=42)[0]
            )
    except Exception:
        pass
    if col_categorica is not None:
        resultados["kruskal_wallis"] = kruskal_wallis(data, col_categorica, y)
    return resultados


def mann_whitney_pares(data, grupo, valor) -> pd.DataFrame:
    tmp = data[[grupo, valor]].dropna()
    cats = list(tmp[grupo].unique())
    rows = []
    from itertools import combinations
    for a, b in combinations(cats, 2):
        g1 = tmp.loc[tmp[grupo] == a, valor]
        g2 = tmp.loc[tmp[grupo] == b, valor]
        if len(g1) and len(g2):
            u, p = stats.mannwhitneyu(g1, g2)
            rows.append({"grupo_1": a, "grupo_2": b, "U": u, "p_valor": p})
    return pd.DataFrame(rows)


def wilcoxon_pareado(data, antes, despues, alpha=ALPHA) -> dict:
    tmp = data[[antes, despues]].dropna()
    if len(tmp) == 0:
        return {
            "prueba": "Wilcoxon", "n": 0, "estadistico": np.nan, "p_valor": np.nan,
            "significativo": "No evaluable", "estado": "No evaluable"
        }
    stat, p = stats.wilcoxon(tmp[antes], tmp[despues])
    return {
        "prueba": "Wilcoxon", "n": len(tmp), "estadistico": stat, "p_valor": p,
        "significativo": "Sí" if p < alpha else "No", "estado": "Válida"
    }


def ajuste_gamma_agrupada(categorias, bins: dict, x0=(3, 4)) -> dict:
    s = pd.Series(categorias)
    frecuencias = s.value_counts()

    def nll(params):
        a, scale = params
        if a <= 0 or scale <= 0:
            return np.inf
        ll = 0.0
        for categoria, (li, ls) in bins.items():
            ncat = float(frecuencias.get(categoria, 0.0))
            if ncat <= 0:
                continue
            prob = stats.gamma.cdf(ls, a, scale=scale) - stats.gamma.cdf(li, a, scale=scale)
            ll += ncat * math.log(max(prob, 1e-12))
        return -ll

    opt = minimize(nll, x0=np.asarray(x0, dtype=float), method="Nelder-Mead")
    a, scale = opt.x
    return {
        "shape": float(a),
        "scale": float(scale),
        "media": float(a * scale),
        "optimizacion": opt,
    }


def regresion_cuantil(data, y, x, q=0.5):
    """Regresión cuantílica mediante statsmodels.QuantReg."""
    if not (0 < q < 1):
        raise ValueError("q debe estar estrictamente entre 0 y 1.")
    cols = [y] + list(x)
    tmp = data[cols].copy()
    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.dropna()
    if len(tmp) == 0:
        raise ValueError("No hay observaciones válidas para la regresión cuantílica.")
    X = sm.add_constant(tmp[list(x)], has_constant="add")
    return sm.QuantReg(tmp[y], X).fit(q=q)


# ---------------------------------------------------------------------
# Catálogo / routing estadístico
# ---------------------------------------------------------------------

def familia_tipo(tipo: str) -> str:
    if tipo in {"dicotomica", "nominal"}:
        return "categorica"
    if tipo == "ordinal":
        return "ordinal"
    if tipo in {"numerica_discreta", "numerica_continua", "numerica_topcoded"}:
        return "numerica"
    return "categorica"


def prueba_automatica(data, x, y, tipo_x, tipo_y) -> dict:
    fx, fy = familia_tipo(tipo_x), familia_tipo(tipo_y)
    if fx == "categorica" and fy == "categorica":
        return chi_cuadrada(data, x, y)
    if fx == "categorica" and fy in {"ordinal", "numerica"}:
        return kruskal_wallis(data, x, y)
    if fy == "categorica" and fx in {"ordinal", "numerica"}:
        return kruskal_wallis(data, y, x)
    return spearman(data, x, y)


# ---------------------------------------------------------------------
# Random Forest con estado original de respuestas faltantes
# ---------------------------------------------------------------------

def _preparar_rf(
    df_limpio: pd.DataFrame,
    catalogo: pd.DataFrame,
    objetivo: str,
    df_original: Optional[pd.DataFrame] = None,
    excluir: Optional[Iterable[str]] = None,
):
    tipos = dict(zip(catalogo["variable"], catalogo["tipo"]))
    excluir = (
        set(excluir or ())
        | set(EXCLUSIONES_LEAKAGE_DEFAULT.get(objetivo, set()))
        | {objetivo}
    )

    candidatos = [
        c for c in catalogo["variable"]
        if c in df_limpio.columns and c not in excluir
    ]

    y = df_limpio[objetivo].copy()
    valid_y = y.notna()
    columnas_x = {}
    grupos = {}

    for c in candidatos:
        base = df_limpio[c].copy()
        tipo = tipos.get(c, "nominal")
        raw = (
            df_original[c]
            if (df_original is not None and c in df_original.columns)
            else base
        )
        estado = raw.map(_razon_missing_original)

        if familia_tipo(tipo) in {"ordinal", "numerica"}:
            base = pd.to_numeric(base, errors="coerce")
            if base.notna().sum() == 0 or base.nunique(dropna=True) < 2:
                continue
            columnas_x[c] = base
            estado_col = f"{c}__estado_respuesta"
            columnas_x[estado_col] = estado
            grupos[c] = [c, estado_col]
        else:
            base = base.astype("object")
            base = base.where(base.notna(), estado)
            if base.nunique(dropna=False) < 2:
                continue
            columnas_x[c] = base.astype(str)
            grupos[c] = [c]

    X = pd.DataFrame(columnas_x, index=df_limpio.index)
    X = X.loc[valid_y].copy()
    y = y.loc[valid_y].copy()
    return X, y, grupos, tipos


def _score_modelo(modelo, X, y, clasificacion):
    pred = modelo.predict(X)
    if clasificacion:
        return balanced_accuracy_score(y, pred)
    return r2_score(y, pred)


def _permutation_importance_grupos(
    modelo, X, y, grupos, clasificacion,
    n_repeats=5, random_state=42
):
    rng = np.random.default_rng(random_state)
    baseline = _score_modelo(modelo, X, y, clasificacion)
    rows = []
    for nombre, cols in grupos.items():
        diffs = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            for c in cols:
                Xp[c] = X[c].to_numpy()[perm]
            score = _score_modelo(modelo, Xp, y, clasificacion)
            diffs.append(baseline - score)
        rows.append({
            "variable": nombre,
            "rf_importance": float(np.mean(diffs)),
            "rf_importance_std": (
                float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            ),
        })
    return pd.DataFrame(rows), baseline


def random_forest_importance(
    df_limpio: pd.DataFrame,
    catalogo: pd.DataFrame,
    objetivo: str,
    df_original: Optional[pd.DataFrame] = None,
    excluir: Optional[Iterable[str]] = None,
    random_state=42,
    test_size=0.25,
    n_estimators=300,
    n_repeats=5,
) -> tuple[pd.DataFrame, dict]:
    X, y, grupos, tipos = _preparar_rf(
        df_limpio, catalogo, objetivo,
        df_original=df_original, excluir=excluir
    )
    if X.empty or len(y) < 10:
        return pd.DataFrame(
            columns=["variable", "rf_importance", "rf_importance_std", "rf_rank"]
        ), {
            "rf_modelo": None,
            "rf_metrica": None,
            "rf_score_test": np.nan,
            "rf_baseline": np.nan,
        }

    clasificacion = familia_tipo(tipos.get(objetivo, "nominal")) == "categorica"
    if clasificacion:
        y = y.astype(str)
    else:
        y = pd.to_numeric(y, errors="coerce")

    numeric_cols = [
        c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
    ]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    if clasificacion:
        estimador = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        metrica = "Balanced accuracy"
    else:
        estimador = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        metrica = "R²"

    modelo = Pipeline([("pre", pre), ("rf", estimador)])

    indices = np.arange(len(X))
    stratify = None
    if clasificacion:
        counts = pd.Series(y).value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = y

    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=stratify
    )
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    modelo.fit(X_train, y_train)
    score_test = _score_modelo(modelo, X_test, y_test, clasificacion)

    imp, baseline = _permutation_importance_grupos(
        modelo, X_test, y_test, grupos, clasificacion,
        n_repeats=n_repeats, random_state=random_state
    )
    if len(imp):
        imp["rf_rank"] = (
            imp["rf_importance"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        imp = imp.sort_values(["rf_rank", "variable"]).reset_index(drop=True)

    meta = {
        "rf_modelo": type(estimador).__name__,
        "rf_metrica": metrica,
        "rf_score_test": float(score_test),
        "rf_baseline": float(baseline),
    }
    return imp, meta


# ---------------------------------------------------------------------
# Screening masivo
# ---------------------------------------------------------------------

def screening_masivo(
    df_limpio: pd.DataFrame,
    catalogo: pd.DataFrame,
    objetivo: str,
    *,
    df_original: Optional[pd.DataFrame] = None,
    excluir: Optional[Iterable[str]] = None,
    random_state=42,
    test_size=0.25,
    n_estimators=300,
    n_repeats=5,
) -> pd.DataFrame:
    if objetivo not in df_limpio.columns:
        raise KeyError(f"El objetivo '{objetivo}' no existe en el DataFrame.")

    tipos = dict(zip(catalogo["variable"], catalogo["tipo"]))
    if objetivo not in tipos:
        raise KeyError(f"El objetivo '{objetivo}' no está clasificado en catalogo.")

    excluir_screening = set(excluir or ()) | {objetivo}
    excluir_rf = (
        excluir_screening
        | set(EXCLUSIONES_LEAKAGE_DEFAULT.get(objetivo, set()))
    )

    rows = []
    for x in catalogo["variable"]:
        if x in excluir_screening or x not in df_limpio.columns:
            continue
        tipo_x = tipos.get(x)
        tipo_y = tipos[objetivo]
        r = prueba_automatica(
            df_limpio, x, objetivo, tipo_x, tipo_y
        )
        rows.append({
            "variable": x,
            "tipo_x": tipo_x,
            "prueba": r.get("prueba"),
            "n": r.get("n"),
            "estadistico": r.get("estadistico"),
            "p_valor": r.get("p_valor"),
            "significativo": r.get("significativo"),
            "effect_size": r.get("effect_size"),
            "magnitud": r.get("magnitud"),
            "estado": r.get("estado"),
            "advertencia": r.get("advertencia"),
        })

    resultados = pd.DataFrame(rows)

    rf, rf_meta = random_forest_importance(
        df_limpio, catalogo, objetivo,
        df_original=df_original,
        excluir=excluir_rf,
        random_state=random_state,
        test_size=test_size,
        n_estimators=n_estimators,
        n_repeats=n_repeats,
    )

    if len(rf):
        resultados = resultados.merge(
            rf[["variable", "rf_importance", "rf_importance_std", "rf_rank"]],
            on="variable", how="left"
        )
    else:
        resultados["rf_importance"] = np.nan
        resultados["rf_importance_std"] = np.nan
        resultados["rf_rank"] = np.nan

    # RF es el criterio global de orden; las métricas de efecto no se comparan entre pruebas.
    resultados["ranking"] = resultados["rf_rank"]
    resultados.attrs.update({
        "objetivo": objetivo,
        "tipo_objetivo": tipos[objetivo],
        **rf_meta,
    })

    cols = [
        "ranking", "variable", "tipo_x", "prueba", "n", "estadistico",
        "p_valor", "significativo", "effect_size", "magnitud", "estado",
        "advertencia", "rf_importance", "rf_rank",
    ]
    return resultados.reindex(columns=cols).sort_values(
        ["rf_rank", "variable"], na_position="last"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Preparación específica de la encuesta
# ---------------------------------------------------------------------

def preparar_encuesta(df, metadata_cols=None):
    """
    PIPELINE ÚNICA DE LIMPIEZA Y CLASIFICACIÓN DE LA ENCUESTA.

    Devuelve:
        df_limpio
        catalogo
        incidencias

    REGLAS PRINCIPALES
    ------------------
    Sí                  -> 1
    No                  -> 0
    No sé / No recuerdo -> NaN

    Ninguno             -> 0
    0 (No)              -> 0

    9+                  -> 9.1
    30+                 -> 30.1
    36+                 -> 36.1
    50 o más            -> 50.1
    etc.

    IMPORTANTE:
    Los valores X+ -> X.1 son códigos para conservar el orden,
    no valores literalmente fraccionarios.

    Medio baño:
        0 (No) -> 0
        1      -> 1
        2      -> 2
        3      -> 3

    Materiales:
        "1 Concreto armado" -> "Concreto armado"
        "6 No sé"           -> NaN

    La función NO convierte variables nominales a números arbitrarios.
    """

    import pandas as pd
    import numpy as np
    import re
    import unicodedata

    if metadata_cols is None:
        # En esta base, las tres primeras columnas son metadatos operativos
        # y no forman parte del catálogo de variables sustantivas.
        metadata_cols = tuple(df.columns[:3])


    # ==============================================================
    # COPIA DEL DATAFRAME
    # ==============================================================

    df_limpio = df.copy(deep=True)

    catalogo = []
    incidencias = []


    # ==============================================================
    # NORMALIZADOR INTERNO
    # ==============================================================

    def norm(x):
        if pd.isna(x):
            return None
    
        texto = (
            unicodedata.normalize("NFKD", str(x))
            .encode("ascii", "ignore")
            .decode("utf-8")
            .strip()
            .upper()
        )
    
        # Colapsa espacios, tabs, saltos de línea, etc.
        texto = re.sub(r"\s+", " ", texto)
    
        return texto


    # ==============================================================
    # VALORES CONSIDERADOS DESCONOCIDOS
    # ==============================================================

    desconocidos = {
        "NO SE",
        "NO RECUERDO",
        "NO SE / NO RECUERDO",
        "NO SE/NO RECUERDO",
        "NO SABE",
        "NS",
        "N/S",
    }

    no_aplica = {
        "NO APLICA",
        "NO APLICA/ES VIVIENDA HORIZONTAL",
    }


    # ==============================================================
    # MAPAS ORDINALES
    # ==============================================================


    # --------------------------------------------------------------
    # EDAD
    # --------------------------------------------------------------

    mapa_edad = {
        "18-24": 1,
        "25-34": 2,
        "35-44": 3,
        "45-54": 4,
        "55-64": 5,
        "65 Y +": 6,
    }


    # --------------------------------------------------------------
    # ESCOLARIDAD
    #
    # Técnica/comercial/preparatoria se consideran en el mismo
    # escalón general cuando tienen el mismo grado de terminación.
    # --------------------------------------------------------------

    mapa_escolaridad = {
        "SIN ESTUDIOS": 0,

        "PRIMARIA INCOMPLETA": 1,
        "PRIMARIA COMPLETA": 2,

        "SECUNDARIA INCOMPLETA": 3,
        "SECUNDARIA COMPLETA": 4,

        "PREPARATORIA INCOMPLETA": 5,
        "CARRERA TECNICA INCOMPLETA": 5,
        "CARRERA COMERCIAL INCOMPLETA": 5,

        "PREPARATORIA COMPLETA": 6,
        "CARRERA TECNICA COMPLETA": 6,
        "CARRERA COMERCIAL COMPLETA": 6,

        "LICENCIATURA INCOMPLETA": 7,
        "LICENCIATURA COMPLETA": 8,

        "MAESTRIA INCOMPLETA": 9,
        "MAESTRIA COMPLETA": 10,
    }


    # --------------------------------------------------------------
    # MANTENIMIENTO VIVIENDA
    # --------------------------------------------------------------

    mapa_mantenimiento = {
        "MALA (DETERIORO EVIDENTE EN VARIOS COMPONENTES, GRAFITI, VANDALISMO DESATENDIDO)": 1,
        "REGULAR (ALGUNOS ELEMENTOS FALTANTES)": 2,
        "BUENA (PINTURA RECIENTE, ENJARRE, VIDRIOS Y HERRERIA COMPLETA)": 3,
    }


    # --------------------------------------------------------------
    # PISO EN VIVIENDA VERTICAL
    # --------------------------------------------------------------

    mapa_piso = {
        "PLANTA BAJA": 1,
        "PISO INTERMEDIO": 2,
        "ULTIMO PISO": 3,
    }


    # --------------------------------------------------------------
    # CONFORT / CLIMA
    # --------------------------------------------------------------

    mapa_clima = {
        "MUY FRIA": 1,
        "FRIA": 2,
        "NEUTRA": 3,
        "CALUROSA": 4,
        "MUY CALUROSA": 5,
    }


    # --------------------------------------------------------------
    # PERCEPCIÓN DE CAMBIO
    # --------------------------------------------------------------

    mapa_cambio = {
        "CAMBIO NEGATIVAMENTE": -1,
        "NO CAMBIO": 0,
        "CAMBIO POSITIVAMENTE": 1,
    }


    # --------------------------------------------------------------
    # EVALUACIÓN URBANA
    # --------------------------------------------------------------

    mapa_evaluacion = {
        "NO HAY": 0,
        "PESIMO": 1,
        "MALO": 2,
        "REGULAR": 3,
        "BUENO": 4,
        "EXCELENTE": 5,
    }


    # --------------------------------------------------------------
    # IMPACTO COVID
    # --------------------------------------------------------------

    mapa_impacto_covid = {
        "INFECCION LEVE": 1,
        "INFECCION GRAVE": 2,
        "OXIGENO/HOSPITALIZACION": 3,
        "FALLECIMIENTO": 4,
    }


    # --------------------------------------------------------------
    # MODIFICACIONES A LA VIVIENDA
    # --------------------------------------------------------------

    mapa_modificaciones = {
        "NO REALIZO MODIFICACIONES EN ESTE RUBRO": 0,
        "ADAPTACIONES TEMPORALES(REVERSIBLES)": 1,
        "ADAPTACIONES TEMPORALES (REVERSIBLES)": 1,
        "ADAPTACIONES PERMANENTES (FIJAS)": 2,
    }


    # --------------------------------------------------------------
    # CANTIDAD DE FOCOS
    # Se conserva como escala ordinal, NO como cantidad exacta.
    # --------------------------------------------------------------

    mapa_focos = {
        "5 O MENOS FOCOS": 1,
        "ENTRE 6 Y 10 FOCOS": 2,
        "ENTRE 11 Y 15 FOCOS": 3,
        "ENTRE 16 Y 20 FOCOS": 4,
        "21 FOCOS O MAS": 5,
    }


    # --------------------------------------------------------------
    # FRECUENCIA DE USO DE ESPACIOS
    # --------------------------------------------------------------

    mapa_frecuencia_espacios = {
        "NO LOS USABAMOS": 0,
        "UNA VEZ A LA SEMANA O MENOS": 1,
        "DOS VECES POR SEMANA": 2,
        "DIARIO": 3,
    }


    # --------------------------------------------------------------
    # DISTANCIA A PARQUE
    # --------------------------------------------------------------

    mapa_distancia_parque = {
        "A MENOS DE 15 MINUTOS CAMINANDO": 1,
        "ENTRE 15 Y 30 MINUTOS CAMINANDO": 2,
        "MAS DE 30 MINUTOS EN CAMINANDO": 3,
        "MAS DE 30 MINUTOS CAMINANDO": 3,
    }


    # --------------------------------------------------------------
    # CUADRAS A TRANSPORTE
    # --------------------------------------------------------------

    mapa_cuadras = {
        "MENOS DE 5 CUADRAS": 1,
        "DE 6 A 10 CUADRAS": 2,
        "MAS DE 10 CUADRAS": 3,
    }


    # --------------------------------------------------------------
    # TIEMPO DE TRASLADO EN AUTOMÓVIL
    # --------------------------------------------------------------

    mapa_tiempo_traslado = {
        "MENOS DE 30 MINUTOS": 1,
        "DE 30 MINUTOS A 1 HORA": 2,
        "DE 1 HORA A 1 HORA, 30 MINUTOS": 3,
        "ENTRE 1 HORA, 31 MINUTOS A 2 HORAS": 4,
        "DE 2 A 3 HORAS": 5,
        "MAS DE 3 HORAS": 6,
    }


    # --------------------------------------------------------------
    # SEGURIDAD DEL BARRIO
    # --------------------------------------------------------------

    mapa_seguridad = {
        "MENOS SEGURO": -1,
        "IGUAL DE SEGURO": 0,
        "MAS SEGURO": 1,
    }


    # ==============================================================
    # RECORRER TODAS LAS COLUMNAS
    # ==============================================================

    for col in df.columns:

        original = df[col].copy()

        # Variables de diseño: se conservan en el DataFrame pero no
        # forman parte del catálogo de variables sustantivas.
        if col in set(metadata_cols):
            df_limpio[col] = original
            continue

        nombre = norm(col)

        n_original = original.nunique(dropna=True)
        na_original = original.isna().sum()


        # ==========================================================
        # LIMPIEZA BÁSICA
        # ==========================================================

        if (
            pd.api.types.is_object_dtype(original)
            or pd.api.types.is_string_dtype(original)
            or isinstance(original.dtype, pd.CategoricalDtype)
        ):

            s = original.astype("string").str.strip()

            s = s.replace({
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
            })

        else:
            s = original.copy()


        # ==========================================================
        # MATERIAL DE AZOTEA / MUROS
        #
        # 1 Concreto armado -> Concreto armado
        # 7 No sé           -> NaN
        # ==============================================================

        if (
            "TIPO DE MATERIAL" in nombre
            or "MATERIAL DE LA AZOTEA" in nombre
            or "MATERIAL DE LOS MUROS" in nombre
        ):

            s2 = s.astype("string").str.replace(
                r"^\s*\d+\s+",
                "",
                regex=True
            ).str.strip()

            normalizados = s2.map(
                lambda x: norm(x) if pd.notna(x) else None
            )

            mascara_desconocidos = normalizados.isin(
                desconocidos | no_aplica
            )

            s2 = s2.mask(mascara_desconocidos)

            df_limpio[col] = s2

            catalogo.append({
                "variable": col,
                "tipo": "nominal",
                "transformacion":
                    "Eliminación de código numérico inicial; "
                    "No sé -> NaN",
                "n_valores_originales": n_original,
                "n_valores_finales": s2.nunique(dropna=True),
                "na_originales": na_original,
                "na_finales": s2.isna().sum(),
            })

            continue


        # ==========================================================
        # SELECCIÓN DE MAPA ORDINAL
        # ==============================================================

        mapa = None
        transformacion = None
        faltantes_adicionales = set()


        if nombre == "EDAD":

            mapa = mapa_edad
            transformacion = "Edad ordinal 1-6"


        elif nombre == "ESCOLARIDAD":

            mapa = mapa_escolaridad
            transformacion = "Escolaridad ordinal 0-10"


        elif "ESTADO DE MANTENIMIENTO" in nombre:

            mapa = mapa_mantenimiento
            transformacion = "Mala=1, Regular=2, Buena=3"


        elif (
            "VIVIENDA VERTICAL" in nombre
            and "EN QUE PISO" in nombre
        ):

            mapa = mapa_piso

            faltantes_adicionales = {
                "NO APLICA/ES VIVIENDA HORIZONTAL"
            }

            transformacion = (
                "Planta baja=1, intermedio=2, último=3"
            )


        elif "PERCIBE AL CLIMA" in nombre:

            mapa = mapa_clima
            transformacion = "Muy fría=1 ... Muy calurosa=5"


        elif "OPCIONES CAMBIARON" in nombre:

            mapa = mapa_cambio
            transformacion = (
                "Negativo=-1, No cambió=0, Positivo=1"
            )


        elif "IMPACTO EN LA SALUD" in nombre:

            mapa = mapa_impacto_covid
            transformacion = (
                "Leve=1, Grave=2, Hospitalización=3, "
                "Fallecimiento=4"
            )


        elif (
            "MODIFICACIONES O ADAPTACIONES" in nombre
            and "|" in col
        ):

            mapa = mapa_modificaciones
            transformacion = (
                "Ninguna=0, Temporal=1, Permanente=2"
            )


        elif "CUANTOS FOCOS" in nombre:

            mapa = mapa_focos

            faltantes_adicionales = desconocidos

            transformacion = (
                "Rangos de focos convertidos a ordinal 1-5"
            )


        elif (
            "CADA CUANDO" in nombre
            and (
                "ESPACIOS URBANOS" in nombre
                or "AREAS VERDES" in nombre
            )
        ):

            mapa = mapa_frecuencia_espacios

            faltantes_adicionales = {
                "NO HAY ESPACIOS URBANOS Y/O AREAS VERDES EN EL BARRIO"
            }

            transformacion = (
                "Nunca=0, <=1/semana=1, "
                "2/semana=2, Diario=3"
            )


        elif (
            "A QUE DISTANCIA" in nombre
            and (
                "PARQUE" in nombre
                or "ESPACIO PUBLICO" in nombre
            )
        ):

            mapa = mapa_distancia_parque

            faltantes_adicionales = {
                "NO HAY PARQUES O ESPACIOS PUBLICOS CERCANOS"
            }

            transformacion = (
                "<15 min=1, 15-30=2, >30=3"
            )


        elif "CUANTAS CUADRAS" in nombre:

            mapa = mapa_cuadras

            transformacion = (
                "<5=1, 6-10=2, >10=3"
            )


        elif (
            "CUANTO TIEMPO LE TOMA" in nombre
            and "AUTOMOVIL" in nombre
        ):

            mapa = mapa_tiempo_traslado

            transformacion = (
                "Tiempo de traslado convertido a ordinal 1-6"
            )


        elif "COMO EVALUA A LAS CONDICIONES" in nombre:

            mapa = mapa_evaluacion


            transformacion = (
                "No hay=0, Pésimo=1, Malo=2, Regular=3, "
                "Bueno=4, Excelente=5"
            )


        elif (
            "ACTUALMENTE PERCIBE" in nombre
            and "BARRIO" in nombre
            and "SEGURO" in nombre
        ):

            mapa = mapa_seguridad

            transformacion = (
                "Menos seguro=-1, Igual=0, Más seguro=1"
            )


        # ==========================================================
        # APLICAR ORDINAL
        # ==============================================================

        if mapa is not None:

            normalizados = s.map(
                lambda x: norm(x) if pd.notna(x) else None
            )

            resultado = normalizados.map(mapa)

            # Detectar cosas que debían mapearse pero no lo hicieron
            mascara_no_mapeada = (
                s.notna()
                & resultado.isna()
                & ~normalizados.isin(
                    desconocidos
                    | no_aplica
                    | faltantes_adicionales
                )
            )

            if mascara_no_mapeada.any():

                for idx in s.index[mascara_no_mapeada]:

                    incidencias.append({
                        "variable": col,
                        "indice": idx,
                        "valor_original": original.loc[idx],
                        "valor_limpio": np.nan,
                        "motivo": "categoría ordinal no reconocida",
                    })


            df_limpio[col] = pd.to_numeric(
                resultado,
                errors="coerce"
            )


            catalogo.append({
                "variable": col,
                "tipo": "ordinal",
                "transformacion": transformacion,
                "n_valores_originales": n_original,
                "n_valores_finales":
                    df_limpio[col].nunique(dropna=True),
                "na_originales": na_original,
                "na_finales":
                    df_limpio[col].isna().sum(),
            })

            continue


        # ==========================================================
        # DICOTÓMICAS AUTOMÁTICAS
        #
        # Sí       -> 1
        # No       -> 0
        # No sé    -> NaN
        # ==============================================================

        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
        ):

            normalizados = s.map(
                lambda x: norm(x) if pd.notna(x) else None
            )

            valores = set(
                normalizados.dropna().unique()
            )

            valores_reales = valores - desconocidos - no_aplica

            if (
                len(valores_reales) > 0
                and valores_reales <= {"SI", "NO"}
            ):

                resultado = normalizados.map({
                    "SI": 1.0,
                    "NO": 0.0,
                })

                df_limpio[col] = resultado

                catalogo.append({
                    "variable": col,
                    "tipo": "dicotomica",
                    "transformacion":
                        "Sí=1, No=0, desconocido=NaN",
                    "n_valores_originales": n_original,
                    "n_valores_finales":
                        resultado.nunique(dropna=True),
                    "na_originales": na_original,
                    "na_finales": resultado.isna().sum(),
                })

                continue


        # ==========================================================
        # CASO ESPECIAL:
        # MESES PARA RECUPERAR EMPLEO
        #
        # 36+ -> 36.1
        # No se ha recuperado -> NaN
        # ==============================================================

        es_tiempo_recuperacion = (
            "SE RECUPERO EL EMPLEO" in nombre
            and "MESES" in nombre
        )


        # ==========================================================
        # INTENTO DE CONVERSIÓN NUMÉRICA AUTOMÁTICA
        #
        # Todo queda AQUÍ dentro de la función.
        # ==============================================================

        valores_convertidos = []
        conversion_valida = True
        hubo_topcode = False


        for idx, valor in s.items():

            # ------------------------------------------------------
            # NaN
            # ------------------------------------------------------

            if pd.isna(valor):

                valores_convertidos.append(np.nan)
                continue


            texto_original = str(valor).strip()
            texto_norm = norm(valor)


            # ------------------------------------------------------
            # DESCONOCIDOS / NO APLICA
            # ------------------------------------------------------

            if (
                texto_norm in desconocidos
                or texto_norm in no_aplica
            ):

                valores_convertidos.append(np.nan)
                continue


            # ------------------------------------------------------
            # CASO "NO SE HA RECUPERADO"
            # ------------------------------------------------------

            if (
                es_tiempo_recuperacion
                and texto_norm == "NO SE HA RECUPERADO"
            ):

                valores_convertidos.append(np.nan)

                incidencias.append({
                    "variable": col,
                    "indice": idx,
                    "valor_original": valor,
                    "valor_limpio": np.nan,
                    "motivo":
                        "empleo aún no recuperado; duración desconocida",
                })

                continue


            # ------------------------------------------------------
            # NINGUNO -> 0
            # ------------------------------------------------------

            if texto_norm == "NINGUNO":

                valores_convertidos.append(0.0)
                continue


            # ------------------------------------------------------
            # 0 (No) -> 0
            #
            # MEDIO BAÑO SE QUEDA NORMAL:
            # 1 -> 1
            # 2 -> 2
            # 3 -> 3
            # ------------------------------------------------------

            if texto_norm in {
                "0 (NO)",
                "0(NO)",
            }:

                valores_convertidos.append(0.0)
                continue


            # ------------------------------------------------------
            # TOP CODE:
            #
            # 9+     -> 9.1
            # 30+    -> 30.1
            # 36+    -> 36.1
            # ------------------------------------------------------

            match = re.fullmatch(
                r"(\d+(?:[.,]\d+)?)\s*\+",
                texto_original
            )

            if match:

                base = float(
                    match.group(1).replace(",", ".")
                )

                nuevo = base + 0.1

                valores_convertidos.append(nuevo)

                hubo_topcode = True

                incidencias.append({
                    "variable": col,
                    "indice": idx,
                    "valor_original": valor,
                    "valor_limpio": nuevo,
                    "motivo": "top-code X+ convertido a X+0.1",
                })

                continue


            # ------------------------------------------------------
            # TOP CODE TEXTUAL:
            #
            # 50 o más -> 50.1
            # 30 o más -> 30.1
            # ------------------------------------------------------

            match = re.fullmatch(
                r"(\d+(?:[.,]\d+)?)\s+(?:o|y)\s+m[aá]s",
                texto_original,
                flags=re.IGNORECASE
            )

            if match:

                base = float(
                    match.group(1).replace(",", ".")
                )

                nuevo = base + 0.1

                valores_convertidos.append(nuevo)

                hubo_topcode = True

                incidencias.append({
                    "variable": col,
                    "indice": idx,
                    "valor_original": valor,
                    "valor_limpio": nuevo,
                    "motivo":
                        "top-code 'X o más' convertido a X+0.1",
                })

                continue


            # ------------------------------------------------------
            # NÚMERO NORMAL
            # ------------------------------------------------------

            try:

                numero = float(
                    texto_original.replace(",", ".")
                )

                valores_convertidos.append(numero)

            except (ValueError, TypeError):

                conversion_valida = False
                break


        # ==========================================================
        # SI TODA LA COLUMNA ES NUMÉRICA, LA APLICAMOS
        # ==============================================================

        if conversion_valida:

            resultado = pd.Series(
                valores_convertidos,
                index=s.index,
                dtype="float64"
            )


            # ======================================================
            # OUTLIERS EVIDENTES EN PAGOS DE SERVICIOS
            #
            # No usamos IQR aquí porque podría eliminar pagos altos
            # pero reales.
            #
            # Únicamente eliminamos valores absurdamente grandes.
            # ==============================================================

            if (
                "PAGA MENSUALMENTE" in nombre
                and "SERVICIOS" in nombre
            ):

                mascara_outlier = (
                    (resultado < 0)
                    | (resultado > 100000)
                )

                if mascara_outlier.any():

                    for idx in resultado.index[mascara_outlier]:

                        incidencias.append({
                            "variable": col,
                            "indice": idx,
                            "valor_original": original.loc[idx],
                            "valor_limpio": np.nan,
                            "motivo":
                                "outlier monetario evidente (>100,000 o <0)",
                        })

                    resultado.loc[mascara_outlier] = np.nan


            # ======================================================
            # VALORES NEGATIVOS EN CONTEOS
            # ==============================================================

            patrones_conteo = [
                "CUANTOS ",
                "CUANTAS ",
                "CANTIDAD DE ",
                "NUMERO DE ",
            ]

            es_conteo = any(
                patron in nombre
                for patron in patrones_conteo
            )

            if es_conteo:

                mascara_negativo = resultado < 0

                if mascara_negativo.any():

                    for idx in resultado.index[mascara_negativo]:

                        incidencias.append({
                            "variable": col,
                            "indice": idx,
                            "valor_original": original.loc[idx],
                            "valor_limpio": np.nan,
                            "motivo":
                                "conteo negativo imposible",
                        })

                    resultado.loc[mascara_negativo] = np.nan


            df_limpio[col] = resultado


            # ======================================================
            # CLASIFICACIÓN NUMÉRICA
            # ==============================================================

            patrones_continuos = [
                "PAGA MENSUALMENTE",
                "INCREMENTO APROXIMADAMENTE",
                "DE CUANTO FUE ESE INCREMENTO",
                "HORAS EN PROMEDIO",
            ]


            if hubo_topcode:

                tipo = "numerica_topcoded"

                transformacion = (
                    "Numérica; Ninguno/0(No)=0; "
                    "X+=X.1"
                )


            elif any(
                patron in nombre
                for patron in patrones_continuos
            ):

                tipo = "numerica_continua"

                transformacion = "Conversión numérica"


            else:

                valores_validos = resultado.dropna()

                if (
                    len(valores_validos) > 0
                    and np.allclose(
                        valores_validos,
                        np.round(valores_validos)
                    )
                ):

                    tipo = "numerica_discreta"

                else:

                    tipo = "numerica_continua"

                transformacion = "Conversión numérica"


            catalogo.append({
                "variable": col,
                "tipo": tipo,
                "transformacion": transformacion,
                "n_valores_originales": n_original,
                "n_valores_finales":
                    resultado.nunique(dropna=True),
                "na_originales": na_original,
                "na_finales":
                    resultado.isna().sum(),
            })

            continue


        # ==========================================================
        # SI NO ES NUMÉRICA, ORDINAL NI DICOTÓMICA:
        # NOMINAL
        # ==============================================================

        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
        ):

            resultado = s.astype("string").str.strip()

            normalizados = resultado.map(
                lambda x: norm(x) if pd.notna(x) else None
            )

            mascara_missing = normalizados.isin(
                desconocidos | no_aplica
            )

            resultado = resultado.mask(
                mascara_missing
            )

        else:

            resultado = s


        df_limpio[col] = resultado


        catalogo.append({
            "variable": col,
            "tipo": "nominal",
            "transformacion":
                "Texto limpio; desconocidos=NaN",
            "n_valores_originales": n_original,
            "n_valores_finales":
                resultado.nunique(dropna=True),
            "na_originales": na_original,
            "na_finales":
                resultado.isna().sum(),
        })


    # ==============================================================
    # REPORTES
    # ==============================================================

    catalogo = pd.DataFrame(catalogo)

    incidencias = pd.DataFrame(
        incidencias,
        columns=[
            "variable",
            "indice",
            "valor_original",
            "valor_limpio",
            "motivo",
        ]
    )

    return df_limpio, catalogo, incidencias
