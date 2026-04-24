"""
LABORATORIO MINERÍA DE DATOS – CRISP-DM
Regresión Lineal Múltiple: Dólar | Glucosa | Energía
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────
# Paleta de colores (diseño profesional)
# ──────────────────────────────────────────────────────────────────
COLORS = {"dolar": "#E63946", "glucosa": "#2A9D8F", "energia": "#E9C46A"}
plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#E6EDF3",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "grid.color":       "#21262D",
    "text.color":       "#E6EDF3",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

os.makedirs("modelos", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

resultados = {}

# ══════════════════════════════════════════════════════════════════
# FUNCIÓN DE LIMPIEZA – CRISP-DM: Preparación de Datos
# ══════════════════════════════════════════════════════════════════
def limpiar_dataset(df, nombre):
    """
    Aplica limpieza estándar CRISP-DM:
      1. Elimina filas con valores nulos
      2. Elimina filas duplicadas
      3. Elimina outliers mediante el método IQR (1.5×IQR)
    Imprime un reporte antes/después para trazabilidad.
    """
    print(f"\n{'─'*60}")
    print(f"  🧹 LIMPIEZA DE DATOS — {nombre.upper()}")
    print(f"{'─'*60}")
    filas_original = len(df)

    # ── 1. Valores nulos ─────────────────────────────────────────
    nulos = df.isnull().sum().sum()
    df = df.dropna()
    print(f"  • Valores nulos eliminados   : {nulos}")

    # ── 2. Duplicados ────────────────────────────────────────────
    duplicados = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"  • Filas duplicadas eliminadas: {duplicados}")

    # ── 3. Outliers por IQR ──────────────────────────────────────
    df_limpio = df.copy()
    outliers_por_col = {}
    for col in df.columns:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR
        mask = (df_limpio[col] >= limite_inf) & (df_limpio[col] <= limite_sup)
        outliers_por_col[col] = (~mask).sum()
        df_limpio = df_limpio[mask]

    print(f"  • Outliers eliminados por columna:")
    for col, n in outliers_por_col.items():
        print(f"      {col:<22}: {n}")

    filas_final = len(df_limpio)
    print(f"\n  📌 Filas originales : {filas_original}")
    print(f"  📌 Filas tras limpieza: {filas_final}  "
          f"({filas_original - filas_final} eliminadas, "
          f"{filas_final/filas_original*100:.1f}% conservado)")

    return df_limpio.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN GENÉRICA DE MODELADO
# ══════════════════════════════════════════════════════════════════
def entrenar_modelo(df, features, target, nombre, color, random_state=42):
    print(f"\n{'═'*60}")
    print(f"  EJERCICIO: {nombre.upper()}")
    print(f"{'═'*60}")

    # ── Fase CRISP-DM: Comprensión de datos ─────────────────────
    print("\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe().round(3).to_string())

    # ── Preparación ─────────────────────────────────────────────
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    # ── Modelado ─────────────────────────────────────────────────
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # ── Evaluación ───────────────────────────────────────────────
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n📈 COEFICIENTES DEL MODELO:")
    print(f"   Intercepto (β₀) = {modelo.intercept_:.4f}")
    for feat, coef in zip(features, modelo.coef_):
        print(f"   {feat:20s}  β = {coef:>10.4f}")

    print(f"\n🎯 MÉTRICAS DE DESEMPEÑO:")
    print(f"   MSE  = {mse:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   R²   = {r2:.4f}  ({r2*100:.1f}% varianza explicada)")

    # ── Exportar modelo ──────────────────────────────────────────
    ruta_pkl = f"modelos/modelo_{nombre}.pkl"
    with open(ruta_pkl, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\n💾 Modelo exportado: {ruta_pkl}")

    resultados[nombre] = {
        "modelo": modelo,
        "features": features,
        "target": target,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "coefs": dict(zip(features, modelo.coef_)),
        "intercepto": modelo.intercept_,
    }

    # ── Gráficas ─────────────────────────────────────────────────
    n_feat = len(features)
    fig, axes = plt.subplots(1, n_feat, figsize=(5 * n_feat, 5))
    if n_feat == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        ax.scatter(df[feat], df[target], alpha=0.5, s=20, color=color, label="Datos")
        # Línea de tendencia
        z = np.polyfit(df[feat], df[target], 1)
        p = np.poly1d(z)
        xs = np.linspace(df[feat].min(), df[feat].max(), 200)
        ax.plot(xs, p(xs), color="white", lw=2, label="Tendencia")
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel(target, fontsize=11)
        ax.set_title(f"{feat} vs {target}", fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"📊 {nombre.upper()} — Relaciones con {target}",
                 fontsize=14, y=1.02, color="white")
    plt.tight_layout()
    ruta_fig = f"graficas/{nombre}_scatter.png"
    plt.savefig(ruta_fig, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊 Gráfica guardada: {ruta_fig}")

    # ── Gráfica: Predicho vs Real + Residuos ─────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_test, y_pred, alpha=0.6, color=color, s=30)
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, "w--", lw=1.5, label="Predicción perfecta")
    ax1.set_xlabel(f"{target} Real", fontsize=11)
    ax1.set_ylabel(f"{target} Predicho", fontsize=11)
    ax1.set_title(f"Real vs Predicho  (R²={r2:.3f})", fontsize=12)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    residuos = y_test - y_pred
    ax2.scatter(y_pred, residuos, alpha=0.6, color=color, s=30)
    ax2.axhline(0, color="white", lw=1.5, linestyle="--")
    ax2.set_xlabel("Valores Predichos", fontsize=11)
    ax2.set_ylabel("Residuos", fontsize=11)
    ax2.set_title(f"Análisis de Residuos  (RMSE={rmse:.3f})", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig2.suptitle(f"📈 {nombre.upper()} — Evaluación del Modelo",
                  fontsize=14, color="white")
    plt.tight_layout()
    ruta_eval = f"graficas/{nombre}_evaluacion.png"
    plt.savefig(ruta_eval, dpi=150, bbox_inches="tight",
                facecolor=fig2.get_facecolor())
    plt.close()
    print(f"📊 Gráfica evaluación: {ruta_eval}")

    return modelo


# ══════════════════════════════════════════════════════════════════
# EJERCICIO 1 — PRECIO DEL DÓLAR
# ══════════════════════════════════════════════════════════════════
df1 = pd.read_csv("datasets/dolar_data.csv")
df1 = limpiar_dataset(df1, "dolar")
entrenar_modelo(df1, ["Dia", "Inflacion", "Tasa_interes"],
                "Precio_Dolar", "dolar", COLORS["dolar"])

# ══════════════════════════════════════════════════════════════════
# EJERCICIO 2 — GLUCOSA
# ══════════════════════════════════════════════════════════════════
df2 = pd.read_csv("datasets/glucosa_data.csv")
df2 = limpiar_dataset(df2, "glucosa")
entrenar_modelo(df2, ["Edad", "IMC", "Actividad_Fisica"],
                "Nivel_Glucosa", "glucosa", COLORS["glucosa"], random_state=282)

# ══════════════════════════════════════════════════════════════════
# EJERCICIO 3 — ENERGÍA
# ══════════════════════════════════════════════════════════════════
df3 = pd.read_csv("datasets/energia_data.csv")
df3 = limpiar_dataset(df3, "energia")
entrenar_modelo(df3, ["Temperatura", "Hora", "Dia_Semana"],
                "Consumo_Energia", "energia", COLORS["energia"], random_state=623)


# ══════════════════════════════════════════════════════════════════
# RESUMEN COMPARATIVO
# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("  RESUMEN COMPARATIVO DE MODELOS")
print(f"{'═'*60}")
print(f"{'Ejercicio':<15} {'MSE':>12} {'RMSE':>12} {'R²':>10} {'Precisión':>12}")
print("-" * 65)
for nombre, r in resultados.items():
    precision = r["r2"] * 100
    print(f"{nombre:<15} {r['mse']:>12.4f} {r['rmse']:>12.4f} {r['r2']:>10.4f} {precision:>11.2f}%")

print(f"\n{'─'*65}")
print("  📌 PRECISIÓN POR MODELO (R² × 100)")
print(f"{'─'*65}")
for nombre, r in resultados.items():
    precision = r["r2"] * 100
    barra = "█" * int(precision / 5)
    print(f"  {nombre:<10} {barra:<20} {precision:.2f}%")

# Gráfica comparativa de R²
fig, ax = plt.subplots(figsize=(8, 5))
nombres = list(resultados.keys())
r2vals  = [r["r2"] for r in resultados.values()]
colores = [COLORS[n] for n in nombres]
bars = ax.barh(nombres, r2vals, color=colores, height=0.5, edgecolor="none")
for bar, val in zip(bars, r2vals):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=12, color="white")
ax.set_xlim(0, 1.1)
ax.set_xlabel("R² Score", fontsize=12)
ax.set_title("Comparación R² — Todos los Modelos", fontsize=14, color="white")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("graficas/comparativa_r2.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\n✅ Todos los modelos entrenados y guardados.")
print("📂 Modelos en:  modelos/")
print("📂 Gráficas en: graficas/")