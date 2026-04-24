"""
Interfaz Web – Laboratorio Minería de Datos
Regresión Lineal Múltiple: Dólar | Glucosa | Energía
"""

import streamlit as st
import pickle
import numpy as np
import os

# ── Configuración de página ───────────────────────────────────────
st.set_page_config(
    page_title="Lab Minería de Datos",
    page_icon="🔬",
    layout="centered",
)

# ── CSS personalizado ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
}
h1 { font-family: 'Space Mono', monospace; color: #E6EDF3; }
h2, h3 { font-family: 'Space Mono', monospace; }

.pred-box {
    background: linear-gradient(135deg, #1C2128, #21262D);
    border: 1px solid #30363D;
    border-left: 4px solid;
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 20px;
    text-align: center;
}
.pred-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    display: block;
    margin-top: 8px;
}
.metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
    margin: 4px;
}
.coef-table { width: 100%; }
.coef-row { padding: 6px 0; border-bottom: 1px solid #21262D; }
</style>
""", unsafe_allow_html=True)

# ── Cargar modelos ────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODELOS = {}

for nombre in ["dolar", "glucosa", "energia"]:
    ruta = os.path.join(BASE, "modelos", f"modelo_{nombre}.pkl")
    if os.path.exists(ruta):
        with open(ruta, "rb") as f:
            MODELOS[nombre] = pickle.load(f)

# ── Encabezado ────────────────────────────────────────────────────
st.markdown("# 🔬 Lab Minería de Datos")
st.markdown("**Regresión Lineal Múltiple** — CRISP-DM")
st.divider()

# ── Selector de ejercicio ─────────────────────────────────────────
ejercicio = st.selectbox(
    "Selecciona el escenario de predicción:",
    options=["💵 Precio del Dólar", "🩸 Nivel de Glucosa", "⚡ Consumo de Energía"],
    index=0,
)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# EJERCICIO 1 — PRECIO DEL DÓLAR
# ════════════════════════════════════════════════════════════════
if "Dólar" in ejercicio:
    st.markdown("## 💵 Predicción del Precio del Dólar")
    st.caption("Modelo: Regresión Lineal Múltiple | Variables: Día, Inflación, Tasa de interés")

    col1, col2, col3 = st.columns(3)
    with col1:
        dia = st.number_input("📅 Día", min_value=1, max_value=500, value=100, step=1)
    with col2:
        inflacion = st.number_input("📈 Inflación (%)", min_value=0.0, max_value=20.0,
                                    value=2.5, step=0.1, format="%.2f")
    with col3:
        tasa = st.number_input("🏦 Tasa de Interés (%)", min_value=0.0, max_value=30.0,
                               value=5.0, step=0.1, format="%.2f")

    if st.button("🚀 Predecir Precio del Dólar", use_container_width=True):
        modelo = MODELOS.get("dolar")
        if modelo:
            pred = modelo.predict([[dia, inflacion, tasa]])[0]
            coefs = dict(zip(["Dia", "Inflacion", "Tasa_interes"], modelo.coef_))

            st.markdown(f"""
            <div class="pred-box" style="border-left-color: #E63946;">
                <span style="color:#8B949E; font-size:0.95rem;">PRECIO DEL DÓLAR PREDICHO</span>
                <span class="pred-value" style="color:#E63946;">$ {pred:,.2f} COP</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Interpretación de coeficientes")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("β Día", f"{coefs['Dia']:.4f}",
                          help="Por cada día adicional, el dólar sube ~$14.80 COP")
            with c2:
                st.metric("β Inflación", f"{coefs['Inflacion']:.4f}",
                          help="1% más de inflación → ~$79.26 más en el dólar")
            with c3:
                st.metric("β Tasa Interés", f"{coefs['Tasa_interes']:.4f}",
                          help="1% más en tasa → ~$49.60 más en el dólar")

            st.info(f"**Intercepto β₀** = {modelo.intercept_:.2f} | "
                    f"**R²** = (99.59% varianza explicada)")
        else:
            st.error("Modelo no encontrado. Ejecuta primero laboratorio_modelos.py")

# ════════════════════════════════════════════════════════════════
# EJERCICIO 2 — GLUCOSA
# ════════════════════════════════════════════════════════════════
elif "Glucosa" in ejercicio:
    st.markdown("## 🩸 Predicción del Nivel de Glucosa")
    st.caption("Modelo: Regresión Lineal Múltiple | Variables: Edad, IMC, Actividad Física")

    col1, col2, col3 = st.columns(3)
    with col1:
        edad = st.number_input("👤 Edad (años)", min_value=1, max_value=120, value=45, step=1)
    with col2:
        imc = st.number_input("⚖️ IMC (kg/m²)", min_value=10.0, max_value=60.0,
                              value=27.0, step=0.1, format="%.1f")
    with col3:
        actividad = st.number_input("🏃 Actividad Física (h/semana)", min_value=0.0,
                                    max_value=40.0, value=5.0, step=0.5, format="%.1f")

    if st.button("🚀 Predecir Nivel de Glucosa", use_container_width=True):
        modelo = MODELOS.get("glucosa")
        if modelo:
            pred = modelo.predict([[edad, imc, actividad]])[0]
            coefs = dict(zip(["Edad", "IMC", "Actividad_Fisica"], modelo.coef_))

            color_glucosa = "#E63946" if pred > 126 else "#2A9D8F"
            estado = "⚠️ ELEVADA" if pred > 126 else "✅ NORMAL"

            st.markdown(f"""
            <div class="pred-box" style="border-left-color: #2A9D8F;">
                <span style="color:#8B949E; font-size:0.95rem;">NIVEL DE GLUCOSA PREDICHO</span>
                <span class="pred-value" style="color:#2A9D8F;">{pred:.1f} mg/dL  {estado}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Interpretación de coeficientes")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("β Edad", f"{coefs['Edad']:.4f}",
                          help="Cada año adicional aumenta ~0.81 mg/dL de glucosa")
            with c2:
                st.metric("β IMC", f"{coefs['IMC']:.4f}",
                          help="1 punto más de IMC → +2.58 mg/dL de glucosa")
            with c3:
                st.metric("β Actividad Física", f"{coefs['Actividad_Fisica']:.4f}",
                          help="1 hora/semana más → -3.43 mg/dL (PROTECTOR)")

            st.info(f"**Variable de mayor impacto:** Actividad Física (mayor |β| relativo) | "
                    f"**R²** = (73.5% varianza explicada)")
            st.caption("Referencia: Glucosa normal en ayunas < 100 mg/dL | Prediabetes 100-125 | Diabetes ≥ 126")
        else:
            st.error("Modelo no encontrado. Ejecuta primero laboratorio_modelos.py")

# ════════════════════════════════════════════════════════════════
# EJERCICIO 3 — ENERGÍA
# ════════════════════════════════════════════════════════════════
elif "Energía" in ejercicio:
    st.markdown("## ⚡ Predicción del Consumo de Energía")
    st.caption("Modelo: Regresión Lineal Múltiple | Variables: Temperatura, Hora, Día de Semana")

    DIAS_NOMBRES = {1:"Lunes", 2:"Martes", 3:"Miércoles",
                    4:"Jueves", 5:"Viernes", 6:"Sábado", 7:"Domingo"}

    col1, col2, col3 = st.columns(3)
    with col1:
        temperatura = st.number_input("🌡️ Temperatura (°C)", min_value=-20.0, max_value=50.0,
                                      value=22.0, step=0.5, format="%.1f")
    with col2:
        hora = st.number_input("🕐 Hora del día (1-24)", min_value=1, max_value=24, value=12, step=1)
    with col3:
        dia_num = st.selectbox("📅 Día de la semana",
                               options=list(DIAS_NOMBRES.keys()),
                               format_func=lambda x: f"{x} – {DIAS_NOMBRES[x]}")

    if st.button("🚀 Predecir Consumo de Energía", use_container_width=True):
        modelo = MODELOS.get("energia")
        if modelo:
            pred = modelo.predict([[temperatura, hora, dia_num]])[0]
            coefs = dict(zip(["Temperatura", "Hora", "Dia_Semana"], modelo.coef_))

            st.markdown(f"""
            <div class="pred-box" style="border-left-color: #E9C46A;">
                <span style="color:#8B949E; font-size:0.95rem;">CONSUMO DE ENERGÍA PREDICHO</span>
                <span class="pred-value" style="color:#E9C46A;">{pred:.2f} kWh</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Interpretación de coeficientes")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("β Temperatura", f"{coefs['Temperatura']:.4f}",
                          help="1°C más → +3.22 kWh de consumo")
            with c2:
                st.metric("β Hora", f"{coefs['Hora']:.4f}",
                          help="Cada hora avanzada del día → +5.56 kWh")
            with c3:
                st.metric("β Día Semana", f"{coefs['Dia_Semana']:.4f}",
                          help="Días posteriores de la semana reducen consumo (-10.38/día)")

            st.info(f"**Variable de mayor impacto:** Hora del día (β=5.56, mayor magnitud) | "
                    f"**R²** = 90.69 | **RMSE** = 20.56 kWh")
        else:
            st.error("Modelo no encontrado. Ejecuta primero laboratorio_modelos.py")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#484F58; font-size:0.8rem;">
    Laboratorio Minería de Datos · CRISP-DM · sklearn LinearRegression · pickle
</div>
""", unsafe_allow_html=True)
