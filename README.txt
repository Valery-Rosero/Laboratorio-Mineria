LABORATORIO MINERIA DE DATOS - ENTREGA FINAL (DATOS REALES)
============================================================

DATASETS REALES:
  dolar_data.csv    -> 500 registros
  glucosa_data.csv  -> 2000 registros
  energia_data.csv  -> 10000 registros

RESULTADOS:
  Dolar   -> R^2 = 0.9963 (99.6%) EXCELENTE
  Glucosa -> R^2 = 0.7353 (73.53) BUENO (justificado academicamente)
  Energia -> R^2 = 0.8968 (89.7%) MUY BUENO

COMO EJECUTAR:
  1. pip install pandas numpy scikit-learn matplotlib seaborn streamlit
  2. python laboratorio_modelos.py   (entrena y exporta los .pkl)
  3. streamlit run app.py            (interfaz web en localhost:8501)
