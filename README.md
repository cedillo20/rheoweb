# PyRheo Web

Backend FastAPI + frontend HTML/JS (Plotly) para ajustar modelos reológicos con pyRheo.

## Correr en local
    cd backend
    python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn main:app --reload
Abre http://127.0.0.1:8000

Si `pyRheo` no se puede importar, la página sigue funcionando con modelos de respaldo
(Maxwell, SpringPot, PowerLaw, Cross) y avisa en pantalla. Útil para probar la interfaz.

## Estructura
- backend/main.py     endpoints: /api/status, /api/preview, /api/fit
- backend/engine.py   ajuste (pyRheo real o respaldo SciPy), R², R² log, RSS ponderado
- frontend/index.html subir archivo, elegir ensayo/columnas/modelos, curvas y tablas

## Lo que debes verificar primero (no pude instalar ni probar pyRheo)
En `engine.py`, bloque "VERIFICAR":
1. Los imports (`pyRheo.creep_model`, etc.) y el nombre del paquete en pip.
2. Los argumentos del constructor (`model`, `initial_guess`, `minimization_algorithm`, `mittag_leffler_type`).
3. Cómo se leen los parámetros ajustados y el modelo elegido por `auto` (`parameters_`, `model_name`).
4. Que `predict` en SAOS devuelva (G', G'').
Compáralo con los notebooks del repo y ajusta `_fit_pyrheo`; el resto no cambia.

## Para publicarlo
Necesita un host con Python (Render, Railway, Fly.io o un VPS). Un hosting estático no sirve.
Ajustes de Mittag-Leffler pueden tardar: conviene poner un timeout y limitar puntos (ya hay tope de 3000).
