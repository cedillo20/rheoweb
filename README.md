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

