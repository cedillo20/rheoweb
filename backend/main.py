import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from engine import IMPORT_ERROR, PYRHEO, available_models, run_fit

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 3000
FRONT = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="PyRheo Web")


def read_table(upload: UploadFile, raw: bytes) -> pd.DataFrame:
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "El archivo pesa más de 5 MB.")
    name = (upload.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(raw))
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception:
        raise HTTPException(400, "No pude leer el archivo. Usa .xlsx, .xls o .csv con encabezados en la primera fila.")


@app.get("/api/status")
def status():
    return {"pyrheo": PYRHEO, "import_error": IMPORT_ERROR, "models": {t: available_models(t) for t in ("creep", "relaxation", "saos", "steady")}}


@app.post("/api/preview")
async def preview(file: UploadFile = File(...)):
    df = read_table(file, await file.read())
    head = df.head(8).astype(object).where(df.head(8).notna(), None)
    return {"columns": [str(c) for c in df.columns], "rows": len(df), "head": head.values.tolist()}


@app.post("/api/fit")
async def fit(file: UploadFile = File(...), test: str = Form(...), x_col: str = Form(...),
              y_col: str = Form(...), y2_col: str = Form(""), models: str = Form(...)):
    if test not in ("creep", "relaxation", "saos", "steady"):
        raise HTTPException(400, "Tipo de ensayo no válido.")
    df = read_table(file, await file.read())
    cols = [x_col, y_col] + ([y2_col] if test == "saos" else [])
    if test == "saos" and not y2_col:
        raise HTTPException(400, "SAOS necesita G′ y G″.")
    try:
        data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    except KeyError:
        raise HTTPException(400, "Alguna columna elegida no existe en el archivo.")
    data = data[(data > 0).all(axis=1)].sort_values(x_col)  # el ajuste ponderado y los ejes log requieren valores > 0
    if len(data) < 5:
        raise HTTPException(400, "Quedaron menos de 5 puntos válidos (positivos y numéricos).")
    if len(data) > MAX_ROWS:
        raise HTTPException(400, f"Máximo {MAX_ROWS} puntos por ajuste.")
    x = data[x_col].to_numpy(float)
    ys = [data[c].to_numpy(float) for c in cols[1:]]
    results = run_fit(test, x, ys, json.loads(models))
    return {"n": len(x), "x": x.tolist(), "ys": [y.tolist() for y in ys], "results": results, "pyrheo": PYRHEO}


@app.get("/")
def index():
    return FileResponse(FRONT / "index.html")


app.mount("/static", StaticFiles(directory=FRONT), name="static")
