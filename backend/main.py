import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict

from engine import IMPORT_ERROR, PYRHEO, available_models, predict_response, run_fit

MAX_BYTES = 10 * 1024 * 1024
MAX_POINTS = 150  # puntos que se mandan al ajuste (muestreo logarítmico); los ensayos largos traen decenas de miles
FRONT = Path(__file__).resolve().parent.parent / "frontend"

# Columnas calculadas para el formato de los reómetros de FIME (esfuerzo en MPa, deformación en %)
J_NAME = "J(t) = Deformación/Tensión [1/MPa]"
G_NAME = "G(t) = Tensión/Deformación [MPa]"

app = FastAPI(title="PyRheo Web")


def _norm(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


# ---------------------------------------------------------------- lectura
def read_raw(upload: UploadFile, raw: bytes, sheet: str = ""):
    """Lee el archivo SIN encabezado. Devuelve (DataFrame crudo, hojas, hoja usada)."""
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "El archivo pesa más de 10 MB.")
    name = (upload.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(io.BytesIO(raw))
            sheets = [str(s) for s in xl.sheet_names]
            used = sheet if sheet in sheets else sheets[0]
            return xl.parse(used, header=None), sheets, used
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", header=None)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", header=None, encoding="latin-1")
        return df, ["datos"], "datos"
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "No pude leer el archivo. Usa .xlsx, .xls o .csv.")


def generic_table(rdf: pd.DataFrame) -> pd.DataFrame:
    """Tabla normal: la primera fila son los encabezados."""
    seen, names = {}, []
    for i, v in enumerate(rdf.iloc[0]):
        n = str(v).strip() if pd.notna(v) and str(v).strip() else f"columna {i + 1}"
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    df = rdf.iloc[1:].reset_index(drop=True)
    df.columns = names
    return df.infer_objects()


def fime_meta(df: pd.DataFrame):
    """Nombres de las columnas de tiempo, esfuerzo y deformación de una muestra FIME."""
    def first(prefix):
        return next((c for c in df.columns if _norm(c).startswith(prefix)), None)
    return {"t": first("tiempo"), "s": first("tension"), "e": first("deformacion")}


def fime_samples(rdf: pd.DataFrame):
    """Formato de los reómetros de FIME:
       fila 1: M1, M2, ... (una etiqueta por muestra, cada una abre un bloque de columnas)
       fila 2: nombres (Tiempo, Fuerza, Desplazamiento, Tension, Deformacion)
       fila 3: unidades (seg, N, mm, N/mm2, %)  ·  fila 4 en adelante: datos.
       Devuelve {etiqueta: DataFrame} o None si el archivo no tiene ese formato."""
    if rdf.shape[0] < 4 or rdf.shape[1] < 2:
        return None
    r0 = [str(v).strip() if pd.notna(v) else "" for v in rdf.iloc[0]]
    starts = [i for i, v in enumerate(r0) if re.fullmatch(r"M\d+", v, re.I)]
    if not starts or not _norm(rdf.iloc[1, starts[0]]).startswith(("tiempo", "time")):
        return None
    out = {}
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else rdf.shape[1]
        names = []
        for j in range(s, e):
            n = str(rdf.iloc[1, j]).strip() if pd.notna(rdf.iloc[1, j]) else ""
            u = str(rdf.iloc[2, j]).strip() if pd.notna(rdf.iloc[2, j]) else ""
            names.append(f"{n} [{u}]" if n and u else (n or f"columna {j + 1}"))
        sub = rdf.iloc[3:, s:e].apply(pd.to_numeric, errors="coerce")
        sub.columns = names
        sub = sub.dropna(how="all").reset_index(drop=True)
        m = fime_meta(sub)
        if not (m["t"] and m["s"] and m["e"]):
            return None  # no trae tiempo, esfuerzo y deformación: se trata como tabla normal
        eps, sig = sub[m["e"]] / 100.0, sub[m["s"]]
        sub[J_NAME] = (eps / sig).where(sig > 0)
        sub[G_NAME] = (sig / eps).where(eps > 0)
        out[r0[s].upper()] = sub
    return out


def load_table(upload, raw, sheet="", sample=""):
    rdf, sheets, used = read_raw(upload, raw, sheet)
    fs = fime_samples(rdf)
    if fs:
        cur = sample if sample in fs else next(iter(fs))
        return {"df": fs[cur], "sheets": sheets, "sheet": used, "samples": list(fs), "sample": cur, "fime": True, "all": fs}
    if rdf.shape[0] < 2:
        raise HTTPException(400, "El archivo no tiene datos.")
    return {"df": generic_table(rdf), "sheets": sheets, "sheet": used, "samples": [], "sample": "", "fime": False, "all": None}


# ---------------------------------------------------------------- detección del ensayo
def classify_fime(df: pd.DataFrame) -> str:
    """creep: esfuerzo constante y la deformación crece · relajación: deformación constante y el
       esfuerzo cae · tensión: ninguno de los dos se mantiene (ensayo a velocidad constante hasta romper)."""
    m = fime_meta(df)
    d = df[[m["s"], m["e"]]].dropna()
    h = len(d) // 2
    if len(d) < 20:
        return ""
    cv = lambda v: float(np.std(v) / abs(np.mean(v))) if np.mean(v) else np.inf
    cvs, cve = cv(d[m["s"]].to_numpy()[h:]), cv(d[m["e"]].to_numpy()[h:])  # segunda mitad del ensayo
    if min(cvs, cve) > 0.03:
        return "tension"
    return "relaxation" if cve < cvs else "creep"


def guess_columns(columns):
    """Adivina ensayo y columnas por el nombre del encabezado (tablas que no son del reómetro FIME)."""
    low = [str(c).lower() for c in columns]

    def find(pattern):
        for c, l in zip(columns, low):
            if re.search(pattern, l):
                return str(c)
        return None

    gdot, eta = find(r"shear.?rate|γ|gamma|rate|veloc"), find(r"visc|η|eta")
    w, g1, g2 = find(r"^w|ω|omega|freq|frec"), find(r"g'|g′|storage|almac|^g1|g_p"), find(r"g''|g″|loss|p[eé]rd|^g2|g_pp")
    t = find(r"^t|time|tiempo")
    j, g = find(r"^j|compl"), find(r"^g|modul|módul")
    if gdot and eta:
        return {"test": "steady", "x": gdot, "y": eta}
    if w and g1 and g2:
        return {"test": "saos", "x": w, "y": g1, "y2": g2}
    if t and j:
        return {"test": "creep", "x": t, "y": j}
    if t and g:
        return {"test": "relaxation", "x": t, "y": g}
    return {}


def guess_fime(df):
    m, kind = fime_meta(df), classify_fime(df)
    if kind == "tension":
        return {"test": "tension", "x": m["e"], "y": m["s"]}
    if kind == "creep":
        return {"test": "creep", "x": m["t"], "y": J_NAME}
    if kind == "relaxation":
        return {"test": "relaxation", "x": m["t"], "y": G_NAME}
    return {}


# ---------------------------------------------------------------- preprocesado
def fime_segment(df, test):
    """Quita la rampa de carga y reinicia el tiempo en el inicio de la parte útil.
       relajación: la parte útil empieza en el pico de esfuerzo (fin de la rampa de deformación).
       creep: empieza cuando el esfuerzo alcanza 98 % de su valor constante."""
    m = fime_meta(df)
    d = df.dropna(subset=[m["t"], m["s"]]).reset_index(drop=True)
    sig = d[m["s"]].to_numpy(float)
    if test == "relaxation":
        i0 = int(np.argmax(sig))
    else:
        plateau = float(np.median(sig[len(sig) // 2:]))
        i0 = int(np.argmax(sig >= 0.98 * plateau))
    out = d.iloc[i0:].copy()
    out[m["t"]] = out[m["t"]] - out[m["t"]].iloc[0]
    return out


def thin(data, xcol, n=MAX_POINTS):
    """Reduce a ~n puntos espaciados en escala logarítmica (data ya viene ordenada por xcol, x > 0)."""
    if len(data) <= n:
        return data
    x = data[xcol].to_numpy(float)
    targets = np.logspace(np.log10(x.min()), np.log10(x.max()), n)
    idx = np.unique(np.clip(np.searchsorted(x, targets), 0, len(x) - 1))
    return data.iloc[idx]


# ---------------------------------------------------------------- endpoints
@app.get("/api/status")
def status():
    return {"pyrheo": PYRHEO, "import_error": IMPORT_ERROR, "models": {t: available_models(t) for t in ("creep", "relaxation", "saos", "steady")}}


@app.post("/api/preview")
async def preview(file: UploadFile = File(...), sheet: str = Form(""), sample: str = Form("")):
    tb = load_table(file, await file.read(), sheet, sample)
    df = tb["df"]
    head = df.head(8).astype(object).where(df.head(8).notna(), None)
    guess = guess_fime(df) if tb["fime"] else guess_columns([str(c) for c in df.columns])
    return {"columns": [str(c) for c in df.columns], "rows": len(df), "head": head.values.tolist(),
            "sheets": tb["sheets"], "sheet": tb["sheet"], "samples": tb["samples"], "sample": tb["sample"],
            "format": "FIME" if tb["fime"] else "", "guess": guess}


@app.post("/api/fit")
async def fit(file: UploadFile = File(...), test: str = Form(...), x_col: str = Form(...),
              y_col: str = Form(...), y2_col: str = Form(""), models: str = Form(...),
              sheet: str = Form(""), sample: str = Form(""), trim: str = Form("0")):
    if test not in ("creep", "relaxation", "saos", "steady"):
        raise HTTPException(400, "Tipo de ensayo no válido.")
    tb = load_table(file, await file.read(), sheet, sample)
    df = tb["df"]
    if tb["fime"] and test in ("creep", "relaxation") and trim == "1":
        df = fime_segment(df, test)
    cols = [x_col, y_col] + ([y2_col] if test == "saos" else [])
    if test == "saos" and not y2_col:
        raise HTTPException(400, "SAOS necesita G′ y G″.")
    try:
        data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    except KeyError:
        raise HTTPException(400, "Alguna columna elegida no existe en el archivo.")
    data = data[(data > 0).all(axis=1)].sort_values(x_col)  # el ajuste ponderado y los ejes log requieren valores > 0
    if not tb["fime"] and test == "relaxation" and trim == "1" and len(data):
        data = data.iloc[int(np.argmax(data[y_col].to_numpy())):].copy()  # desde el pico, quita la rampa de carga
        data[x_col] = data[x_col] - data[x_col].iloc[0]
        data = data[data[x_col] > 0]
    if len(data) < 5:
        raise HTTPException(400, "Quedaron menos de 5 puntos válidos (positivos y numéricos).")
    n_raw = len(data)
    data = thin(data, x_col)
    x = data[x_col].to_numpy(float)
    ys = [data[c].to_numpy(float) for c in cols[1:]]
    results = run_fit(test, x, ys, json.loads(models))

    resp = {"n": len(x), "n_raw": n_raw, "x": x.tolist(), "ys": [y.tolist() for y in ys], "results": results, "pyrheo": PYRHEO}
    if tb["fime"] and test in ("creep", "relaxation"):
        resp["axes"] = ["t [s]", "J(t) [1/MPa]" if test == "creep" else "G(t) [MPa]"]
        resp["note"] = (f"Formato FIME, muestra {tb['sample']}. " +
                        (("Tiempo reiniciado cuando el esfuerzo se vuelve constante. " if test == "creep"
                          else "Tiempo reiniciado en el pico de esfuerzo. ") if trim == "1" else "") +
                        ("J = Deformación/Tensión en 1/MPa. " if test == "creep" else "G = Tensión/Deformación en MPa. ") +
                        f"Se ajustaron {len(x)} de {n_raw} puntos (muestreo logarítmico).")
    return resp


@app.post("/api/tensile")
async def tensile(file: UploadFile = File(...), sheet: str = Form(""), x_col: str = Form(""), y_col: str = Form("")):
    """Ensayo de tensión: curvas esfuerzo-deformación y resumen. No usa pyRheo."""
    tb = load_table(file, await file.read(), sheet, "")
    series = {}
    if tb["fime"]:
        for label, d in tb["all"].items():
            m = fime_meta(d)
            series[label] = pd.DataFrame({"e": d[m["e"]], "s": d[m["s"]]})
    else:
        try:
            series["M1"] = pd.DataFrame({"e": tb["df"][x_col], "s": tb["df"][y_col]})
        except KeyError:
            raise HTTPException(400, "Elige las columnas de deformación y esfuerzo.")
    out = []
    for label, d in series.items():
        d = d.apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) < 20:
            continue
        ev, sv = d["e"].to_numpy(float), d["s"].to_numpy(float)
        imax = int(np.argmax(sv))
        smax = float(sv[imax])
        below = np.where(sv[imax:] < 0.5 * smax)[0]
        ebreak = float(ev[imax + below[0]]) if len(below) else None
        # módulo: pendiente máxima (regresión en ventanas deslizantes) antes del esfuerzo máximo, deformación en fracción
        w = max(10, (imax + 1) // 30)
        best = None
        for i in range(0, max(1, imax + 1 - w), max(1, w // 2)):
            seg_e, seg_s = ev[i:i + w] / 100.0, sv[i:i + w]
            if len(seg_e) >= 3 and np.ptp(seg_e) > 0:
                k = float(np.polyfit(seg_e, seg_s, 1)[0])
                best = k if best is None or k > best else best
        step = max(1, int(np.ceil(len(ev) / 600)))
        out.append({"sample": label, "strain": ev[::step].tolist(), "stress": sv[::step].tolist(),
                    "E": best, "sigma_max": smax, "eps_at_sigma_max": float(ev[imax]), "eps_break": ebreak})
    if not out:
        raise HTTPException(400, "No encontré datos suficientes de esfuerzo y deformación.")
    return {"results": out, "format": "FIME" if tb["fime"] else ""}


class PredictIn(BaseModel):
    model: str
    params: Dict[str, float]
    target: str            # "saos" | "relaxation" | "creep"
    x_min: float           # 10^x_min ... 10^x_max (rad/s para saos, s para las otras)
    x_max: float
    n: int = 200


@app.post("/api/predict")
def predict(q: PredictIn):
    """Usa los parámetros de un modelo ya ajustado para calcular otra respuesta (p. ej. G′, G″ a partir de creep)."""
    if q.target not in ("saos", "relaxation", "creep"):
        raise HTTPException(400, "Respuesta no válida.")
    if not (-12 <= q.x_min < q.x_max <= 12) or not (10 <= q.n <= 1000):
        raise HTTPException(400, "Revisa el rango: mínimo < máximo, entre 10^-12 y 10^12, con 10 a 1000 puntos.")
    x = np.logspace(q.x_min, q.x_max, q.n)
    try:
        ys = predict_response(q.model, q.params, q.target, x)
    except Exception as e:
        raise HTTPException(400, str(e))
    clean = lambda v: [float(a) if np.isfinite(a) and a > 0 else None for a in v]
    return {"x": x.tolist(), "ys": [clean(y) for y in ys], "target": q.target, "model": q.model}


@app.get("/")
def index():
    return FileResponse(FRONT / "index.html")


app.mount("/static", StaticFiles(directory=FRONT), name="static")
