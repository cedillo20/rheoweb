"""Motor de ajuste.

Dos rutas:
  1) pyRheo real (si se puede importar). Ver `_fit_pyrheo` y la sección VERIFICAR.
  2) Respaldo con SciPy (modelos clásicos simples) para probar la página sin pyRheo.
"""
import numpy as np
from scipy.optimize import least_squares

IMPORT_ERROR=""
NUM_GUESSES = 16  # pyRheo usa 64 por defecto; 16 mantiene el tiempo de respuesta razonable en Render

try:  # --- VERIFICAR contra el repo: github.com/mirandi1/pyRheo ---
    from pyRheo.creep_model import CreepModel
    from pyRheo.relaxation_model import RelaxationModel
    from pyRheo.oscillation_model import SAOSModel
    from pyRheo.rotation_model import SteadyShearModel
    PYRHEO = True
except Exception as e:  # noqa
    PYRHEO = False
    IMPORT_ERROR= repr(e)


# Modelos de pyRheo (Tabla 2 del paper) por tipo de ensayo
PYRHEO_MODELS = {
    "creep": ["auto", "Maxwell", "SpringPot", "FractionalMaxwellGel", "FractionalMaxwellLiquid",
              "FractionalMaxwell", "FractionalKelvinVoigt", "FractionalZener"],
    "relaxation": ["auto", "Maxwell", "SpringPot", "FractionalMaxwellGel", "FractionalMaxwellLiquid",
                   "FractionalMaxwell", "FractionalKelvinVoigt", "FractionalZenerSolidS"],
    "saos": ["auto", "Maxwell", "SpringPot", "FractionalMaxwellGel", "FractionalMaxwellLiquid",
             "FractionalMaxwell", "FractionalKelvinVoigt", "FractionalZener"],
    "steady": ["HerschelBulkley", "Bingham", "PowerLaw", "CarreauYasuda", "Cross", "Casson"],
}

# ---------- Respaldo SciPy ----------
def _mx_relax(t, G, tau): return G * np.exp(-t / tau)
def _sp_relax(t, V, a): return V * t ** (-a)
def _mx_creep(t, G, eta): return 1 / G + t / eta
def _sp_creep(t, V, a): return t ** a / V
def _mx_saos(w, G, tau):
    x = (w * tau) ** 2
    return np.concatenate([G * x / (1 + x), G * np.sqrt(x) / (1 + x)])
def _sp_saos(w, V, a):
    m = V * w ** a
    return np.concatenate([m * np.cos(np.pi * a / 2), m * np.sin(np.pi * a / 2)])
def _power(g, K, n): return K * g ** (n - 1)
def _cross(g, e0, einf, K, m): return einf + (e0 - einf) / (1 + (K * g) ** m)

# nombre -> (función, nombres de parámetros, límites inferiores, superiores)
FALLBACK = {
    "relaxation": {"Maxwell": (_mx_relax, ["G", "tau"], [1e-6, 1e-6], [1e12, 1e6]),
                   "SpringPot": (_sp_relax, ["V", "alpha"], [1e-6, 0.0], [1e12, 1.0])},
    "creep": {"Maxwell": (_mx_creep, ["G", "eta"], [1e-6, 1e-6], [1e12, 1e14]),
              "SpringPot": (_sp_creep, ["V", "alpha"], [1e-6, 0.0], [1e12, 1.0])},
    "saos": {"Maxwell": (_mx_saos, ["G", "tau"], [1e-6, 1e-6], [1e12, 1e6]),
             "SpringPot": (_sp_saos, ["V", "alpha"], [1e-6, 0.0], [1e12, 1.0])},
    "steady": {"PowerLaw": (_power, ["K", "n"], [1e-9, 0.0], [1e9, 2.0]),
               "Cross": (_cross, ["eta0", "eta_inf", "K", "m"], [1e-6, 0.0, 1e-6, 0.0], [1e9, 1e6, 1e6, 3.0])},
}


def available_models(test):
    return PYRHEO_MODELS[test] if PYRHEO else list(FALLBACK[test])


def _fit_scipy(test, model, x, y, restarts=10):
    if model not in FALLBACK[test]:
        raise ValueError(f"Modelo '{model}' no disponible en el modo de respaldo.")
    f, names, lo, hi = FALLBACK[test][model]
    lo, hi = np.array(lo, float), np.array(hi, float)
    resid = lambda p: (y - f(x, *p)) / y  # residual ponderado, igual que RSS_w del paper
    rng = np.random.default_rng(0)
    best = None
    for _ in range(restarts):
        # arranque aleatorio: log-uniforme si el rango es grande, uniforme si no
        p0 = np.where(hi / np.maximum(lo, 1e-12) > 1e3,
                      np.exp(rng.uniform(np.log(np.maximum(lo, 1e-12)), np.log(hi))),
                      rng.uniform(lo, hi))
        try:
            r = least_squares(resid, p0, bounds=(lo, hi))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("El ajuste no convergió.")
    return dict(zip(names, best.x)), (lambda xx: f(xx, *best.x))


def _fit_pyrheo(test, model, x, y):
    """VERIFICAR: nombres de clases, argumentos y atributos según el repo/documentación de pyRheo."""
    cls = {"creep": CreepModel, "relaxation": RelaxationModel,
           "saos": SAOSModel, "steady": SteadyShearModel}[test]
    # Verificado contra el codigo fuente de pyRheo: el argumento es `initial_guesses` (plural);
    # SAOS y SteadyShear NO aceptan `mittag_leffler_type`; el algoritmo por defecto
    # de SAOS/Steady es Nelder-Mead y el de creep/relajacion es Powell.
    kw = dict(model=model, initial_guesses="random", num_initial_guesses=NUM_GUESSES)
    if test in ("creep", "relaxation"):
        kw.update(minimization_algorithm="Powell", mittag_leffler_type="Pade32")
    m = cls(**kw)
    if test == "saos":
        m.fit(x, y[:len(x)], y[len(x):])
        pred = lambda xx: np.concatenate(m.predict(xx))
    else:
        m.fit(x, y)
        pred = lambda xx: np.asarray(m.predict(xx))
    if getattr(m, "params_", None) is None:
        raise RuntimeError("pyRheo no encontro una solucion para este modelo.")
    # get_parameters() agrega "Cost" y "Cost Metric" (texto): se descartan.
    params = {k: v for k, v in m.get_parameters().items() if k not in ("Cost", "Cost Metric")}
    return params, pred, m.model  # m.model queda con el modelo elegido si se uso "auto"


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def fit_settings(test):
    """Configuracion con la que se ajusta, para incluirla en el JSON descargable (transparencia y replicabilidad)."""
    if not PYRHEO:
        return {"motor": "SciPy least_squares (modo de respaldo, sin pyRheo)", "funcion_de_costo": "residual ponderado (y - f)/y",
                "arranques_aleatorios": 10, "semilla": 0}
    try:
        from importlib.metadata import version
        ver = version("pyRheo")
    except Exception:  # noqa
        ver = "desconocida"
    s = {"motor": "pyRheo", "version_pyrheo": ver, "repositorio": "https://github.com/mirandi1/pyRheo",
         "initial_guesses": "random", "num_initial_guesses": NUM_GUESSES, "cost_function": "RSS (ponderado por y)",
         "minimization_algorithm": "Powell" if test in ("creep", "relaxation") else "Nelder-Mead",
         "semilla": "no fijada: los parametros pueden variar ligeramente entre ejecuciones",
         "limite_por_modelo_s": MODEL_TIMEOUT}
    if test in ("creep", "relaxation"):
        s["mittag_leffler_type"] = "Pade32"
    return s


def predict_response(model, params, target, x):
    """Evalua un modelo ya ajustado en otro tipo de respuesta.
    pyRheo usa los mismos nombres de modelo y de parametros en creep, relajacion y oscilacion,
    asi que los parametros de un ajuste de creep sirven tal cual para calcular G'(w), G''(w) o G(t).
    target: "saos" -> (G', G'') ; "relaxation" -> (G(t),) ; "creep" -> (J(t),)"""
    if not PYRHEO:
        raise ValueError("La prediccion necesita pyRheo instalado en el servidor.")
    import importlib
    mod = importlib.import_module({"saos": "pyRheo.oscillation_model", "relaxation": "pyRheo.relaxation_model",
                                   "creep": "pyRheo.creep_model"}[target])
    if model not in mod.MODEL_FUNCS:
        raise ValueError(f"Modelo '{model}' no disponible para esta respuesta.")
    names = mod.MODEL_PARAMS[model]
    missing = [n for n in names if n not in params]
    if missing:
        raise ValueError("Faltan parametros: " + ", ".join(missing))
    with np.errstate(all="ignore"):
        y = np.asarray(mod.MODEL_FUNCS[model](*[float(params[n]) for n in names], x), float)
    if target == "saos":
        half = len(y) // 2
        return y[:half], y[half:]
    return (y,)


def _fit_one(test, name, x, y, xd):
    """Ajusta un modelo y devuelve el dict de resultado (solo tipos simples, para poder cruzar procesos)."""
    if PYRHEO:
        params, pred, chosen = _fit_pyrheo(test, name, x, y)
    else:
        params, pred = _fit_scipy(test, name, x, y)
        chosen = name
    yhat, yd = pred(x), pred(xd)
    split = lambda v: [v[:len(v) // 2], v[len(v) // 2:]] if test == "saos" else [v]
    return {
        "requested": name, "model": chosen,
        "params": {k: float(v) for k, v in params.items()},
        "r2": _r2(y, yhat), "r2_log": _r2(np.log10(y), np.log10(np.abs(yhat) + 1e-300)),
        "rss_w": float(np.sum(((y - yhat) / y) ** 2)),
        "curve_x": xd.tolist(), "curve_y": [c.tolist() for c in split(yd)],
    }


def _worker(conn, test, name, x, y, xd):
    import contextlib, io, warnings
    warnings.filterwarnings("ignore")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            conn.send(_fit_one(test, name, x, y, xd))
    except Exception as e:  # noqa
        conn.send({"requested": name, "error": str(e)})
    finally:
        conn.close()


MODEL_TIMEOUT = 30   # segundos por modelo; los de 5-6 parametros pueden no terminar nunca
TOTAL_BUDGET = 80    # segundos por solicitud (Render corta las peticiones largas)


def run_fit(test, x, ys, models):
    """x: array; ys: lista de arrays (1 para creep/relax/steady, 2 para SAOS: G', G'').
    Cada modelo corre en su propio proceso con limite de tiempo: si uno no termina se corta y se avisa."""
    import multiprocessing as mp
    import time
    y = np.concatenate(ys)
    xd = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    out, t_start = [], time.time()
    for name in models:
        left = TOTAL_BUDGET - (time.time() - t_start)
        if left < 3:
            out.append({"requested": name, "error": "Omitido: se agotó el tiempo total de la solicitud. Ajusta menos modelos a la vez."})
            continue
        limit = min(MODEL_TIMEOUT, left)
        parent, child = mp.Pipe(duplex=False)
        p = mp.Process(target=_worker, args=(child, test, name, x, y, xd), daemon=True)
        p.start()
        child.close()
        res = None
        if parent.poll(limit):
            try:
                res = parent.recv()
            except EOFError:
                res = {"requested": name, "error": "El ajuste terminó sin resultado (posible falta de memoria)."}
        else:
            res = {"requested": name, "error": f"Tardó más de {int(limit)} s y se canceló. Prueba con un modelo con menos parámetros."}
        if p.is_alive():
            p.terminate()
        p.join(2)
        out.append(res)
    out.sort(key=lambda r: r.get("rss_w", float("inf")))
    return out
