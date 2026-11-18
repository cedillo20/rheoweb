"""Motor de ajuste.

Dos rutas:
  1) pyRheo real (si se puede importar). Ver `_fit_pyrheo` y la sección VERIFICAR.
  2) Respaldo con SciPy (modelos clásicos simples) para probar la página sin pyRheo.
"""
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import gamma as _gamma

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
def _sp_relax(t, V, a): return V * t ** (-a) / _gamma(1 - a)
def _mx_creep(t, G, eta): return 1 / G + t / eta
def _sp_creep(t, V, a): return t ** a / (V * _gamma(1 + a))
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
                   "SpringPot": (_sp_relax, ["V", "alpha"], [1e-6, 0.0], [1e12, 0.99])},
    "creep": {"Maxwell": (_mx_creep, ["G", "eta"], [1e-6, 1e-6], [1e12, 1e14]),
              "SpringPot": (_sp_creep, ["V", "alpha"], [1e-6, 0.0], [1e12, 0.99])},
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


ML_FIT = "Pade63"     # aproximación rápida para optimizar (la Padé 3/2 se degrada con alpha > 0.3)
ML_EXACT = "Garrappa"  # método de Garrappa: error < 1e-6 % frente a la serie de alta precisión
ML_TOL = 0.005         # si la Padé difiere más de 0.5 % del método exacto, se repule el ajuste con el exacto


def _module(test):
    import importlib
    return importlib.import_module({"saos": "pyRheo.oscillation_model", "relaxation": "pyRheo.relaxation_model",
                                    "creep": "pyRheo.creep_model", "steady": "pyRheo.rotation_model"}[test])


def _fzss_creep(p, t):
    """Creep del Zener fraccionario sólido (resorte G_p en paralelo con Maxwell fraccionario gel G_s, V, alpha).
    pyRheo 1.0.3 usa tau = (V/G_s)^(1/alpha), que es el tiempo de RELAJACIÓN; en creep el tiempo de retardo es
    tau_r = tau * ((G_p + G_s)/G_p)^(1/alpha) (con alpha = 1 se recupera el Zener clásico). Derivado por
    transformada de Laplace de la relajación y verificado con la identidad integral_0^t G(t-s) J(s) ds = t."""
    from pyRheo.mittag_leffler.ml_garrappa import E_alpha_beta
    Gp, Gs, V, a = (float(p[k]) for k in ("G_p", "G_s", "V", "alpha"))
    tau_r = (V / Gs) ** (1 / a) * ((Gp + Gs) / Gp) ** (1 / a)
    t = np.asarray(t, float)
    e = np.real(np.asarray(E_alpha_beta(-np.power(t / tau_r, a), a, 1), complex))
    return 1 / (Gp + Gs) + Gs / (Gp * (Gp + Gs)) * (1 - e)


def model_eval(test, model, params, x, ml=ML_EXACT):
    """Evalúa un modelo de pyRheo con parámetros dados. Para SAOS devuelve G' y G'' concatenados."""
    if test == "creep" and model == "FractionalZenerSolidS":
        return _fzss_creep(params, x)
    mod = _module(test)
    if model not in mod.MODEL_FUNCS:
        raise ValueError(f"Modelo '{model}' no disponible para esta respuesta.")
    fn, names = mod.MODEL_FUNCS[model], mod.MODEL_PARAMS[model]
    missing = [n for n in names if n not in params]
    if missing:
        raise ValueError("Faltan parametros: " + ", ".join(missing))
    args = [float(params[n]) for n in names]
    x = np.asarray(x, float)
    uses_ml = "mittag_leffler_type" in fn.__code__.co_varnames

    def run(kind):
        with np.errstate(all="ignore"):
            y = fn(*args, x, mittag_leffler_type=kind) if uses_ml else fn(*args, x)
        return np.real(np.asarray(y, complex)).astype(float)

    try:
        y = run(ml)
        if uses_ml and not np.all(np.isfinite(y)):
            raise ValueError("resultado no finito")
    except Exception:
        if not uses_ml or ml != ML_EXACT:
            raise
        # El algoritmo de Garrappa falla cuando el orden de la Mittag-Leffler es casi cero (p. ej. alpha - beta ~ 0
        # en Kelvin-Voigt fraccionario). En esa región la Padé 6/3 tiene error < 1e-8 %.
        y = run(ML_FIT)
    return y


def _polish(test, model, params, x, y, maxfev=250):
    """Reajuste local con la Mittag-Leffler exacta, partiendo de los parámetros obtenidos con la Padé."""
    names = list(params)
    frac = [n in ("alpha", "beta", "kappa") for n in names]
    eps = 1e-9

    def to_th(p):
        out = []
        for n, f in zip(names, frac):
            v = float(p[n])
            if f:
                v = min(max(v, eps), 1 - eps)
                out.append(np.log(v / (1 - v)))
            else:
                out.append(np.log(max(v, 1e-300)))
        return np.array(out)

    def from_th(th):
        return {n: (float(1 / (1 + np.exp(-v))) if f else float(np.exp(v))) for n, v, f in zip(names, th, frac)}

    def cost(th):
        r = (y - model_eval(test, model, from_th(th), x)) / y
        c = float(np.sum(r ** 2))
        return c if np.isfinite(c) else 1e300

    th0 = to_th(params)
    c0 = cost(th0)
    r = minimize(cost, th0, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-7, "fatol": 1e-12})
    return (from_th(r.x), True) if r.fun < c0 else (dict(params), False)


def _analytic_start(test, model, x, y):
    """Valores iniciales exactos por linealización para los modelos que la admiten.
    Maxwell relajación: ln G = ln G_s - t/tau ; Maxwell creep: J = 1/G_s + t/eta_s ;
    Springpot: ln G = ln(V/Gamma(1-a)) - a ln t (relajación) y ln J = -ln(V Gamma(1+a)) + a ln t (creep).
    pyRheo sortea los valores iniciales en un rango derivado de los datos y, con tiempos espaciados linealmente,
    no encontraba la solución de Maxwell en relajación (R² = -0.49 con datos exactos)."""
    try:
        if model == "Maxwell" and test == "relaxation":
            k, b = np.polyfit(x, np.log(y), 1)
            return {"G_s": float(np.exp(b)), "eta_s": float(np.exp(b) * (-1 / k))} if k < 0 else None
        if model == "Maxwell" and test == "creep":
            k, b = np.polyfit(x, y, 1)
            return {"G_s": float(1 / b), "eta_s": float(1 / k)} if k > 0 and b > 0 else None
        if model == "SpringPot" and test in ("creep", "relaxation"):
            k, b = np.polyfit(np.log(x), np.log(y), 1)
            a = -k if test == "relaxation" else k
            if not 0 < a < 1:
                return None
            V = np.exp(b) * _gamma(1 - a) if test == "relaxation" else 1 / (np.exp(b) * _gamma(1 + a))
            return {"V": float(V), "alpha": float(a)}
    except Exception:  # noqa
        return None
    return None


def _fit_pyrheo(test, model, x, y, n_guesses=None):
    """VERIFICAR: nombres de clases, argumentos y atributos según el repo/documentación de pyRheo."""
    cls = {"creep": CreepModel, "relaxation": RelaxationModel,
           "saos": SAOSModel, "steady": SteadyShearModel}[test]
    # Verificado contra el codigo fuente de pyRheo: el argumento es `initial_guesses` (plural);
    # SAOS y SteadyShear NO aceptan `mittag_leffler_type`; el algoritmo por defecto
    # de SAOS/Steady es Nelder-Mead y el de creep/relajacion es Powell.
    kw = dict(model=model, initial_guesses="random", num_initial_guesses=n_guesses or NUM_GUESSES)
    if test in ("creep", "relaxation"):
        kw.update(minimization_algorithm="Powell", mittag_leffler_type=ML_FIT)
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
         "semillas": list(SEEDS),
         "reintentos": f"primer intento: semilla {ATTEMPTS[0][0]} de numpy.random con {ATTEMPTS[0][1]} valores iniciales; si R² log < "
                       f"{RETRY_R2_LOG} se repite con " + " y ".join(f"semilla {a} ({n} valores iniciales)" for a, n in ATTEMPTS[1:]) +
                       " y se conserva el intento de menor RSS_w. La semilla de cada modelo se reporta en sus resultados: con el "
                       "mismo archivo y configuración el ajuste es idéntico. El optimizador es local: un R² log bajo puede ser un "
                       "mínimo local y no una limitación del modelo; los residuos ayudan a distinguirlo.",
         "limite_por_modelo_s": MODEL_TIMEOUT}
    if test in ("creep", "relaxation"):
        s["mittag_leffler"] = (f"la optimización usa la aproximación {ML_FIT}; curvas, métricas y residuos se calculan con el método "
                               f"exacto ({ML_EXACT}). Si ambos difieren más de {ML_TOL * 100:g} % en los datos, los parámetros se "
                               "repulen con el método exacto (Nelder-Mead, máx. 250 evaluaciones).")
    s["inicio_analitico"] = ("Maxwell y Springpot en creep y relajación también se ajustan desde valores iniciales exactos por "
                             "linealización; se conserva el resultado de menor RSS_w")
    s["correcciones_pyrheo"] = ["creep de FractionalZenerSolidS con el tiempo de retardo correcto (pyRheo 1.0.3 usa el de relajación)",
                                "las curvas se evalúan con la misma función con la que se reportan las métricas"]
    return s


def predict_response(model, params, target, x):
    """Evalua un modelo ya ajustado en otro tipo de respuesta (Mittag-Leffler exacta).
    pyRheo usa los mismos nombres de modelo y de parametros en creep, relajacion y oscilacion; se verificó
    numéricamente que sus funciones son transformadas exactas entre sí (salvo el creep de Zener sólido S,
    que aquí se calcula con la expresión corregida).
    target: "saos" -> (G', G'') ; "relaxation" -> (G(t),) ; "creep" -> (J(t),)"""
    if not PYRHEO:
        raise ValueError("La prediccion necesita pyRheo instalado en el servidor.")
    y = model_eval(target, model, params, x)
    if target == "saos":
        half = len(y) // 2
        return y[:half], y[half:]
    return (y,)


def _fit_one(test, name, x, y, xd, n_guesses=None):
    """Ajusta un modelo y devuelve el dict de resultado (solo tipos simples, para poder cruzar procesos)."""
    ml, start = None, "aleatorio (pyRheo)"
    if PYRHEO:
        params, _, chosen = _fit_pyrheo(test, name, x, y, n_guesses)
        params = {k: float(v) for k, v in params.items()}
        # pyRheo predice con la aproximación por defecto de cada función (no la usada al ajustar): se evalúa aquí
        yhat = model_eval(test, chosen, params, x)
        if test in ("creep", "relaxation"):
            y_pade = model_eval(test, chosen, params, x, ml=ML_FIT)
            dev = float(np.nanmax(np.abs(y_pade / yhat - 1))) if np.all(np.isfinite(yhat)) else float("inf")
            ml = {"optimizacion": ML_FIT, "curvas_y_metricas": ML_EXACT, "desviacion_pade_pct": dev * 100, "repulido": False}
            if dev > ML_TOL:
                params, ml["repulido"] = _polish(test, chosen, params, x, y)
                yhat = model_eval(test, chosen, params, x)
        # inicio analítico: si da menor RSS_w que la búsqueda aleatoria de pyRheo, se usa ese resultado
        p0 = _analytic_start(test, chosen, x, y)
        if p0:
            cost = lambda p: float(np.sum(((y - model_eval(test, chosen, p, x)) / y) ** 2))
            cand, _ = _polish(test, chosen, p0, x, y, maxfev=400)
            c_cand, c_cur = cost(cand), cost(params)
            if np.isfinite(c_cand) and (not np.isfinite(c_cur) or c_cand < c_cur):
                params, start = cand, "analítico (linealización)"
                yhat = model_eval(test, chosen, params, x)
        pred = lambda xx: model_eval(test, chosen, params, xx)
    else:
        params, pred = _fit_scipy(test, name, x, y)
        chosen = name
        yhat = pred(x)
    yd = pred(xd)
    split = lambda v: [v[:len(v) // 2], v[len(v) // 2:]] if test == "saos" else [v]
    return {
        "requested": name, "model": chosen,
        "params": {k: float(v) for k, v in params.items()},
        "r2": _r2(y, yhat), "r2_log": _r2(np.log10(y), np.log10(np.abs(yhat) + 1e-300)),
        "rss_w": float(np.sum(((y - yhat) / y) ** 2)),
        "curve_x": xd.tolist(), "curve_y": [c.tolist() for c in split(yd)],
        # residuo relativo en cada punto medido, en %: (dato - modelo) / dato
        "resid": [[float(v) if np.isfinite(v) else None for v in c] for c in split((y - yhat) / y * 100.0)],
        "mittag_leffler": ml,
        "inicio": start,
    }


def _worker(conn, test, name, x, y, xd, seed, n_guesses):
    import contextlib, io, warnings
    warnings.filterwarnings("ignore")
    try:
        # pyRheo genera sus valores iniciales con el generador global de NumPy: fijarlo hace el ajuste reproducible
        np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            conn.send(_fit_one(test, name, x, y, xd, n_guesses))
    except Exception as e:  # noqa
        conn.send({"requested": name, "error": str(e)})
    finally:
        conn.close()


MODEL_TIMEOUT = 30   # segundos por intento; los modelos de 5-6 parametros pueden no terminar nunca
TOTAL_BUDGET = 80    # segundos por solicitud (Render corta las peticiones largas)
# Intentos (semilla de numpy.random, número de valores iniciales aleatorios): el primero siempre; los demás solo
# si el ajuste sale pobre. El optimizador de pyRheo es local: con más valores iniciales es más probable hallar el
# mínimo global (en relajación, Zener sólido S pasó de R² log 0.82 con 16 a 0.995 con 64).
ATTEMPTS = ((0, NUM_GUESSES), (1, 64), (2, 64))
SEEDS = tuple(a[0] for a in ATTEMPTS)
RETRY_R2_LOG = 0.95  # si R2 log < este valor se reintenta con la siguiente semilla y se conserva el de menor RSS_w


def _attempt(test, name, x, y, xd, seed, n_guesses, limit):
    import multiprocessing as mp
    parent, child = mp.Pipe(duplex=False)
    p = mp.Process(target=_worker, args=(child, test, name, x, y, xd, seed, n_guesses), daemon=True)
    p.start()
    child.close()
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
    return res


def run_fit(test, x, ys, models):
    """x: array; ys: lista de arrays (1 para creep/relax/steady, 2 para SAOS: G', G'').
    Cada intento corre en su propio proceso con limite de tiempo y semilla fija (resultado reproducible).
    Si el ajuste sale pobre (R2 log < RETRY_R2_LOG) se repite con las semillas siguientes y se conserva el mejor."""
    import time
    y = np.concatenate(ys)
    xd = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    out, t_start = [], time.time()
    for name in models:
        best, tried = None, []
        for seed, n_guesses in ATTEMPTS:
            left = TOTAL_BUDGET - (time.time() - t_start)
            if left < 3:
                if best is None:
                    best = {"requested": name, "error": "Omitido: se agotó el tiempo total de la solicitud. Ajusta menos modelos a la vez."}
                break
            res = _attempt(test, name, x, y, xd, seed, n_guesses, min(MODEL_TIMEOUT, left))
            tried.append(seed)
            if "error" in res:
                if best is None:
                    best = res
                if "Tardó" in res["error"] or not PYRHEO:
                    break          # si un intento no termina a tiempo, los demás tampoco lo harán
                continue
            if best is None or "error" in best or res["rss_w"] < best["rss_w"]:
                best = {**res, "semilla": seed, "valores_iniciales": n_guesses}
            if not PYRHEO or res["r2_log"] >= RETRY_R2_LOG:
                break
        best["intentos"] = len(tried)
        best["semillas_probadas"] = tried
        out.append(best)
    out.sort(key=lambda r: r.get("rss_w", float("inf")))
    return out
