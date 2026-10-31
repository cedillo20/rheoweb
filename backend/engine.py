"""Motor de ajuste.

Dos rutas:
  1) pyRheo real (si se puede importar). Ver `_fit_pyrheo` y la sección VERIFICAR.
  2) Respaldo con SciPy (modelos clásicos simples) para probar la página sin pyRheo.
"""
import numpy as np
from scipy.optimize import least_squares

IMPORT_ERROR=""

try:  # --- VERIFICAR contra el repo: github.com/mirandi1/pyRheo ---
    from pyRheo.creep_model import CreepModel
    from pyRheo.relaxation_model import RelaxationModel
    from pyRheo.oscillation_model import SAOSModel
    from pyRheo.rotation_model import SteadyShearModel
    PYRHEO = True
except Exception as e:  # noqa
    PYRHEO = False
    IMPORT_ERROR= repre(e)


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
    m = cls(model=model, initial_guess="random", minimization_algorithm="Powell",
            mittag_leffler_type="Pade32")
    if test == "saos":
        m.fit(x, y[:len(x)], y[len(x):])
        pred = lambda xx: np.concatenate(m.predict(xx))
    else:
        m.fit(x, y)
        pred = lambda xx: np.asarray(m.predict(xx))
    params = getattr(m, "parameters_", None) or getattr(m, "parameters", {})
    chosen = getattr(m, "model_name", None) or getattr(m, "model", model)
    return dict(params), pred, chosen


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def run_fit(test, x, ys, models):
    """x: array; ys: lista de arrays (1 para creep/relax/steady, 2 para SAOS: G', G'')."""
    y = np.concatenate(ys)
    xd = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    out = []
    for name in models:
        try:
            if PYRHEO:
                params, pred, chosen = _fit_pyrheo(test, name, x, y)
            else:
                params, pred = _fit_scipy(test, name, x, y)
                chosen = name
            yhat, yd = pred(x), pred(xd)
            split = lambda v: [v[:len(v) // 2], v[len(v) // 2:]] if test == "saos" else [v]
            out.append({
                "requested": name, "model": chosen,
                "params": {k: float(v) for k, v in params.items()},
                "r2": _r2(y, yhat), "r2_log": _r2(np.log10(y), np.log10(np.abs(yhat) + 1e-300)),
                "rss_w": float(np.sum(((y - yhat) / y) ** 2)),
                "curve_x": xd.tolist(), "curve_y": [c.tolist() for c in split(yd)],
            })
        except Exception as e:
            out.append({"requested": name, "error": str(e)})
    out.sort(key=lambda r: r.get("rss_w", float("inf")))
    return out
