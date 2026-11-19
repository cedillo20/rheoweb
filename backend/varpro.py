"""Ajuste por proyección de variables (variable projection).

En casi todos los modelos, la respuesta es LINEAL en los coeficientes con dimensiones (módulos, viscosidades,
cuasi-propiedades de los springpots) una vez fijados los órdenes fraccionarios y, si hace falta, un tiempo
característico:

    y(x) = sum_j c_j * phi_j(x; theta)        con c_j >= 0

Para cada theta, los c_j óptimos se obtienen EXACTAMENTE con mínimos cuadrados no negativos (NNLS) sobre el
residuo relativo ponderado; el optimizador solo busca en theta (órdenes en (0, 1) y log10 del tiempo).

Por qué: pyRheo acota todos los parámetros con dimensiones al rango (min(y)/100, max(y)*100) de la respuesta
(o de 1/J en creep). Ese rango tiene unidades de módulo, pero las viscosidades (MPa·s) y las cuasi-propiedades
(MPa·s^alpha) no; con datos FIME la viscosidad de Maxwell fraccionario líquido en creep quedaba pegada al límite
(eta_s = 1644, error RMS de 312 %) cuando el valor correcto es ~6e5 MPa·s (error 0.079 %).

La pérdida es la de pyRheo: sum w_i ((y_i - yhat_i) / y_i)^2.
Las funciones de Mittag-Leffler se evalúan aquí con la Padé 6/3 (rápida); engine.py verifica después la curva
con el método exacto de Garrappa y repule si difieren.
"""
import itertools

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import gamma

try:
    from pyRheo.mittag_leffler.ml_pade63 import R_alpha_beta_6_3 as _ML
except Exception:  # noqa
    _ML = None


def _ml(z, a, b):
    with np.errstate(all="ignore"):
        return np.real(np.asarray(_ML(np.asarray(z, float), a, b), complex)).astype(float)


def _cplx(v):
    """Columna compleja G* -> [G'; G''] apilados (mismo orden que pyRheo)."""
    return np.concatenate([np.real(v), np.imag(v)])


# Cada modelo: lista de parámetros no lineales y función (x, th) -> (matriz A, función c -> parámetros pyRheo)
# Tipos de parámetro no lineal: "frac" en (0, 1); "frac<alpha" en (0, alpha); "logt" = log10 de un tiempo;
# "pos2" en (0, 2); "pos5" en (0.05, 5).
def _specs():
    S = {}
    # ---------------- creep: J(t)
    S["creep:Maxwell"] = ([], lambda t, p: (np.column_stack([np.ones_like(t), t]),
                                           lambda c: {"G_s": 1 / c[0], "eta_s": 1 / c[1]}))
    S["creep:SpringPot"] = (["alpha:frac"], lambda t, p: (np.column_stack([t ** p["alpha"] / gamma(1 + p["alpha"])]),
                                                          lambda c: {"V": 1 / c[0], "alpha": p["alpha"]}))
    S["creep:FractionalMaxwellGel"] = (["alpha:frac"], lambda t, p: (
        np.column_stack([np.ones_like(t), t ** p["alpha"] / gamma(1 + p["alpha"])]),
        lambda c: {"G_s": 1 / c[0], "V": 1 / c[1], "alpha": p["alpha"]}))
    S["creep:FractionalMaxwellLiquid"] = (["beta:frac"], lambda t, p: (
        np.column_stack([t, t ** p["beta"] / gamma(1 + p["beta"])]),
        lambda c: {"G": 1 / c[1], "eta_s": 1 / c[0], "beta": p["beta"]}))
    S["creep:FractionalMaxwell"] = (["alpha:frac", "beta:frac<alpha"], lambda t, p: (
        np.column_stack([t ** p["alpha"] / gamma(1 + p["alpha"]), t ** p["beta"] / gamma(1 + p["beta"])]),
        lambda c: {"G": 1 / c[1], "V": 1 / c[0], "alpha": p["alpha"], "beta": p["beta"]}))

    def ckv(t, p):  # J = t^a/V * E_{a-b,1+a}(-(t/tau)^(a-b)),  G = V tau^-(a-b)
        a, b, tau = p["alpha"], p["beta"], 10 ** p["logt"]
        col = t ** a * _ml(-(t / tau) ** (a - b), a - b, 1 + a)
        return np.column_stack([col]), lambda c: {"G": (1 / c[0]) * tau ** -(a - b), "V": 1 / c[0], "alpha": a, "beta": b}
    S["creep:FractionalKelvinVoigt"] = (["alpha:frac", "beta:frac<alpha", "logt:logt"], ckv)

    # ---------------- relajación: G(t)
    S["relaxation:Maxwell"] = (["logt:logt"], lambda t, p: (
        np.column_stack([np.exp(-t / 10 ** p["logt"])]),
        lambda c: {"G_s": c[0], "eta_s": c[0] * 10 ** p["logt"]}))
    S["relaxation:SpringPot"] = (["alpha:frac"], lambda t, p: (
        np.column_stack([t ** -p["alpha"] / gamma(1 - p["alpha"])]), lambda c: {"V": c[0], "alpha": p["alpha"]}))
    S["relaxation:FractionalKelvinVoigt"] = (["alpha:frac", "beta:frac<alpha"], lambda t, p: (
        np.column_stack([t ** -p["alpha"] / gamma(1 - p["alpha"]), t ** -p["beta"] / gamma(1 - p["beta"])]),
        lambda c: {"G": c[1], "V": c[0], "alpha": p["alpha"], "beta": p["beta"]}))

    def rfmg(t, p):  # G = G_s E_a(-(t/tau)^a),  V = G_s tau^a
        a, tau = p["alpha"], 10 ** p["logt"]
        return np.column_stack([_ml(-(t / tau) ** a, a, 1)]), lambda c: {"G_s": c[0], "V": c[0] * tau ** a, "alpha": a}
    S["relaxation:FractionalMaxwellGel"] = (["alpha:frac", "logt:logt"], rfmg)

    def rfzss(t, p):  # G = G_p + G_s E_a(-(t/tau)^a),  V = G_s tau^a
        a, tau = p["alpha"], 10 ** p["logt"]
        return (np.column_stack([np.ones_like(t), _ml(-(t / tau) ** a, a, 1)]),
                lambda c: {"G_p": c[0], "G_s": c[1], "V": c[1] * tau ** a, "alpha": a})
    S["relaxation:FractionalZenerSolidS"] = (["alpha:frac", "logt:logt"], rfzss)

    def rfmm(t, p):  # G = Gc (t/tau)^-b E_{a-b,1-b}(-(t/tau)^(a-b)),  Gc = V tau^-a,  G = V tau^-(a-b)
        a, b, tau = p["alpha"], p["beta"], 10 ** p["logt"]
        col = (t / tau) ** -b * _ml(-(t / tau) ** (a - b), a - b, 1 - b)
        return np.column_stack([col]), lambda c: {"G": c[0] * tau ** b, "V": c[0] * tau ** a, "alpha": a, "beta": b}
    S["relaxation:FractionalMaxwell"] = (["alpha:frac", "beta:frac<alpha", "logt:logt"], rfmm)

    def rfml(t, p):  # G = Gc (t/tau)^-b E_{1-b,1-b}(-(t/tau)^(1-b)),  Gc = eta/tau,  G = eta tau^-(1-b)
        b, tau = p["beta"], 10 ** p["logt"]
        col = (t / tau) ** -b * _ml(-(t / tau) ** (1 - b), 1 - b, 1 - b + 1e-16)
        return np.column_stack([col]), lambda c: {"G": c[0] * tau ** b, "eta_s": c[0] * tau, "beta": b}
    S["relaxation:FractionalMaxwellLiquid"] = (["beta:frac", "logt:logt"], rfml)

    # ---------------- SAOS: G*(w) = sum c_j phi_j(i w)
    S["saos:Maxwell"] = (["logt:logt"], lambda w, p: (
        np.column_stack([_cplx(1j * w * 10 ** p["logt"] / (1 + 1j * w * 10 ** p["logt"]))]),
        lambda c: {"G_s": c[0], "eta_s": c[0] * 10 ** p["logt"]}))
    S["saos:SpringPot"] = (["alpha:frac"], lambda w, p: (
        np.column_stack([_cplx((1j * w) ** p["alpha"])]), lambda c: {"V": c[0], "alpha": p["alpha"]}))
    S["saos:FractionalKelvinVoigt"] = (["alpha:frac", "beta:frac<alpha"], lambda w, p: (
        np.column_stack([_cplx((1j * w) ** p["alpha"]), _cplx((1j * w) ** p["beta"])]),
        lambda c: {"G": c[1], "V": c[0], "alpha": p["alpha"], "beta": p["beta"]}))

    def sfmg(w, p):
        a, tau = p["alpha"], 10 ** p["logt"]
        z = (1j * w * tau) ** a
        return np.column_stack([_cplx(z / (1 + z))]), lambda c: {"G_s": c[0], "V": c[0] * tau ** a, "alpha": a}
    S["saos:FractionalMaxwellGel"] = (["alpha:frac", "logt:logt"], sfmg)

    def sfmm(w, p):  # G* = Gc (iwt)^a / (1 + (iwt)^(a-b)),  Gc = V tau^-a
        a, b, tau = p["alpha"], p["beta"], 10 ** p["logt"]
        iwt = 1j * w * tau
        return (np.column_stack([_cplx(iwt ** a / (1 + iwt ** (a - b)))]),
                lambda c: {"G": c[0] * tau ** b, "V": c[0] * tau ** a, "alpha": a, "beta": b})
    S["saos:FractionalMaxwell"] = (["alpha:frac", "beta:frac<alpha", "logt:logt"], sfmm)

    def sfml(w, p):  # G* = Gc (iwt) / (1 + (iwt)^(1-b)),  Gc = eta / tau
        b, tau = p["beta"], 10 ** p["logt"]
        iwt = 1j * w * tau
        return (np.column_stack([_cplx(iwt / (1 + iwt ** (1 - b)))]),
                lambda c: {"G": c[0] * tau ** b, "eta_s": c[0] * tau, "beta": b})
    S["saos:FractionalMaxwellLiquid"] = (["beta:frac", "logt:logt"], sfml)

    # ---------------- flujo estacionario: eta(gamma_dot)
    S["steady:PowerLaw"] = (["n:pos2"], lambda g, p: (np.column_stack([g ** (p["n"] - 1)]), lambda c: {"k": c[0], "n": p["n"]}))
    S["steady:HerschelBulkley"] = (["n:pos2"], lambda g, p: (
        np.column_stack([1 / g, g ** (p["n"] - 1)]), lambda c: {"sigma_y": c[0], "k": c[1], "n": p["n"]}))
    S["steady:Bingham"] = ([], lambda g, p: (np.column_stack([1 / g, np.ones_like(g)]), lambda c: {"sigma_y": c[0], "eta_p": c[1]}))

    def cross(g, p):  # eta = eta_inf + (eta_0 - eta_inf) / (1 + (k g)^n)
        f = 1 / (1 + (10 ** p["logt"] * g) ** p["n"])
        return (np.column_stack([1 - f, f]),
                lambda c: {"eta_inf": c[0], "eta_zero": c[1], "k": 10 ** p["logt"], "n": p["n"]})
    S["steady:Cross"] = (["logt:logt", "n:pos2"], cross)

    def carreau(g, p):  # eta = eta_inf + (eta_0 - eta_inf) (1 + (k g)^a)^((n-1)/a)
        h = (1 + (10 ** p["logt"] * g) ** p["a"]) ** ((p["n"] - 1) / p["a"])
        return (np.column_stack([1 - h, h]),
                lambda c: {"eta_inf": c[0], "eta_zero": c[1], "k": 10 ** p["logt"], "a": p["a"], "n": p["n"]})
    S["steady:CarreauYasuda"] = (["logt:logt", "a:pos5", "n:pos2"], carreau)
    return S


SPECS = _specs()
_FRAC_STARTS = (0.1, 0.3, 0.5, 0.7, 0.9)


def supported(test, model):
    return _ML is not None and f"{test}:{model}" in SPECS


def _decode(kinds, u, logt_range):
    """Vector sin restricciones -> parámetros no lineales dentro de su dominio."""
    sig = lambda v: 1 / (1 + np.exp(-v))
    p = {}
    for (name, kind), v in zip(kinds, u):
        if kind == "frac":
            p[name] = 1e-6 + (1 - 2e-6) * sig(v)
        elif kind == "frac<alpha":
            p[name] = p["alpha"] * (1e-6 + (1 - 2e-6) * sig(v))
        elif kind == "logt":
            lo, hi = logt_range
            p[name] = lo + (hi - lo) * sig(v)
        elif kind == "pos2":
            p[name] = 1e-4 + (2 - 2e-4) * sig(v)
        elif kind == "pos5":
            p[name] = 0.05 + (5 - 0.05) * sig(v)
    return p


def _encode_start(kind, value, logt_range):
    """Valor inicial en el dominio -> vector sin restricciones (inverso de _decode, aproximado para frac<alpha)."""
    logit = lambda q: float(np.log(q / (1 - q)))
    if kind in ("frac", "frac<alpha"):
        return logit(min(max(value, 1e-3), 1 - 1e-3))
    if kind == "logt":
        lo, hi = logt_range
        return logit(min(max((value - lo) / (hi - lo), 1e-3), 1 - 1e-3))
    if kind == "pos2":
        return logit(min(max(value / 2, 1e-3), 1 - 1e-3))
    if kind == "pos5":
        return logit(min(max((value - 0.05) / 4.95, 1e-3), 1 - 1e-3))
    return 0.0


def fit(test, model, x, y, w=None):
    """Devuelve (parámetros pyRheo, pérdida) o None si el modelo no admite proyección de variables."""
    key = f"{test}:{model}"
    if not supported(test, model):
        return None
    kinds = [tuple(s.split(":")) for s in SPECS[key][0]]
    build = SPECS[key][1]
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    sw = np.sqrt(w)
    # rango del tiempo característico: 6 décadas más allá de los datos (en SAOS y flujo, tiempo ~ 1/x). En los modelos
    # con dos órdenes, tau = (V/G)^(1/(alpha-beta)) puede quedar muy lejos de la ventana medida cuando alpha ~ beta.
    lx = np.log10(x)
    logt_range = ((-lx.max() - 6, -lx.min() + 6) if test in ("saos", "steady") else (lx.min() - 6, lx.max() + 6))

    def solve(u):
        p = _decode(kinds, u, logt_range)
        try:
            A, to_params = build(x, p)
            Aw = A / y[:, None] * sw[:, None]
            if not np.all(np.isfinite(Aw)):
                return np.inf, None, None
            c, _ = nnls(Aw, sw)
        except Exception:  # noqa
            return np.inf, None, None
        c = np.maximum(c, 1e-14 * max(c.max(), 1e-300))   # coeficiente nulo -> parámetro muy grande pero finito
        r = (y - A @ c) / y
        loss = float(np.sum(w * r ** 2))
        return (loss if np.isfinite(loss) else np.inf), c, to_params

    if not kinds:
        loss, c, to_params = solve(np.array([]))
        return (to_params(c), loss) if c is not None else None

    # arranques deterministas en una rejilla; se refinan los 3 mejores con Nelder-Mead
    grids = []
    for name, kind in kinds:
        if kind in ("frac", "frac<alpha"):
            grids.append([(kind, v) for v in _FRAC_STARTS])
        elif kind == "logt":
            grids.append([(kind, v) for v in np.linspace(*logt_range, 9)[1:-1]])
        elif kind == "pos2":
            grids.append([(kind, v) for v in (0.3, 0.7, 1.0, 1.4)])
        elif kind == "pos5":
            grids.append([(kind, v) for v in (0.5, 1.0, 2.0)])
    starts = [np.array([_encode_start(k, v, logt_range) for k, v in combo]) for combo in itertools.product(*grids)]
    scored = sorted(((solve(u)[0], i) for i, u in enumerate(starts)), key=lambda z: z[0])
    best = None
    opts = {"maxfev": 400 * len(kinds), "xatol": 1e-9, "fatol": 1e-16}
    for _, i in scored[:3]:
        r = minimize(lambda u: solve(u)[0], starts[i], method="Nelder-Mead", options=opts)
        # Nelder-Mead puede detenerse en un simplex degenerado: se reinicia desde su mejor punto hasta que deja de mejorar
        for _ in range(4):
            r2 = minimize(lambda u: solve(u)[0], r.x, method="Nelder-Mead", options=opts)
            if r2.fun >= r.fun * (1 - 1e-9):
                break
            r = r2
        if best is None or r.fun < best.fun:
            best = r
    loss, c, to_params = solve(best.x)
    if c is None or not np.isfinite(loss):
        return None
    return {k: float(v) for k, v in to_params(c).items()}, loss
