"""Pruebas de regresión con datos sintéticos de solución conocida.

Correr desde la carpeta backend:
    pip install pytest httpx
    python -m pytest tests -v

Cada prueba fija un caso que ya falló alguna vez o una propiedad teórica que la app debe cumplir.
"""
import json
import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import engine  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)


def _csv(**cols):
    names = list(cols)
    n = len(cols[names[0]])
    rows = [",".join(repr(float(cols[k][i])) for k in names) for i in range(n)]
    return (",".join(names) + "\n" + "\n".join(rows)).encode()


def _fit(test, t, y, model):
    r = client.post("/api/fit", files={"file": ("s.csv", _csv(t=t, y=y))},
                    data={"test": test, "x_col": "t", "y_col": "y", "models": json.dumps([model]), "trim": "0"})
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    assert "error" not in res, res
    return res


def _tensile(e, s, **units):
    r = client.post("/api/tensile", files={"file": ("s.csv", _csv(e=e, s=s))},
                    data={"x_col": "e", "y_col": "s", "secant": "1", **units})
    return r


# ------------------------------------------------------------------ ajuste de modelos
@pytest.mark.parametrize("t", [np.linspace(0, 50, 101), np.linspace(1, 60, 60), np.logspace(-2, 3, 100)])
def test_maxwell_relajacion_recupera_parametros(t):
    # falló con tiempos espaciados linealmente: G_s = 62 872, eta_s = 7 558, R² = -0.49
    res = _fit("relaxation", t, 1000 * np.exp(-t / 10), "Maxwell")
    assert res["params"]["G_s"] == pytest.approx(1000, rel=1e-3)
    assert res["params"]["eta_s"] == pytest.approx(10000, rel=1e-3)
    assert res["r2"] > 0.9999


def test_maxwell_creep_recupera_parametros():
    t = np.linspace(0, 50, 101)
    res = _fit("creep", t, 1 / 1000 + t / 10000, "Maxwell")
    assert res["params"]["G_s"] == pytest.approx(1000, rel=1e-3)
    assert res["params"]["eta_s"] == pytest.approx(10000, rel=1e-3)


@pytest.mark.parametrize("test", ["relaxation", "creep"])
def test_springpot_recupera_parametros(test):
    t = np.logspace(-1, 3, 80)
    y = engine.model_eval(test, "SpringPot", {"V": 50.0, "alpha": 0.2}, t)
    res = _fit(test, t, y, "SpringPot")
    assert res["params"]["alpha"] == pytest.approx(0.2, rel=1e-3)
    assert res["params"]["V"] == pytest.approx(50.0, rel=1e-3)


def test_ajuste_reproducible():
    t = np.logspace(-1, 3, 80)
    y = engine.model_eval("relaxation", "FractionalMaxwellGel", {"G_s": 20.0, "V": 100.0, "alpha": 0.25}, t)
    a = _fit("relaxation", t, y, "FractionalMaxwellGel")
    b = _fit("relaxation", t, y, "FractionalMaxwellGel")
    assert a["params"] == b["params"]
    assert a["r2_log"] > 0.999


# ------------------------------------------------------------------ teoría de los modelos
def _interconversion(model, p, t):
    g = lambda u: engine.model_eval("relaxation", model, p, [u])[0]
    j = lambda u: engine.model_eval("creep", model, p, [u])[0]
    total, a = 0.0, 0.0
    for b in [t * q for q in (1e-6, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99)] + [t]:
        total += quad(lambda u: g(u) * j(t - u), a, b, limit=200)[0]
        a = b
    return total / t


@pytest.mark.parametrize("model,p", [
    ("Maxwell", {"G_s": 17.0, "eta_s": 2000.0}),
    ("FractionalMaxwellGel", {"G_s": 17.77, "V": 173.9, "alpha": 0.2554}),
    ("FractionalMaxwell", {"G": 17.48, "V": 434.5, "alpha": 0.2872, "beta": 0.033}),
    ("Zener", {"G_p": 8.0, "G_s": 11.0, "eta_s": 300.0}),
    # pyRheo 1.0.3 calcula mal este creep (hasta 20 % de error); la app usa la expresión corregida
    ("FractionalZenerSolidS", {"G_p": 8.0, "G_s": 11.0, "V": 60.0, "alpha": 0.3}),
])
def test_creep_y_relajacion_son_interconvertibles(model, p):
    # identidad de la viscoelasticidad lineal: integral_0^t G(t-s) J(s) ds = t
    for t in (1.0, 100.0):
        assert _interconversion(model, p, t) == pytest.approx(1.0, abs=2e-3)


def test_mittag_leffler_exacta():
    import mpmath as mp
    from pyRheo.mittag_leffler.ml_garrappa import E_alpha_beta
    mp.mp.dps = 40
    for a, b, z in [(0.3, 1, -2.0), (0.9, 0.9, -5.0), (0.9, 1, -10.0)]:
        ref = float(mp.nsum(lambda k: mp.mpf(z) ** k * mp.rgamma(a * k + b), [0, mp.inf]))
        val = np.real(complex(np.asarray(E_alpha_beta(np.array([z]), a, b)).ravel()[0]))
        assert val == pytest.approx(ref, rel=1e-8)


# ------------------------------------------------------------------ tensión (ASTM D638)
def test_tension_recta_sin_ruptura():
    # antes reportaba "ruptura" en el último punto aunque la curva nunca cae
    e = np.linspace(0, 10, 100)
    r = _tensile(e, e / 100 * 1.0)  # módulo 1 MPa, deformación en %
    assert r.status_code == 200, r.text
    x = r.json()["results"][0]
    assert x["modulo"]["tipo"] == "elasticidad"
    assert x["modulo"]["valor"] == pytest.approx(1.0, rel=1e-6)
    assert x["ruptura"]["detectada"] is False
    assert x["ruptura"]["esfuerzo"] is None and x["tenacidad_MJ_m3"] is None
    assert any(c["clausula"] == "10.6" for c in r.json()["verificaciones"])


def test_tension_en_pa_y_fraccion():
    # antes: "index 2499 is out of bounds for axis 0 with size 100"
    e = np.linspace(0, 0.10, 100)
    r = _tensile(e, e * 1e6, strain_unit="frac", stress_unit="Pa")
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["modulo"]["valor"] == pytest.approx(1.0, rel=1e-6)


def test_tension_unidades_mal_elegidas_no_truena():
    e = np.linspace(0, 0.10, 100)  # fracción declarada como %
    r = _tensile(e, e * 1e6)
    assert r.status_code in (200, 400)
    if r.status_code == 200:
        assert r.json()["verificaciones"][0]["clausula"] == "Unidades"


def test_tension_con_ruptura():
    e = np.linspace(0, 50, 501)
    s = np.where(e <= 40, 2.0 * e / 100, 0.0)  # recta y caída a 0 después de 40 %
    r = _tensile(e, s)
    x = r.json()["results"][0]
    assert x["ruptura"]["detectada"] is True
    assert x["ruptura"]["deformacion"] == pytest.approx(40.0, abs=0.2)
    assert x["tenacidad_MJ_m3"] == pytest.approx(0.5 * 0.8 * 0.40, rel=1e-2)  # área del triángulo: 1/2 σ ε
