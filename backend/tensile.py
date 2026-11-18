"""Análisis de tensión según ASTM D638-14.

Cada cálculo cita la cláusula que sigue. Donde la norma describe un procedimiento gráfico sin fijar
un algoritmo (por ejemplo, "la porción lineal inicial"), la regla numérica usada se documenta en
IMPLEMENTACION para que se pueda revisar y reportar.
"""
import numpy as np

NORMA = "ASTM D638-14"

# Reglas numéricas propias de esta implementación (D638 no fija un algoritmo para estos pasos)
IMPLEMENTACION = {
    "region_inicial": "puntos con esfuerzo <= 25 % del esfuerzo máximo",
    "ventana_tangente": "regresión lineal sobre 0.25 % de deformación (mínimo 5 puntos)",
    "porcion_lineal": "tramo contiguo alrededor de la tangente donde la pendiente local se mantiene >= 95 % de E; "
                      "se considera porción lineal si abarca al menos 0.25 % de deformación",
    "cedencia": "primer punto donde el esfuerzo (promedio móvil de 9 puntos) deja de aumentar y no se supera "
                "en el siguiente 1 % de deformación",
    "ruptura": "último punto antes de la mayor caída de esfuerzo después del máximo; si ninguna caída entre puntos "
               "consecutivos supera 10 % del esfuerzo máximo, se considera que el registro no incluye la ruptura y no se "
               "reportan propiedades a la ruptura ni tenacidad",
    "tenacidad": "integral trapezoidal del esfuerzo nominal sobre la deformación (en fracción) desde el cero "
                 "corregido hasta la ruptura",
}

# Tabla 1 de D638: velocidades de ensayo (mm/min) y tolerancia
TABLA1 = [(1, .25, "V"), (5, .25, "I, II, III, IV"), (10, .25, "V"), (50, .10, "I, II, III, IV"),
          (100, .25, "V"), (500, .10, "I, II, III, IV")]
GAGE = {"I, II, III": 50.0, "IV": 25.0, "V": 7.62}       # G de la Fig. 1, mm
GRIPS = {"I": (115, 5), "II": (135, 5), "III": (115, 5)}  # D de la Fig. 1, mm


def _fit(x, y):
    p = np.polyfit(x, y, 1)
    r = y - np.polyval(p, x)
    ss = np.sum((y - y.mean()) ** 2)
    return p[0], p[1], (1 - np.sum(r ** 2) / ss) if ss > 0 else 1.0


def _smooth(y, n=9):
    if len(y) < n:
        return y.copy()
    k = np.ones(n) / n
    pad = n // 2
    return np.convolve(np.pad(y, pad, mode="edge"), k, mode="valid")


def analyze(strain_pct, stress, secant_pct=1.0, time=None, force=None, disp=None):
    """strain_pct: deformación en %, stress: esfuerzo nominal en MPa. time (s), force (N), disp (mm) son opcionales."""
    e = np.asarray(strain_pct, float)
    s = np.asarray(stress, float)
    ok = np.isfinite(e) & np.isfinite(s)
    e, s = e[ok], s[ok]
    t = np.asarray(time, float)[ok] if time is not None else None
    F = np.asarray(force, float)[ok] if force is not None else None
    X = np.asarray(disp, float)[ok] if disp is not None else None
    if len(e) < 20:
        raise ValueError("Se necesitan al menos 20 puntos.")
    imax = int(np.argmax(s))
    smax = float(s[imax])

    # --- Ruptura (10.6, A2.8.1, A2.17.1): momento de la ruptura
    drops = np.diff(s[imax:]) if imax < len(s) - 1 else np.array([0.0])
    jd = int(np.argmin(drops)) if len(drops) else 0
    rotura = bool(len(drops) and -drops[jd] >= 0.10 * smax)
    r = imax + jd if rotura else len(s) - 1

    # --- Tangente de pendiente máxima en la región inicial (11.4, A1.2, A1.3)
    lim = np.where(s[: imax + 1] <= 0.25 * smax)[0]
    n_ini = int(lim[-1]) + 1 if len(lim) else imax + 1
    step = np.median(np.diff(e[: max(n_ini, 3)])) if n_ini > 2 else np.nan
    if n_ini < 5:
        raise ValueError("La región inicial (esfuerzo <= 25 % del máximo) tiene menos de 5 puntos; no se puede trazar la "
                         "tangente del módulo. Revisa la frecuencia de muestreo o las unidades de deformación.")
    w = max(5, int(np.ceil(0.25 / step)) if np.isfinite(step) and step > 0 else 5)
    w = min(w, n_ini)  # la ventana nunca puede ser mayor que la región inicial
    best = None
    for i in range(0, max(1, n_ini - w + 1)):
        seg = slice(i, i + w)
        if len(e[seg]) < 3 or np.ptp(e[seg]) <= 0:
            continue
        m, b, r2 = _fit(e[seg] / 100.0, s[seg])
        if best is None or m > best[0]:
            best = (m, b, r2, i)
    if best is None or best[0] <= 0:
        raise ValueError("No se encontró una región inicial con pendiente positiva.")
    E_t, b_t, r2_t, i_t = best

    # Corrección del pie de curva (11.1, Anexo A1): el cero de deformación se mueve a la intersección de la tangente
    toe = max(0.0, -b_t / E_t * 100.0)   # % de deformación
    ec = e - toe

    # ¿Existe porción lineal? (11.4 frente a 11.5)
    local = np.gradient(_smooth(s, 5), e / 100.0)
    a = i_t
    z = i_t + w - 1
    while a > 0 and local[a - 1] >= 0.95 * E_t:
        a -= 1
    while z < n_ini - 1 and local[z + 1] >= 0.95 * E_t:
        z += 1
    span = float(e[z] - e[a])
    lineal = span >= 0.25

    if lineal:
        modulo = {"tipo": "elasticidad", "valor": float(E_t), "clausula": "11.4 y A1.2",
                  "descripcion": f"pendiente de la porción lineal inicial ({e[a] - toe:.2f}–{e[z] - toe:.2f} % de deformación corregida)",
                  "r2_tangente": float(r2_t)}
    else:
        # 11.5 y A1.3: sin proporcionalidad evidente, módulo secante desde el cero corregido B'
        target = secant_pct
        if target > ec[r]:
            raise ValueError("La deformación designada para el módulo secante es mayor que la deformación a la ruptura.")
        sig = float(np.interp(target, ec, s))
        modulo = {"tipo": "secante", "valor": sig / (target / 100.0), "clausula": "11.5 y A1.3",
                  "descripcion": f"módulo secante a {target:g} % de deformación, medido desde el cero corregido",
                  "deformacion_designada": target, "tangente_maxima": float(E_t), "r2_tangente": float(r2_t)}

    # --- Punto de cedencia (A2.22): primer punto donde la deformación aumenta sin aumento de esfuerzo
    ss = _smooth(s[: r + 1])
    yi = None
    k_span = 1.0
    for i in range(i_t + w, r):
        if ss[i + 1] <= ss[i]:
            ahead = np.where(e[i + 1: r + 1] <= e[i] + k_span)[0]
            nxt = ss[i + 1: i + 1 + max(len(ahead), 1)]
            if len(nxt) and ss[i] > np.max(nxt) - 1e-12:
                yi = int(i - 4 + np.argmax(s[max(0, i - 4): i + 5]))  # máximo crudo cerca del punto suavizado
                break
    cedencia = None if yi is None else {"esfuerzo": float(s[yi]), "deformacion": float(ec[yi])}

    # --- Resistencia a la tensión (11.2, A2.18)
    if yi is not None and s[yi] >= smax * 0.999:
        designacion = "en la cedencia"
    elif not rotura:
        designacion = "máximo del registro (no se registró la ruptura)"
    elif s[r] >= smax * 0.999:
        designacion = "a la ruptura"
    else:
        designacion = "esfuerzo máximo entre cedencia y ruptura"

    # --- Tenacidad (no definida en D638; la pide el manual del laboratorio)
    seg = slice(0, r + 1)
    e_t = np.clip(ec[seg], 0, None) / 100.0
    s_t = np.clip(s[seg], 0, None)
    tenacidad = float(np.sum((s_t[1:] + s_t[:-1]) * np.diff(e_t)) / 2) if rotura else None  # trapecio; sin ruptura no aplica

    out = {
        "n_puntos": int(len(e)),
        "pie_de_curva": toe,
        "modulo": modulo,
        "resistencia_tension": smax,
        "designacion": designacion,
        "deformacion_en_maximo": float(ec[imax]),
        "cedencia": cedencia,
        "ruptura": ({"esfuerzo": float(s[r]), "deformacion": float(ec[r]), "indice": int(r), "detectada": True} if rotura else
                    {"esfuerzo": None, "deformacion": None, "indice": None, "detectada": False,
                     "fin_del_registro": {"esfuerzo": float(s[r]), "deformacion": float(ec[r])}}),
        "tenacidad_MJ_m3": tenacidad,
    }
    if t is not None:
        out["tiempo_a_ruptura_s"] = float(t[r] - t[0])
    if X is not None and t is not None:
        q = slice(0, max(r // 2, 10))
        out["velocidad_mm_min"] = float(np.polyfit(t[q], X[q], 1)[0] * 60.0)
    if X is not None:
        q = (e > 1) & (np.arange(len(e)) <= r)
        if q.sum() > 10:
            out["longitud_referencia_mm"] = float(np.polyfit(e[q], X[q], 1)[0] * 100.0)
    if F is not None:
        q = s > 0.05 * smax
        out["area_mm2"] = float(np.median(F[q] / s[q]))

    # curva corregida y reducida para graficar (se conservan siempre cedencia, máximo y ruptura)
    keep = set(range(0, r + 1, max(1, int(np.ceil((r + 1) / 600))))) | {imax, r} | ({yi} if yi is not None else set())
    keep |= set(range(r + 1, min(len(e), r + 4)))
    idx = sorted(keep)
    out["deformacion"] = ec[idx].tolist()
    out["esfuerzo"] = s[idx].tolist()
    return out


def _stats(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    sd = float(np.sqrt((np.sum(v ** 2) - len(v) * v.mean() ** 2) / (len(v) - 1))) if len(v) > 1 else None  # Ec. (1), 11.7
    return {"n": int(len(v)), "promedio": float(v.mean()), "desv_est": sd}


def summarize(results):
    """11.6 y 11.7: promedio y desviación estándar estimada de cada propiedad."""
    g = lambda f: [f(r) for r in results]
    return {
        "modulo": _stats(g(lambda r: r["modulo"]["valor"])),
        "resistencia_tension": _stats(g(lambda r: r["resistencia_tension"])),
        "esfuerzo_cedencia": _stats(g(lambda r: r["cedencia"]["esfuerzo"] if r["cedencia"] else None)),
        "deformacion_cedencia": _stats(g(lambda r: r["cedencia"]["deformacion"] if r["cedencia"] else None)),
        "esfuerzo_ruptura": _stats(g(lambda r: r["ruptura"]["esfuerzo"])),
        "deformacion_ruptura": _stats(g(lambda r: r["ruptura"]["deformacion"])),
        "tenacidad": _stats(g(lambda r: r["tenacidad_MJ_m3"])),
    }


def checks(results):
    """Verificaciones de conformidad con D638 que se pueden hacer a partir de los datos."""
    c = []
    n = len(results)
    c.append({"clausula": "7.1", "estado": "ok" if n >= 5 else "aviso",
              "texto": f"Se ensayaron {n} probetas." + ("" if n >= 5 else " D638 pide al menos 5 por muestra en materiales isotrópicos.")})
    sin = [r.get("sample", "") for r in results if not r["ruptura"]["detectada"]]
    if sin:
        c.append({"clausula": "10.6", "estado": "aviso",
                  "texto": f"No se registró la ruptura en {', '.join(sin)}: la curva termina sin caída de esfuerzo. No se reportan "
                           "esfuerzo ni elongación a la ruptura ni tenacidad para esas probetas."})
    c.append({"clausula": "7.3", "estado": "info",
              "texto": "Descarte las probetas que rompan en un defecto o fuera de la sección angosta y repita el ensayo. "
                       "Esto no se puede verificar con los datos: revise cada probeta."})
    v = [r["velocidad_mm_min"] for r in results if "velocidad_mm_min" in r]
    if v:
        vm = float(np.mean(v))
        match = [f"{sp} mm/min ± {int(tol * 100)} % (tipos {ty})" for sp, tol, ty in TABLA1 if abs(vm - sp) <= sp * tol]
        c.append({"clausula": "8.2 y Tabla 1", "estado": "ok" if match else "aviso",
                  "texto": f"Velocidad medida: {vm:.3g} mm/min. " + ("Coincide con " + "; ".join(match) + "."
                           if match else "No coincide con ninguna velocidad de la Tabla 1.")})
    tr = [r["tiempo_a_ruptura_s"] / 60 for r in results if "tiempo_a_ruptura_s" in r]
    if tr:
        fuera = [x for x in tr if not 0.5 <= x <= 5]
        c.append({"clausula": "8.2", "estado": "ok" if not fuera else "aviso",
                  "texto": f"Tiempo a la ruptura: {min(tr):.2g}–{max(tr):.2g} min. La norma pide la velocidad más baja que produzca la ruptura entre 0.5 y 5 min."})
    L = [r["longitud_referencia_mm"] for r in results if "longitud_referencia_mm" in r]
    if L:
        Lm = float(np.mean(L))
        es_g = any(abs(Lm - g) <= 0.05 * g for g in GAGE.values())
        c.append({"clausula": "10.3 Nota 11 y 11.3", "estado": "ok" if es_g else "aviso",
                  "texto": (f"La deformación corresponde a una longitud de referencia de {Lm:.3g} mm. " +
                            ("Coincide con una longitud calibrada de la Fig. 1." if es_g else
                             "No coincide con una longitud calibrada de extensómetro de la Fig. 1 (50 mm en tipos I–III, 25 mm en IV, 7.62 mm en V); "
                             "parece la separación entre mordazas. D638 prohíbe usar la separación de mordazas para calcular módulo o "
                             "elongación: el módulo es aparente y la deformación a la ruptura se reporta como deformación nominal (11.3.2)."))})
    else:
        c.append({"clausula": "10.3 Nota 11", "estado": "info",
                  "texto": "No se pudo verificar si la deformación viene de un extensómetro. El módulo y la elongación solo son válidos con extensómetro."})
    A = [r["area_mm2"] for r in results if "area_mm2" in r]
    c.append({"clausula": "10.1", "estado": "info",
              "texto": (f"Área deducida de los datos: {np.mean(A):.3g} mm². " if A else "") +
                       "Registre el ancho y el espesor medidos de cada probeta (al 0.025 mm) y el tipo de probeta de la Fig. 1."})
    c.append({"clausula": "11.1 y Anexo A1", "estado": "ok",
              "texto": "Se aplicó la corrección del pie de curva: las deformaciones se miden desde la intersección de la tangente con el eje de deformación ("
                       + ", ".join(f"{r['pie_de_curva']:.3g} %" for r in results) + ")."})
    tipos = {r["modulo"]["tipo"] for r in results}
    c.append({"clausula": "11.4 / 11.5", "estado": "info",
              "texto": "Módulo de elasticidad de la porción lineal inicial (11.4)." if tipos == {"elasticidad"} else
                       "Al menos una probeta no muestra porción lineal evidente: se reporta el módulo secante a la deformación designada (11.5), "
                       "como pide la norma." if tipos == {"secante"} else
                       "Las probetas difieren: unas tienen porción lineal (módulo de elasticidad, 11.4) y otras no (módulo secante, 11.5). "
                       "Revise antes de promediar."})
    c.append({"clausula": "9 y 12.1", "estado": "info",
              "texto": "Acondicionamiento (Práctica D618, procedimiento A) y condiciones del cuarto (el manual pide 23 ± 2 °C y 50 ± 5 % HR): "
                       "regístrelos en el reporte; no vienen en el archivo."})
    c.append({"clausula": "Manual del laboratorio", "estado": "info",
              "texto": "La tenacidad (área bajo la curva) la pide el manual de la práctica 7; D638 no la define."})
    return c
