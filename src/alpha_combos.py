"""
alpha_combos.py
------------------
Motor compartido para la Estrategia 20 (Alpha combos), Seccion 3.20 del
paper. Implementa el procedimiento de fijacion de pesos de Kakushadze &
Yu (2017b) citado en el paper (Eq. 360 y los 11 pasos que la rodean),
para combinar N "alphas" en un unico "mega-alpha".

ADAPTACION A UN SOLO ACTIVO (consistente con el resto del proyecto):
En el paper, cada alpha i produce holdings para un universo de ~2,500
acciones, y N (numero de alphas) puede ser enorme. Aqui, como venimos
haciendo desde la Estrategia 1, se colapsa el universo a un solo activo
(NEE): cada "alpha i" es una de las estrategias YA implementadas en este
proyecto (momentum, value, low-vol, residual momentum, pairs, mean-rev
cluster, medias moviles, etc.) aplicada a NEE, y "R_is" (retorno
realizado del alpha i en el periodo s, Eq. 360 y alrededores) es el
retorno neto mensual de ESA estrategia en el mes s. El paso de
"demeaning cross-seccional" del paper (entre miles de acciones) se
traduce aqui en demeaning entre las N estrategias disponibles en cada
mes.

DIFERENCIA DELIBERADA vs. el paper (y por que): el paper describe el
procedimiento como si se corriera una sola vez sobre toda la muestra
para fijar los pesos w_i de una vez. Eso tiene look-ahead bias si se usa
tal cual en un backtest (los pesos "verian" datos futuros). Aqui el
procedimiento de los 11 pasos se recalcula en una ventana movil de
'ventana_M' meses, usando SOLO datos hasta el mes anterior al que se
va a operar, y el resultado se aplica con 1 mes adicional de rezago
dentro de backtest_alpha_combo() -- exactamente el mismo criterio de
"sin look-ahead" que se uso en backtest_engine.py y multi_asset_reversion.py.
"""

import numpy as np
import pandas as pd


def _pesos_en_ventana(ventana: pd.DataFrame, dias_promedio_E: int) -> pd.Series:
    """
    Aplica los pasos 1-11 del procedimiento sobre una ventana de retornos
    realizados de N alphas (columnas) x M periodos (filas, orden
    cronologico, el ultimo renglon es el periodo mas reciente DISPONIBLE
    -- es decir, ya es pasado respecto al periodo que se va a predecir).

    Devuelve una Serie de pesos w_i (una por alpha/columna), normalizada
    para que sum(|w_i|) = 1 (paso 11).
    """
    R = ventana  # filas = tiempo (s = 1..M), columnas = alphas (i = 1..N)

    # Paso 2: retornos demeaned en el tiempo, por alpha -> X_is
    X = R - R.mean(axis=0)

    # Paso 3: varianza muestral de cada alpha -> sigma_i^2
    sigma2 = X.var(axis=0, ddof=1)
    sigma = np.sqrt(sigma2).replace(0, np.nan)

    # Paso 4: retornos normalizados -> Y_is
    Y = X.div(sigma, axis=1)

    # Paso 6: demeaned cross-seccional (entre alphas, en cada periodo s) -> Lambda_is
    Lambda = Y.sub(Y.mean(axis=1), axis=0)

    # Paso 8: retorno esperado del alpha E_i, via promedio movil de los
    # ultimos 'dias_promedio_E' periodos de la ventana (Eq. 360), luego
    # normalizado -> E_i~ = E_i / sigma_i
    E = R.iloc[-dias_promedio_E:].mean(axis=0)
    E_norm = E / sigma

    # Paso 9: residuos de la regresion (sin intercepto, pesos unitarios)
    # de E_i~ (una observacion por alpha) sobre Lambda_is. Se usa el
    # promedio temporal de Lambda_i como el regresor por alpha -- es la
    # forma natural de tener "una observacion por alpha" a partir de una
    # matriz tiempo x alpha, y es consistente con que Lambda captura la
    # exposicion cross-seccional promedio de cada alpha en la ventana.
    x = Lambda.mean(axis=0)
    y = E_norm

    validos = x.notna() & y.notna()
    x_v, y_v = x[validos], y[validos]

    if len(x_v) < 2 or float(x_v @ x_v) == 0:
        residuos = y_v  # si no se puede regresar (poca varianza/datos), no se resta nada
    else:
        beta = float((x_v @ y_v) / (x_v @ x_v))
        residuos = y_v - beta * x_v

    # Paso 10: peso propuesto w_i = eta * residuo_i / sigma_i
    w_bruto = residuos / sigma[validos.index][validos]

    # Paso 11: normalizar para que sum(|w_i|) = 1
    suma_abs = w_bruto.abs().sum()
    if suma_abs == 0 or np.isnan(suma_abs):
        return pd.Series(0.0, index=R.columns)

    w = (w_bruto / suma_abs).reindex(R.columns).fillna(0.0)
    return w


def calcular_pesos_alpha_combo(
    retornos_alphas: pd.DataFrame,
    ventana_M: int = 36,
    dias_promedio_E: int = 12,
    min_alphas_activos: int = 3,
) -> pd.DataFrame:
    """
    retornos_alphas: DataFrame, index = fechas mensuales, columnas = una
    por cada estrategia/alpha ya implementada (su 'retorno_estrategia_neto'
    mensual). Puede tener NaN al inicio si una estrategia empieza mas
    tarde que las demas (ej. pairs trading con CEG, que cotiza poco
    tiempo).

    Devuelve un DataFrame de pesos w_i(t) (mismo shape que retornos_alphas),
    calculado en cada fecha t usando SOLO los 'ventana_M' meses previos a
    t (sin ver el mes t ni posteriores). backtest_alpha_combo() se
    encarga de rezagar un mes adicional antes de aplicar estos pesos.
    """
    fechas = retornos_alphas.index
    pesos = pd.DataFrame(0.0, index=fechas, columns=retornos_alphas.columns)

    for idx in range(ventana_M, len(fechas)):
        fecha_actual = fechas[idx]
        ventana = retornos_alphas.iloc[idx - ventana_M: idx]  # estrictamente pasado

        activos = ventana.columns[ventana.notna().all(axis=0)]
        if len(activos) < min_alphas_activos:
            continue

        w = _pesos_en_ventana(ventana[activos], dias_promedio_E)
        pesos.loc[fecha_actual, activos] = w.values

    return pesos


def backtest_alpha_combo(
    retornos_alphas: pd.DataFrame,
    pesos: pd.DataFrame,
    costo_bps: float = 10,
) -> pd.DataFrame:
    """
    Combina los retornos NETOS de las N estrategias usando los pesos
    w_i(t) ya calculados (estos ya solo usan pasado hasta t-1; aqui se
    rezagan un mes adicional para asegurar que se conocen ANTES de que
    empiece el mes que van a ponderar -- mismo criterio que las demas
    estrategias del proyecto).

    NOTA: los costos de transaccion aqui son costos ADICIONALES por
    reponderar el combo (cambios en w_i). Los costos de operar cada
    estrategia individual YA estan incluidos en 'retornos_alphas' (son
    retornos netos), asi que no se duplican.
    """
    pesos_aplicados = pesos.shift(1).reindex(retornos_alphas.index).fillna(0)

    ret_combo_bruto = (pesos_aplicados * retornos_alphas.fillna(0)).sum(axis=1)

    cambios = pesos_aplicados.diff().abs().sum(axis=1).fillna(0)
    costos = cambios * (costo_bps / 10000)
    ret_combo_neto = ret_combo_bruto - costos

    resultado = pd.DataFrame({
        "retorno_estrategia_bruto": ret_combo_bruto,
        "retorno_estrategia_neto": ret_combo_neto,
    })
    resultado["equity_estrategia"] = (1 + resultado["retorno_estrategia_neto"]).cumprod()
    return resultado
