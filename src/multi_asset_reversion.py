"""
multi_asset_reversion.py
--------------------------
Motor compartido para las Estrategias 8 (Pairs trading, N=2) y 9
(Mean-reversion cluster, N=3), Secciones 3.8 y 3.9 del paper.

Logica (Eqs. 292-298): en cada mes, se calcula el retorno logaritmico de
cada activo en la ventana de formacion, se "demedia" respecto al promedio
del grupo, y se apuesta en contra de la desviacion (short al que se alejo
por arriba = "rico"; long al que se alejo por abajo = "barato"), con
pesos dollar-neutral proporcionales a la desviacion:

    R_i = ln(P_i(t2)/P_i(t1))                    Eq. (292)
    R_promedio = mean(R_i)                        Eq. (293)
    R_demeaned_i = R_i - R_promedio                Eq. (294)
    peso_i = -R_demeaned_i / sum(|R_demeaned_i|)   Eqs. (295)-(298)

El peso resultante es dollar-neutral (suma de pesos firmados = 0) y con
exposicion bruta normalizada a 1 (suma de |pesos| = 1).
"""

import numpy as np
import pandas as pd


def calcular_pesos_mean_reversion(precios: pd.DataFrame, ventana_formacion: int) -> pd.DataFrame:
    """
    precios: DataFrame con una columna por activo, precios mensuales.
    Devuelve un DataFrame de pesos (una columna por activo) para cada mes,
    calculados con el retorno logaritmico acumulado de los 'ventana_formacion'
    meses previos.
    """
    log_ret = np.log(precios / precios.shift(1)).dropna()
    ret_ventana = log_ret.rolling(ventana_formacion).sum().dropna()

    promedio = ret_ventana.mean(axis=1)
    demeaned = ret_ventana.sub(promedio, axis=0)

    suma_abs = demeaned.abs().sum(axis=1)
    pesos = demeaned.div(-suma_abs, axis=0)  # signo negativo: corto al "rico", largo al "barato"
    pesos = pesos.replace([np.inf, -np.inf], np.nan).dropna()
    return pesos


def backtest_multi_activo(precios: pd.DataFrame, pesos: pd.DataFrame, costo_bps: float = 10) -> pd.DataFrame:
    ret_mensual = precios.pct_change().dropna()

    pesos_aplicados = pesos.shift(1).reindex(ret_mensual.index).fillna(0)

    ret_estrategia_bruto = (pesos_aplicados * ret_mensual).sum(axis=1)

    cambios = pesos_aplicados.diff().abs().sum(axis=1).fillna(0)
    costos = cambios * (costo_bps / 10000)
    ret_estrategia_neto = ret_estrategia_bruto - costos

    resultado = pd.DataFrame({
        "retorno_estrategia_bruto": ret_estrategia_bruto,
        "retorno_estrategia_neto": ret_estrategia_neto,
    })
    resultado["equity_estrategia"] = (1 + resultado["retorno_estrategia_neto"]).cumprod()

    # buy & hold de referencia: promedio simple (equal-weight) de los activos del grupo
    ret_bh_equal = ret_mensual.mean(axis=1)
    resultado["retorno_mensual_activo"] = ret_bh_equal
    resultado["equity_buy_and_hold"] = (1 + ret_bh_equal).cumprod()

    return resultado
