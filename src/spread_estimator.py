"""
spread_estimator.py
---------------------
Estimador de spread efectivo bid-ask a partir de precios OHLC diarios,
sin necesidad de datos de cotizaciones (bid/ask reales), siguiendo:

  Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask
  Spreads from Daily High and Low Prices." Journal of Finance, 67(2).

Se usa en la Estrategia 19 (Market-making) como proxy del costo de
cruzar el spread dia a dia, en vez de asumir un costo de transaccion
generico y fijo (10 bps) como en las demas estrategias del proyecto.

LOGICA (resumen): el rango High-Low de un dia refleja tanto la
volatilidad verdadera del precio como el spread bid-ask. Usando dos dias
consecutivos y el rango del periodo combinado de 2 dias, el metodo
separa estadisticamente ambos componentes (la volatilidad escala con
sqrt(tiempo), el spread no), y despeja el spread implicito.

LIMITACION EXPLICITA (para el reporte): esto NO es un spread bid-ask
observado. Es una estimacion estadistica agregada, validada en la
literatura empirica para acciones liquidas de EE.UU., pero sigue siendo
una aproximacion -- no reemplaza datos reales de nivel de libro (NBBO).
"""

import numpy as np
import pandas as pd

_K1 = 3 - 2 * np.sqrt(2)  # constante del paper original (~0.1716)


def estimar_spread_corwin_schultz(precios_ohlc: pd.DataFrame) -> pd.Series:
    """
    precios_ohlc: DataFrame diario con columnas 'High' y 'Low' (indice de
    fechas de trading, ordenado ascendente).

    Devuelve una Serie diaria con el spread efectivo estimado (fraccion
    decimal, ej. 0.004 = 40 pbs). Los valores negativos (posibles por
    construccion del estimador) se truncan a 0, como recomienda el paper
    original. Los primeros valores son NaN (se necesitan 2 dias previos).
    """
    H = precios_ohlc["High"]
    L = precios_ohlc["Low"]

    if (L <= 0).any():
        raise ValueError("Se encontraron precios Low <= 0; revisar datos de entrada.")

    beta = (np.log(H / L)) ** 2 + (np.log(H.shift(1) / L.shift(1))) ** 2

    H_max = pd.concat([H, H.shift(1)], axis=1).max(axis=1)
    L_min = pd.concat([L, L.shift(1)], axis=1).min(axis=1)
    gamma = (np.log(H_max / L_min)) ** 2

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K1 - np.sqrt(gamma / _K1)

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0)
    spread.name = "spread_estimado"
    return spread
