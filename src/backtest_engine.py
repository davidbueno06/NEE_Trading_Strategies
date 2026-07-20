"""
backtest_engine.py
-------------------
Motor de backtest generico: recibe precios y una serie de posiciones
(-1, 0, +1, o pesos) YA calculadas por la estrategia, y se encarga de:
  - aplicar la posicion con 1 mes de rezago (evita look-ahead bias)
  - sostener la posicion durante el "holding period"
  - descontar costos de transaccion
  - construir las curvas de equity (estrategia vs buy & hold)

Cada estrategia (1 a 21) solo necesita generar su propia serie de
"senales/posiciones"; el resto del proceso es identico para todas.
"""

import numpy as np
import pandas as pd


def backtest(precios: pd.Series, posiciones: pd.Series, holding: int = 1, costo_bps: float = 10) -> pd.DataFrame:
    ret_mensual = precios.pct_change().dropna()

    pos_aplicada = posiciones.shift(1).reindex(ret_mensual.index)
    if holding > 1:
        pos_aplicada = pos_aplicada.ffill(limit=holding - 1)
    pos_aplicada = pos_aplicada.fillna(0)

    ret_estrategia_bruto = pos_aplicada * ret_mensual

    cambios = pos_aplicada.diff().abs().fillna(0)
    costos = cambios * (costo_bps / 10000)
    ret_estrategia_neto = ret_estrategia_bruto - costos

    resultado = pd.DataFrame({
        "retorno_mensual_activo": ret_mensual,
        "posicion": pos_aplicada,
        "retorno_estrategia_bruto": ret_estrategia_bruto,
        "retorno_estrategia_neto": ret_estrategia_neto,
    })
    resultado["equity_estrategia"] = (1 + resultado["retorno_estrategia_neto"]).cumprod()
    resultado["equity_buy_and_hold"] = (1 + resultado["retorno_mensual_activo"]).cumprod()
    return resultado
