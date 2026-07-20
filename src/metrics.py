"""
metrics.py
----------
Calculo de metricas de desempeno a partir de una serie de retornos
mensuales. Reutilizable por todas las estrategias.
"""

import numpy as np
import pandas as pd


def calcular_metricas(resultado: pd.DataFrame, columna_retorno: str, columna_equity: str) -> dict:
    """
    columna_retorno: nombre de la columna con los retornos mensuales netos
                      (ej. 'retorno_estrategia_neto' o 'retorno_mensual_activo')
    columna_equity:  nombre de la columna con la curva de equity acumulada
                      correspondiente (ej. 'equity_estrategia' o 'equity_buy_and_hold')
    """
    r = resultado[columna_retorno].dropna()
    n_meses = len(r)

    ret_total = resultado[columna_equity].iloc[-1] - 1
    ret_anual = (1 + ret_total) ** (12 / n_meses) - 1 if n_meses > 0 else np.nan
    vol_anual = r.std(ddof=1) * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol_anual if vol_anual > 0 else np.nan

    equity = (1 + r).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "retorno_total": ret_total,
        "retorno_anualizado": ret_anual,
        "volatilidad_anualizada": vol_anual,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "meses": n_meses,
    }


def imprimir_metricas(nombre: str, metricas: dict):
    print(f">> {nombre}:")
    for k, v in metricas.items():
        if k != "meses":
            print(f"   {k:25s}: {v:.4f}")
