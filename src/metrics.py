"""
metrics.py
----------
Calculo de metricas de desempeno a partir de una serie de retornos
periodicos (mensuales o diarios). Reutilizable por todas las estrategias.

FIX (bug estrategias 10-14): la anualizacion ya NO esta hardcodeada en
base mensual (12). Ahora se recibe 'periods_per_year' explicito:
  - periods_per_year=12   -> estrategias con datos MENSUALES (1-9, 15+)
  - periods_per_year=252  -> estrategias con datos DIARIOS (10-14)
"""

import numpy as np
import pandas as pd


def calcular_metricas(
    resultado: pd.DataFrame,
    columna_retorno: str,
    columna_equity: str,
    periods_per_year: int = 12,
) -> dict:
    """
    columna_retorno: nombre de la columna con los retornos periodicos netos
                      (ej. 'retorno_estrategia_neto' o 'retorno_mensual_activo')
    columna_equity:  nombre de la columna con la curva de equity acumulada
                      correspondiente (ej. 'equity_estrategia' o 'equity_buy_and_hold')
    periods_per_year: cuantos periodos de esta serie caben en un ano.
                       12 para datos mensuales, 252 para datos diarios de trading.
                       ¡Debe coincidir con la frecuencia real de 'resultado'!
    """
    r = resultado[columna_retorno].dropna()
    n_periodos = len(r)

    ret_total = resultado[columna_equity].iloc[-1] - 1
    ret_anual = (1 + ret_total) ** (periods_per_year / n_periodos) - 1 if n_periodos > 0 else np.nan
    vol_anual = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / vol_anual if vol_anual > 0 else np.nan

    equity = (1 + r).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "retorno_total": ret_total,
        "retorno_anualizado": ret_anual,
        "volatilidad_anualizada": vol_anual,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "periodos": n_periodos,
    }


def imprimir_metricas(nombre: str, metricas: dict):
    print(f">> {nombre}:")
    for k, v in metricas.items():
        if k != "periodos":
            print(f"   {k:25s}: {v:.4f}")
