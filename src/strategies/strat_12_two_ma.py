"""
Estrategia 11 (Seccion 3.12): Two moving averages.

Signal (Eq. 322): largo si MA(T') > MA(T); corto si MA(T') < MA(T),
con T' < T (aqui T'=10, T=30 dias).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import descargar_precios_diarios_ohlc
from metrics import calcular_metricas, imprimir_metricas
from backtest_engine import backtest

TICKER = "NEE"
T_CORTA = 10
T_LARGA = 30
COST_BPS = 10
PERIODS_PER_YEAR = 252  # datos DIARIOS (fix bug de anualizacion)


def calcular_senales(precios: pd.Series, t_corta: int, t_larga: int) -> pd.Series:
    ma_corta = precios.rolling(t_corta).mean()
    ma_larga = precios.rolling(t_larga).mean()
    posiciones = np.sign(ma_corta - ma_larga).fillna(0)
    posiciones.name = "posicion"
    return posiciones


def main():
    ohlc = descargar_precios_diarios_ohlc(TICKER, start="2005-01-01")
    precios = ohlc["Close"]
    posiciones = calcular_senales(precios, T_CORTA, T_LARGA)

    resultado = backtest(precios, posiciones, holding=1, costo_bps=COST_BPS)
    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia", periods_per_year=PERIODS_PER_YEAR)
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold", periods_per_year=PERIODS_PER_YEAR)

    print(f"\n=== Estrategia 11: Two moving averages (T'={T_CORTA}d, T={T_LARGA}d) sobre {TICKER} ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['periodos']} dias)\n")
    imprimir_metricas("Estrategia dos MA (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia dos MA (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 11 (Two Moving Averages) vs Buy & Hold — {TICKER}")
    ax.set_ylabel("Valor de $1 invertido")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    carpeta_figuras = os.path.join(carpeta_raiz, "outputs", "figures")
    carpeta_resultados = os.path.join(carpeta_raiz, "outputs", "results")
    os.makedirs(carpeta_figuras, exist_ok=True)
    os.makedirs(carpeta_resultados, exist_ok=True)
    fig.savefig(os.path.join(carpeta_figuras, "strat_11_two_ma_NEE.png"), dpi=150)
    resultado.to_csv(os.path.join(carpeta_resultados, "strat_11_two_ma_NEE.csv"))
    print(f"\nArchivos guardados en outputs/figures y outputs/results.")


if __name__ == "__main__":
    main()
