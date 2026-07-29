"""
Estrategia 12 (Seccion 3.13): Three moving averages.

Signal (Eq. 324), con histeresis (T1 < T2 < T3, aqui 3, 10, 21 dias):
  - Establecer largo si MA(T1) > MA(T2) > MA(T3)
  - Liquidar largo si MA(T1) <= MA(T2)
  - Establecer corto si MA(T1) < MA(T2) < MA(T3)
  - Liquidar corto si MA(T1) >= MA(T2)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import descargar_precios_diarios_ohlc
from metrics import calcular_metricas, imprimir_metricas
from backtest_engine import backtest
from state_machine_signals import aplicar_maquina_estados

TICKER = "NEE"
T1, T2, T3 = 3, 10, 21
COST_BPS = 10
PERIODS_PER_YEAR = 252  # datos DIARIOS (fix bug de anualizacion)


def calcular_senales(precios: pd.Series, t1: int, t2: int, t3: int) -> pd.Series:
    ma1 = precios.rolling(t1).mean()
    ma2 = precios.rolling(t2).mean()
    ma3 = precios.rolling(t3).mean()

    df = pd.DataFrame({"ma1": ma1, "ma2": ma2, "ma3": ma3}).dropna()

    entra_largo = (df["ma1"] > df["ma2"]) & (df["ma2"] > df["ma3"])
    sale_largo = df["ma1"] <= df["ma2"]
    entra_corto = (df["ma1"] < df["ma2"]) & (df["ma2"] < df["ma3"])
    sale_corto = df["ma1"] >= df["ma2"]

    posiciones = aplicar_maquina_estados(entra_largo, sale_largo, entra_corto, sale_corto)
    return posiciones


def main():
    ohlc = descargar_precios_diarios_ohlc(TICKER, start="2005-01-01")
    precios = ohlc["Close"]
    posiciones = calcular_senales(precios, T1, T2, T3)

    resultado = backtest(precios, posiciones, holding=1, costo_bps=COST_BPS)
    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia", periods_per_year=PERIODS_PER_YEAR)
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold", periods_per_year=PERIODS_PER_YEAR)

    print(f"\n=== Estrategia 12: Three moving averages ({T1}/{T2}/{T3}d) sobre {TICKER} ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['periodos']} dias)\n")
    imprimir_metricas("Estrategia tres MA (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia tres MA (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 12 (Three Moving Averages) vs Buy & Hold — {TICKER}")
    ax.set_ylabel("Valor de $1 invertido")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    carpeta_figuras = os.path.join(carpeta_raiz, "outputs", "figures")
    carpeta_resultados = os.path.join(carpeta_raiz, "outputs", "results")
    os.makedirs(carpeta_figuras, exist_ok=True)
    os.makedirs(carpeta_resultados, exist_ok=True)
    fig.savefig(os.path.join(carpeta_figuras, "strat_12_three_ma_NEE.png"), dpi=150)
    resultado.to_csv(os.path.join(carpeta_resultados, "strat_12_three_ma_NEE.csv"))
    print(f"\nArchivos guardados en outputs/figures y outputs/results.")


if __name__ == "__main__":
    main()
