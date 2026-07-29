"""
Estrategia 13 (Seccion 3.14): Support and resistance.

Pivote (Eqs. 325-327), calculado con el High/Low/Close del DIA ANTERIOR:
    C = (PH + PL + PC) / 3
    R = 2*C - PL
    S = 2*C - PH

Signal (Eq. 328), con histeresis:
  - Establecer largo si P > C ; liquidar largo si P >= R
  - Establecer corto si P < C ; liquidar corto si P <= S
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
COST_BPS = 10
PERIODS_PER_YEAR = 252  # datos DIARIOS (fix bug de anualizacion)


def calcular_senales(ohlc: pd.DataFrame) -> pd.Series:
    anterior = ohlc.shift(1)  # High/Low/Close del dia anterior
    C = (anterior["High"] + anterior["Low"] + anterior["Close"]) / 3
    R = 2 * C - anterior["Low"]
    S = 2 * C - anterior["High"]

    df = pd.DataFrame({"P": ohlc["Close"], "C": C, "R": R, "S": S}).dropna()

    entra_largo = df["P"] > df["C"]
    sale_largo = df["P"] >= df["R"]
    entra_corto = df["P"] < df["C"]
    sale_corto = df["P"] <= df["S"]

    posiciones = aplicar_maquina_estados(entra_largo, sale_largo, entra_corto, sale_corto)
    return posiciones


def main():
    ohlc = descargar_precios_diarios_ohlc(TICKER, start="2005-01-01")
    precios = ohlc["Close"]
    posiciones = calcular_senales(ohlc)

    resultado = backtest(precios, posiciones, holding=1, costo_bps=COST_BPS)
    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia", periods_per_year=PERIODS_PER_YEAR)
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold", periods_per_year=PERIODS_PER_YEAR)

    print(f"\n=== Estrategia 13: Support and resistance sobre {TICKER} ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['periodos']} dias)\n")
    imprimir_metricas("Estrategia soporte/resistencia (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia soporte/resistencia (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 13 (Support & Resistance) vs Buy & Hold — {TICKER}")
    ax.set_ylabel("Valor de $1 invertido")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    carpeta_figuras = os.path.join(carpeta_raiz, "outputs", "figures")
    carpeta_resultados = os.path.join(carpeta_raiz, "outputs", "results")
    os.makedirs(carpeta_figuras, exist_ok=True)
    os.makedirs(carpeta_resultados, exist_ok=True)
    fig.savefig(os.path.join(carpeta_figuras, "strat_13_support_resistance_NEE.png"), dpi=150)
    resultado.to_csv(os.path.join(carpeta_resultados, "strat_13_support_resistance_NEE.csv"))
    print(f"\nArchivos guardados en outputs/figures y outputs/results.")


if __name__ == "__main__":
    main()
