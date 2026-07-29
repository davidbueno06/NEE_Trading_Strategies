"""
Estrategia 14 (Seccion 3.15): Channel (Donchian Channel).

Banda (Eqs. 329-330), calculada sobre los T dias PREVIOS (sin incluir hoy):
    Bup = max(P) en la ventana
    Bdown = min(P) en la ventana

Signal (Eq. 331): largo cuando el precio toca el piso del canal (Bdown);
corto cuando toca el techo (Bup). Se usa histeresis: mantiene la posicion
hasta que se toca el extremo opuesto.
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
T_CANAL = 20  # dias de trading (~1 mes)
COST_BPS = 10
PERIODS_PER_YEAR = 252  # datos DIARIOS (fix bug de anualizacion)


def calcular_senales(precios: pd.Series, t_canal: int) -> pd.Series:
    banda_sup = precios.shift(1).rolling(t_canal).max()
    banda_inf = precios.shift(1).rolling(t_canal).min()

    df = pd.DataFrame({"P": precios, "sup": banda_sup, "inf": banda_inf}).dropna()

    toca_piso = df["P"] <= df["inf"]
    toca_techo = df["P"] >= df["sup"]

    # entra largo al tocar el piso, se mantiene hasta tocar el techo (y viceversa)
    posiciones = aplicar_maquina_estados(
        entra_largo=toca_piso, sale_largo=toca_techo,
        entra_corto=toca_techo, sale_corto=toca_piso,
    )
    return posiciones


def main():
    ohlc = descargar_precios_diarios_ohlc(TICKER, start="2005-01-01")
    precios = ohlc["Close"]
    posiciones = calcular_senales(precios, T_CANAL)

    resultado = backtest(precios, posiciones, holding=1, costo_bps=COST_BPS)
    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia", periods_per_year=PERIODS_PER_YEAR)
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold", periods_per_year=PERIODS_PER_YEAR)

    print(f"\n=== Estrategia 14: Channel / Donchian ({T_CANAL}d) sobre {TICKER} ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['periodos']} dias)\n")
    imprimir_metricas("Estrategia canal (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia canal (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 14 (Channel / Donchian) vs Buy & Hold — {TICKER}")
    ax.set_ylabel("Valor de $1 invertido")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    carpeta_figuras = os.path.join(carpeta_raiz, "outputs", "figures")
    carpeta_resultados = os.path.join(carpeta_raiz, "outputs", "results")
    os.makedirs(carpeta_figuras, exist_ok=True)
    os.makedirs(carpeta_resultados, exist_ok=True)
    fig.savefig(os.path.join(carpeta_figuras, "strat_14_channel_NEE.png"), dpi=150)
    resultado.to_csv(os.path.join(carpeta_resultados, "strat_14_channel_NEE.csv"))
    print(f"\nArchivos guardados en outputs/figures y outputs/results.")


if __name__ == "__main__":
    main()
