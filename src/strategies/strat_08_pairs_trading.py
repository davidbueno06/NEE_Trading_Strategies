"""
Estrategia 8 (Seccion 3.8 del paper): Pairs trading.

Par: NEE (NextEra Energy) y CEG (Constellation Energy) — ambas del sector
de energia/utilities del S&P 500, historicamente correlacionadas.

Cada mes, se compara el retorno logaritmico de NEE contra el de CEG en la
ventana de formacion. El que "se adelanto" (retorno relativo positivo) se
vende en corto; el que "se quedo atras" (retorno relativo negativo) se
compra -- apostando a que la brecha se revierte. Posicion dollar-neutral.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import descargar_precios_mensuales
from metrics import calcular_metricas, imprimir_metricas
from multi_asset_reversion import calcular_pesos_mean_reversion, backtest_multi_activo

TICKERS = ["NEE", "CEG"]
VENTANA_FORMACION = 3   # meses usados para medir la divergencia entre el par
COST_BPS = 10


def main():
    precios = pd.DataFrame({t: descargar_precios_mensuales(t, start="2005-01-01") for t in TICKERS}).dropna()

    pesos = calcular_pesos_mean_reversion(precios, VENTANA_FORMACION)
    resultado = backtest_multi_activo(precios, pesos, costo_bps=COST_BPS)

    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia")
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold")

    print(f"\n=== Estrategia 8: Pairs trading ({TICKERS[0]} / {TICKERS[1]}) ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['meses']} meses)\n")
    imprimir_metricas("Estrategia pairs trading (neta de costos)", m_estrategia)
    print()
    imprimir_metricas(f"Referencia: promedio equal-weight {TICKERS[0]}/{TICKERS[1]}", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Pairs trading (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label=f"Promedio {TICKERS[0]}/{TICKERS[1]} (buy & hold)", linestyle="--")
    ax.set_title(f"Estrategia 8 (Pairs Trading: {TICKERS[0]} vs {TICKERS[1]})")
    ax.set_ylabel("Valor de $1 invertido")
    ax.set_xlabel("Fecha")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    carpeta_figuras = os.path.join(carpeta_raiz, "outputs", "figures")
    carpeta_resultados = os.path.join(carpeta_raiz, "outputs", "results")
    os.makedirs(carpeta_figuras, exist_ok=True)
    os.makedirs(carpeta_resultados, exist_ok=True)
    ruta_png = os.path.join(carpeta_figuras, "strat_08_pairs_trading_NEE_CEG.png")
    ruta_csv = os.path.join(carpeta_resultados, "strat_08_pairs_trading_NEE_CEG.csv")
    fig.savefig(ruta_png, dpi=150)
    resultado.to_csv(ruta_csv)
    print(f"\nGrafico guardado en: {ruta_png}")
    print(f"Detalle mensual guardado en: {ruta_csv}")


if __name__ == "__main__":
    main()
