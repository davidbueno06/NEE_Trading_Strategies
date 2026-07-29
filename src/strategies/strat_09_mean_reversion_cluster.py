"""
Estrategia 9 (Seccion 3.9 del paper): Mean-reversion - cluster de 3 acciones.

Generalizacion de pairs trading a N=3 activos historicamente correlacionados
del mismo sector: NEE, DUK (Duke Energy) y SO (Southern Company) — tres
grandes utilities del S&P 500.

Misma logica que pairs trading (Eqs. 292-298): cada mes se compra el(los)
activo(s) que quedaron por debajo del retorno promedio del cluster, y se
vende(n) en corto el(los) que quedaron por arriba, con pesos dollar-neutral
proporcionales a la desviacion respecto al promedio.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import descargar_precios_mensuales
from metrics import calcular_metricas, imprimir_metricas
from multi_asset_reversion import calcular_pesos_mean_reversion, backtest_multi_activo

TICKERS = ["NEE", "DUK", "SO"]
VENTANA_FORMACION = 3
COST_BPS = 10


def main():
    precios = pd.DataFrame({t: descargar_precios_mensuales(t, start="2005-01-01") for t in TICKERS}).dropna()

    pesos = calcular_pesos_mean_reversion(precios, VENTANA_FORMACION)
    resultado = backtest_multi_activo(precios, pesos, costo_bps=COST_BPS)

    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia")
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold")

    nombre_cluster = "/".join(TICKERS)
    print(f"\n=== Estrategia 9: Mean-reversion, cluster ({nombre_cluster}) ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['meses']} meses)\n")
    imprimir_metricas("Estrategia mean-reversion cluster (neta de costos)", m_estrategia)
    print()
    imprimir_metricas(f"Referencia: promedio equal-weight {nombre_cluster}", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Mean-reversion cluster (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label=f"Promedio {nombre_cluster} (buy & hold)", linestyle="--")
    ax.set_title(f"Estrategia 9 (Mean-reversion cluster: {nombre_cluster})")
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
    ruta_png = os.path.join(carpeta_figuras, "strat_09_mean_reversion_cluster.png")
    ruta_csv = os.path.join(carpeta_resultados, "strat_09_mean_reversion_cluster.csv")
    fig.savefig(ruta_png, dpi=150)
    resultado.to_csv(ruta_csv)
    print(f"\nGrafico guardado en: {ruta_png}")
    print(f"Detalle mensual guardado en: {ruta_csv}")


if __name__ == "__main__":
    main()
