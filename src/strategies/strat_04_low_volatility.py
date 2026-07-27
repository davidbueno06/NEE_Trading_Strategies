"""
Estrategia 4 (Seccion 3.4 del paper): Low-volatility anomaly.

Observacion empirica: acciones con MENOR volatilidad historica tienden a
tener, en el futuro, mejor desempeno ajustado por riesgo que las de MAYOR
volatilidad (contrario a la intuicion de que mas riesgo = mas retorno).

El paper compra el decil inferior de volatilidad (sigma_i, Eq. 270) y
vende el decil superior. Ventana tipica: 6-12 meses, sin skip period.

ADAPTACION: sin universo de acciones, se compara la volatilidad reciente
de NEE contra su propia historia (percentil/z-score sobre ventana movil):

    z_t = (sigma_t - media_historica_sigma) / desviacion_historica_sigma

z_t < 0 -> volatilidad actual BAJA respecto a su propia historia -> largo
z_t > 0 -> volatilidad actual ALTA respecto a su propia historia -> corto
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import descargar_precios_mensuales
from metrics import calcular_metricas, imprimir_metricas
from backtest_engine import backtest

TICKER = "NEE"
VENTANA_VOL_MESES = 12          # ventana para calcular sigma_i (Eq. 270)
VENTANA_ZSCORE_MESES = 60       # ~5 anios de historia propia de sigma para el z-score
COST_BPS = 10


def calcular_senales_low_vol(precios: pd.Series, ventana_vol: int, ventana_zscore: int) -> pd.DataFrame:
    ret = precios.pct_change().dropna()

    sigma = ret.rolling(ventana_vol).std(ddof=1)  # Eq. (270)
    media_sigma = sigma.rolling(ventana_zscore, min_periods=ventana_vol).mean()
    desv_sigma = sigma.rolling(ventana_zscore, min_periods=ventana_vol).std(ddof=1)
    z = (sigma - media_sigma) / desv_sigma

    return pd.DataFrame({"sigma": sigma, "z_score": z}).dropna()


def generar_posiciones(senales: pd.DataFrame) -> pd.Series:
    # signo invertido: baja volatilidad relativa (z negativo) -> largo
    posiciones = -np.sign(senales["z_score"])
    posiciones.name = "posicion"
    return posiciones


def main():
    precios = descargar_precios_mensuales(TICKER, start="2005-01-01")
    senales = calcular_senales_low_vol(precios, VENTANA_VOL_MESES, VENTANA_ZSCORE_MESES)
    posiciones = generar_posiciones(senales)

    resultado = backtest(precios, posiciones, holding=1, costo_bps=COST_BPS)

    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia")
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold")

    print(f"\n=== Estrategia 4: Low-volatility anomaly (adaptada) sobre {TICKER} ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['meses']} meses)\n")
    imprimir_metricas("Estrategia low-volatility (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia low-volatility (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 4 (Low-volatility anomaly) vs Buy & Hold — {TICKER}")
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

    ruta_png = os.path.join(carpeta_figuras, "strat_04_low_vol_NEE.png")
    ruta_csv = os.path.join(carpeta_resultados, "strat_04_low_vol_NEE.csv")
    fig.savefig(ruta_png, dpi=150)
    resultado.to_csv(ruta_csv)
    print(f"\nGrafico guardado en: {ruta_png}")
    print(f"Detalle mensual guardado en: {ruta_csv}")


if __name__ == "__main__":
    main()
