"""
Estrategia 3 (Seccion 3.3 del paper): Value.

Criterio de seleccion: razon Book-to-Price (B/P), definida con el valor en
libros por accion (patrimonio contable / acciones en circulacion) sobre el
precio de mercado. El paper compra el decil superior de B/P (empresas
"baratas" en relacion a su valor contable) y vende el decil inferior
(empresas "caras").

ADAPTACION: sin universo de acciones para rankear, se compara el B/P actual
de NEE contra su propia historia (z-score de la ventana movil de los
ultimos N trimestres):

    z_t = (B/P_t - media_historica) / desviacion_historica

z_t > 0  -> NEE esta "barata" en relacion a su propia historia -> largo
z_t < 0  -> NEE esta "cara" en relacion a su propia historia -> corto

Holding period tipico en el paper: 1-6 meses. Aqui se mantiene la senal
del trimestre hasta el siguiente reporte de balance.
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
from fundamentals_edgar import descargar_book_value_trimestral_edgar, CIK_NEE

TICKER = "NEE"
VENTANA_ZSCORE_TRIMESTRES = 20  # ~5 anios de historia para la media/desv. propias
COST_BPS = 10


def descargar_book_value_trimestral(ticker: str) -> pd.Series:
    """
    Book value por accion desde SEC EDGAR (StockholdersEquity / acciones en
    circulacion, fuente oficial con historia completa), en vez de yfinance
    (que solo da 4-5 trimestres de balance).
    """
    return descargar_book_value_trimestral_edgar(CIK_NEE)


def calcular_senales_value(bvps: pd.Series, precios: pd.Series, ventana_trimestres: int) -> pd.DataFrame:
    precios_en_fechas_bvps = precios.reindex(bvps.index, method="ffill")
    bp = bvps / precios_en_fechas_bvps  # Book-to-Price

    media = bp.rolling(ventana_trimestres, min_periods=4).mean()
    desv = bp.rolling(ventana_trimestres, min_periods=4).std(ddof=1)
    z = (bp - media) / desv

    return pd.DataFrame({"BVPS": bvps, "BP": bp, "z_score": z}).dropna()


def generar_posiciones_mensuales(senales_trimestrales: pd.DataFrame, indice_mensual: pd.DatetimeIndex) -> pd.Series:
    signo = np.sign(senales_trimestrales["z_score"])
    signo.index = signo.index + pd.DateOffset(days=1)
    posiciones = signo.reindex(indice_mensual, method="ffill").fillna(0)
    posiciones.name = "posicion"
    return posiciones


def main():
    precios = descargar_precios_mensuales(TICKER, start="2005-01-01")
    bvps = descargar_book_value_trimestral(TICKER)
    senales = calcular_senales_value(bvps, precios, ventana_trimestres=VENTANA_ZSCORE_TRIMESTRES)
    posiciones = generar_posiciones_mensuales(senales, precios.index)

    resultado = backtest(precios, posiciones, holding=1, costo_bps=COST_BPS)

    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia")
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold")

    print(f"\n=== Estrategia 3: Value (B/P, adaptada) sobre {TICKER} ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['meses']} meses)")
    print(f"Trimestres con B/P disponible: {len(senales)}\n")
    imprimir_metricas("Estrategia value (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia value (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 3 (Value / B-P) vs Buy & Hold — {TICKER}")
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

    ruta_png = os.path.join(carpeta_figuras, "strat_03_value_NEE.png")
    ruta_csv = os.path.join(carpeta_resultados, "strat_03_value_NEE.csv")
    fig.savefig(ruta_png, dpi=150)
    resultado.to_csv(ruta_csv)
    print(f"\nGrafico guardado en: {ruta_png}")
    print(f"Detalle mensual guardado en: {ruta_csv}")


if __name__ == "__main__":
    main()
