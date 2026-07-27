"""
Estrategia 2 (Seccion 3.2 del paper): Earnings-momentum.

Igual logica que price-momentum (comprar ganadores, vender perdedores),
pero el criterio de seleccion es el SUE (Standardized Unexpected Earnings):

    SUE_i = (E_i - E_i0) / sigma_i                     Eq. (274)

E_i:   EPS trimestral mas reciente reportado
E_i0:  EPS reportado hace 4 trimestres
sigma_i: desviacion estandar de (E_i - E_i0) en los ultimos 8 trimestres

ADAPTACION: el paper compra el decil superior de SUE (cross-sectional) y
vende el decil inferior. Con un solo activo se usa el signo de SUE:
SUE > 0 -> largo (sorpresa de utilidades positiva y creciente)
SUE < 0 -> corto

Holding period tipico en el paper: 6 meses. Aqui se mantiene la posicion
desde el mes siguiente al anuncio de utilidades hasta el siguiente anuncio.
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
from fundamentals_edgar import descargar_eps_trimestral_edgar, CIK_NEE

TICKER = "NEE"
N_TRIMESTRES_SIGMA = 8
COST_BPS = 10


def descargar_eps_trimestral(ticker: str) -> pd.Series:
    """
    EPS diluido trimestral desde SEC EDGAR (fuente oficial, historia
    completa desde que la empresa reporta en XBRL, ~2009+), en vez de
    yfinance (que solo da 4-5 trimestres).
    """
    return descargar_eps_trimestral_edgar(CIK_NEE)


def calcular_senales_sue(eps: pd.Series, n_trimestres_sigma: int) -> pd.DataFrame:
    """Eq. (274): SUE_i = (E_i - E_i0) / sigma_i"""
    sorpresa = eps.diff(4)  # E_i - E_i0 (4 trimestres atras)
    sigma = sorpresa.rolling(n_trimestres_sigma).std(ddof=1)
    sue = sorpresa / sigma

    return pd.DataFrame({
        "EPS": eps,
        "sorpresa_no_esperada": sorpresa,
        "sigma": sigma,
        "SUE": sue,
    }).dropna()


def generar_posiciones_mensuales(senales_trimestrales: pd.DataFrame, indice_mensual: pd.DatetimeIndex) -> pd.Series:
    """
    Convierte la senal trimestral (SUE, calculado en la fecha de anuncio) en
    una serie mensual: la senal de un trimestre se mantiene fija hasta el
    siguiente anuncio de utilidades (forward-fill), y se aplica desde el mes
    siguiente al anuncio (para no usar informacion antes de que se publique).
    """
    signo = np.sign(senales_trimestrales["SUE"])
    signo.index = signo.index + pd.DateOffset(days=1)  # disponible desde el dia siguiente al anuncio
    posiciones = signo.reindex(indice_mensual, method="ffill").fillna(0)
    posiciones.name = "posicion"
    return posiciones


def main():
    precios = descargar_precios_mensuales(TICKER, start="2005-01-01")
    eps = descargar_eps_trimestral(TICKER)
    senales = calcular_senales_sue(eps, n_trimestres_sigma=N_TRIMESTRES_SIGMA)
    posiciones = generar_posiciones_mensuales(senales, precios.index)

    resultado = backtest(precios, posiciones, holding=1, costo_bps=COST_BPS)

    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia")
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold")

    print(f"\n=== Estrategia 2: Earnings-momentum (SUE, adaptada) sobre {TICKER} ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['meses']} meses)")
    print(f"Trimestres de EPS disponibles: {len(eps)} | Señales SUE calculadas: {len(senales)}\n")
    imprimir_metricas("Estrategia earnings-momentum (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia earnings-momentum (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 2 (Earnings-momentum / SUE) vs Buy & Hold — {TICKER}")
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

    ruta_png = os.path.join(carpeta_figuras, "strat_02_earnings_momentum_NEE.png")
    ruta_csv = os.path.join(carpeta_resultados, "strat_02_earnings_momentum_NEE.csv")
    fig.savefig(ruta_png, dpi=150)
    resultado.to_csv(ruta_csv)
    print(f"\nGrafico guardado en: {ruta_png}")
    print(f"Detalle mensual guardado en: {ruta_csv}")


if __name__ == "__main__":
    main()
