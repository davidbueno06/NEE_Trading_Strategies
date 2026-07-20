"""
Estrategia 1 (Seccion 3.1 del paper "151 Trading Strategies",
Kakushadze & Serur, 2018): Price-momentum.

Aplicada a: NextEra Energy (NEE), componente del S&P 500.

ADAPTACION: el paper rankea un universo de N acciones (cross-sectional).
Con un solo activo se usa la version time-series: signo del retorno
acumulado de formacion (equivalente a Moskowitz-Ooi-Pedersen, 2012).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# permite importar los modulos comunes en src/ sin instalar un paquete
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import descargar_precios_mensuales
from metrics import calcular_metricas, imprimir_metricas
from backtest_engine import backtest

TICKER = "NEE"
T_FORMATION = 12
S_SKIP = 1
HOLDING = 1
COST_BPS = 10


def calcular_senales_momentum(precios: pd.Series, T: int, S: int) -> pd.DataFrame:
    """Eqs. (266)-(270) del paper."""
    ret = precios.pct_change().dropna()
    ret.name = "R"

    filas = []
    for i in range(T + S, len(ret) + 1):
        ventana_total = ret.iloc[i - (T + S): i]
        ventana_formacion = ventana_total.iloc[: T]

        R_mean = ventana_formacion.mean()
        sigma = ventana_formacion.std(ddof=1)
        R_riskadj = R_mean / sigma if sigma > 0 else np.nan
        R_cum = (1 + ventana_formacion).prod() - 1

        fecha_decision = ventana_total.index[-1]
        filas.append({
            "fecha": fecha_decision,
            "R_cum": R_cum,
            "R_mean": R_mean,
            "R_riskadj": R_riskadj,
            "sigma": sigma,
        })

    return pd.DataFrame(filas).set_index("fecha")


def generar_posiciones(senales: pd.DataFrame) -> pd.Series:
    posiciones = np.sign(senales["R_cum"]).fillna(0)
    posiciones.name = "posicion"
    return posiciones


def main():
    precios = descargar_precios_mensuales(TICKER, start="2005-01-01")
    senales = calcular_senales_momentum(precios, T=T_FORMATION, S=S_SKIP)
    posiciones = generar_posiciones(senales)
    resultado = backtest(precios, posiciones, holding=HOLDING, costo_bps=COST_BPS)

    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia")
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold")

    print(f"\n=== Estrategia 1: Price-momentum (adaptada, time-series) sobre {TICKER} ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['meses']} meses)\n")
    imprimir_metricas("Estrategia de momentum (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia momentum (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 1 (Price-momentum) vs Buy & Hold — {TICKER}")
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

    ruta_png = os.path.join(carpeta_figuras, "strat_01_momentum_NEE.png")
    ruta_csv = os.path.join(carpeta_resultados, "strat_01_momentum_NEE.csv")

    fig.savefig(ruta_png, dpi=150)
    resultado.to_csv(ruta_csv)
    print(f"\nGrafico guardado en: {ruta_png}")
    print(f"Detalle mensual guardado en: {ruta_csv}")


if __name__ == "__main__":
    main()
