"""
Estrategia 7 (Seccion 3.7 del paper): Residual momentum.

Igual que price-momentum, pero en vez de usar el retorno crudo de NEE,
se usa el RESIDUO de una regresion de NEE contra los 3 factores de
Fama-French (mercado, tamano, valor):

    R_i(t) = alpha_i + b1*MKT(t) + b2*SMB(t) + b3*HML(t) + eps_i(t)   Eq. (278)

Interpretacion de la metodologia (el texto original es ambiguo en los
detalles exactos de las ventanas, aqui se documenta la interpretacion
usada):
  1) Se estiman alpha, b1, b2, b3 con una regresion sobre una ventana de
     36 meses (Blitz, Huij & Martens 2011), terminando 13 meses antes de
     la fecha de decision (1 mes de skip + 12 meses de formacion).
  2) Con esos coeficientes (sin el alpha, Eq. 279) se calculan los
     residuos eps_i(t) en los 12 meses MAS RECIENTES antes del skip:

        eps_i(t) = R_i(t) - b1*MKT(t) - b2*SMB(t) - b3*HML(t)          Eq. (279)

  3) Se calcula el retorno residual ajustado por riesgo (Eqs. 280-282):

        eps_mean = promedio(eps) en la ventana de formacion
        sigma_eps = desviacion estandar de eps en la ventana
        R_riskadj = eps_mean / sigma_eps

ADAPTACION a un solo activo: igual que en la Estrategia 1, se usa el
signo de R_riskadj en vez de comprar/vender deciles de un universo.
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
from factors_famafrench import descargar_factores_ff3_mensuales

TICKER = "NEE"
VENTANA_REGRESION = 36   # meses para estimar alpha, b1, b2, b3
VENTANA_FORMACION = 12   # meses para calcular los residuos / la señal
SKIP = 1                 # mes de skip antes de la fecha de decision
COST_BPS = 10


def _ols_con_intercepto(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS simple con intercepto, sin dependencias externas (numpy puro)."""
    X_full = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X_full, y, rcond=None)
    return coef  # [alpha, b1, b2, b3]


def calcular_senales_residual_momentum(
    retornos_nee: pd.Series, factores: pd.DataFrame,
    ventana_regresion: int, ventana_formacion: int, skip: int,
) -> pd.DataFrame:
    df = pd.DataFrame({"R": retornos_nee}).join(factores[["Mkt-RF", "SMB", "HML", "RF"]], how="inner")
    df["R_exceso"] = df["R"] - df["RF"]

    filas = []
    total_necesario = ventana_regresion + skip + ventana_formacion
    for i in range(total_necesario, len(df) + 1):
        bloque = df.iloc[i - total_necesario: i]

        ventana_reg = bloque.iloc[: ventana_regresion]
        ventana_form = bloque.iloc[ventana_regresion + skip:]  # ultimos 'ventana_formacion' meses, tras el skip

        y_reg = ventana_reg["R_exceso"].values
        X_reg = ventana_reg[["Mkt-RF", "SMB", "HML"]].values
        alpha, b1, b2, b3 = _ols_con_intercepto(y_reg, X_reg)

        X_form = ventana_form[["Mkt-RF", "SMB", "HML"]].values
        residuos = ventana_form["R_exceso"].values - (b1 * X_form[:, 0] + b2 * X_form[:, 1] + b3 * X_form[:, 2])

        eps_mean = residuos.mean()
        sigma_eps = residuos.std(ddof=1)
        r_riskadj = eps_mean / sigma_eps if sigma_eps > 0 else np.nan

        fecha_decision = bloque.index[-1]
        filas.append({"fecha": fecha_decision, "eps_mean": eps_mean, "sigma_eps": sigma_eps, "R_riskadj": r_riskadj})

    return pd.DataFrame(filas).set_index("fecha")


def generar_posiciones(senales: pd.DataFrame) -> pd.Series:
    posiciones = np.sign(senales["R_riskadj"]).fillna(0)
    posiciones.name = "posicion"
    return posiciones


def main():
    precios = descargar_precios_mensuales(TICKER, start="2000-01-01")
    retornos = precios.pct_change().dropna()
    factores = descargar_factores_ff3_mensuales()

    senales = calcular_senales_residual_momentum(
        retornos, factores, VENTANA_REGRESION, VENTANA_FORMACION, SKIP
    )
    posiciones = generar_posiciones(senales)
    resultado = backtest(precios, posiciones, holding=1, costo_bps=COST_BPS)

    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia")
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold")

    print(f"\n=== Estrategia 7: Residual momentum (Fama-French 3F) sobre {TICKER} ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['meses']} meses)\n")
    imprimir_metricas("Estrategia residual momentum (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold", m_bh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia residual momentum (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 7 (Residual momentum) vs Buy & Hold — {TICKER}")
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
    ruta_png = os.path.join(carpeta_figuras, "strat_07_residual_momentum_NEE.png")
    ruta_csv = os.path.join(carpeta_resultados, "strat_07_residual_momentum_NEE.csv")
    fig.savefig(ruta_png, dpi=150)
    resultado.to_csv(ruta_csv)
    print(f"\nGrafico guardado en: {ruta_png}")
    print(f"Detalle mensual guardado en: {ruta_csv}")


if __name__ == "__main__":
    main()
