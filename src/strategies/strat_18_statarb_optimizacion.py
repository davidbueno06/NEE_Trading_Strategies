"""
Estrategia 18 (Seccion 3.18): Statistical Arbitrage -- Optimizacion,
con Dollar-Neutrality (Seccion 3.18.1).

Extiende la Estrategia 9 (mean-reversion cluster, pesos simples
proporcionales a la desviacion demeaned) usando en su lugar una
OPTIMIZACION explicita de los pesos: en vez de apostar proporcional a
la desviacion de cada activo, se pondera por la matriz de covarianza
inversa del grupo (estilo Markowitz), lo que da mas peso a activos con
menor varianza/covarianza y menos peso a los mas ruidosos o muy
correlacionados entre si -- mismo signal direccional que la 9, pero con
un tamano de apuesta optimizado en vez de proporcional puro.

Grupo de activos: mismo que la Estrategia 9 -- NEE + DUK + SO (elegido
por el usuario, para poder comparar ambas variantes de mean-reversion
sobre el mismo grupo).

Metodologia (por mes t, usando SOLO datos hasta t, sin look-ahead):
  1. alpha_i = retorno logaritmico demeaned de cada activo en la ventana
     de formacion (igual que en la Estrategia 9, Eqs. 292-294).
  2. Sigma = matriz de covarianza de los retornos mensuales del grupo en
     la MISMA ventana de formacion, con shrinkage hacia la diagonal
     (para estabilizar la inversa con solo 3 activos y ventanas cortas).
  3. peso_raw = Sigma^-1 @ alpha  (direccion optima estilo Markowitz)
  4. Se proyecta peso_raw para que sea dollar-neutral exacto (se le resta
     su propia media, forzando suma = 0).
  5. Se normaliza la exposicion bruta a 1 (suma de |pesos| = 1).

Esto respeta dollar-neutrality (suma de pesos = 0) y gross exposure
normalizada (suma de |pesos| = 1), igual que en la Estrategia 9, pero el
"como" repartir el riesgo entre los 3 activos ya no es proporcional
ingenuo sino el resultado de una optimizacion de varianza.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import descargar_precios_mensuales
from metrics import calcular_metricas, imprimir_metricas
from multi_asset_reversion import backtest_multi_activo

TICKERS = ["NEE", "DUK", "SO"]
VENTANA_FORMACION = 12  # meses, igual orden de magnitud que Estrategia 9
COST_BPS = 10
PERIODS_PER_YEAR = 12
SHRINKAGE = 0.3  # 0 = covarianza muestral pura, 1 = solo diagonal (varianzas)


def _covarianza_con_shrinkage(retornos_ventana: pd.DataFrame, shrinkage: float) -> np.ndarray:
    sigma_muestral = retornos_ventana.cov().values
    diagonal = np.diag(np.diag(sigma_muestral))
    return (1 - shrinkage) * sigma_muestral + shrinkage * diagonal


def calcular_pesos_statarb_optimizado(precios: pd.DataFrame, ventana_formacion: int,
                                       shrinkage: float) -> pd.DataFrame:
    log_ret = np.log(precios / precios.shift(1)).dropna()

    fechas_pesos = []
    filas_pesos = []

    for i in range(ventana_formacion, len(log_ret) + 1):
        ventana = log_ret.iloc[i - ventana_formacion:i]
        fecha = ventana.index[-1]

        ret_acumulado = ventana.sum()  # Eq. (292), log-retorno acumulado en la ventana
        alpha = ret_acumulado - ret_acumulado.mean()  # Eq. (294), demeaned

        sigma = _covarianza_con_shrinkage(ventana, shrinkage)
        try:
            sigma_inv = np.linalg.inv(sigma + 1e-8 * np.eye(len(TICKERS)))
        except np.linalg.LinAlgError:
            continue

        peso_raw = sigma_inv @ (-alpha.values)  # signo negativo: igual convencion que Estrategia 9
                                                  # (corto al que se alejo "rico", largo al "barato")

        peso_dollar_neutral = peso_raw - peso_raw.mean()  # fuerza suma = 0

        suma_abs = np.abs(peso_dollar_neutral).sum()
        if suma_abs == 0 or not np.isfinite(suma_abs):
            continue
        peso_final = peso_dollar_neutral / suma_abs  # fuerza suma |peso| = 1

        fechas_pesos.append(fecha)
        filas_pesos.append(peso_final)

    pesos = pd.DataFrame(filas_pesos, index=fechas_pesos, columns=TICKERS)
    return pesos


def main():
    precios = {}
    for ticker in TICKERS:
        precios[ticker] = descargar_precios_mensuales(ticker, start="2005-01-01")
    df_precios = pd.DataFrame(precios).dropna()

    pesos = calcular_pesos_statarb_optimizado(df_precios, VENTANA_FORMACION, SHRINKAGE)

    resultado = backtest_multi_activo(df_precios, pesos, costo_bps=COST_BPS)
    m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia", periods_per_year=PERIODS_PER_YEAR)
    m_bh = calcular_metricas(resultado, "retorno_mensual_activo", "equity_buy_and_hold", periods_per_year=PERIODS_PER_YEAR)

    print(f"\n=== Estrategia 18: Statistical arbitrage -- optimizacion dollar-neutral ({'/'.join(TICKERS)}) ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({m_estrategia['periodos']} meses)")
    print(f"Ventana de formacion: {VENTANA_FORMACION} meses | Shrinkage covarianza: {SHRINKAGE}\n")
    imprimir_metricas("Estrategia stat-arb optimizada (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold equal-weight del grupo", m_bh)

    print("\nPesos promedio por activo (valor absoluto, para ver reparto de riesgo):")
    print(pesos.abs().mean())

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Stat-arb optimizada (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold equal-weight", linestyle="--")
    ax.set_title(f"Estrategia 18 (Statistical Arbitrage -- Optimizacion) — {'/'.join(TICKERS)}")
    ax.set_ylabel("Valor de $1 invertido")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    carpeta_figuras = os.path.join(carpeta_raiz, "outputs", "figures")
    carpeta_resultados = os.path.join(carpeta_raiz, "outputs", "results")
    os.makedirs(carpeta_figuras, exist_ok=True)
    os.makedirs(carpeta_resultados, exist_ok=True)
    fig.savefig(os.path.join(carpeta_figuras, "strat_18_statarb_optimizacion_NEE.png"), dpi=150)
    resultado.to_csv(os.path.join(carpeta_resultados, "strat_18_statarb_optimizacion_NEE.csv"))
    print(f"\nArchivos guardados en outputs/figures y outputs/results.")


if __name__ == "__main__":
    main()
