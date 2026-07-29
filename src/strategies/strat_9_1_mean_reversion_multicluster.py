"""
strat_21_mean_reversion_multicluster.py
------------------------------------------
Estrategia 21 (Seccion 3.9.1 del paper): Mean-reversion - multiple
clusters. Generalizacion de la Estrategia 9 a K=2 clusters de utilities
correlacionadas, usando multi_cluster_reversion.py. Reutiliza el motor de
backtest de la Estrategia 9 (backtest_multi_activo en
multi_asset_reversion.py) sin modificarlo, porque la mecanica de aplicar
pesos/costos/equity es identica.

CLUSTERS (decision explicita, cambiable si no te convence):
  Cluster A ("Sureste EE.UU. / grandes con sesgo renovable"):
    NEE, DUK, SO  -> el mismo cluster ya usado en la Estrategia 9.
  Cluster B ("Utilities diversificadas, otras regiones"):
    XEL, WEC, AEP (Xcel Energy, WEC Energy Group, American Electric Power)
    -> anadido para tener K=2 y probar la generalizacion multi-cluster.
    Son utilities reguladas de gran capitalizacion, historicamente
    correlacionadas entre si por exposicion comun a tasas de interes y
    regulacion de servicios publicos, pero en geografias distintas al
    Cluster A -- para que no sea trivialmente "el mismo grupo con otro
    nombre" y la generalizacion a multi-cluster tenga contenido real.

CEG (usado en la Estrategia 8, pairs trading) NO se incluye aqui: cotiza
publicamente solo desde 2022, y meterlo recortaria la historia disponible
de 2005-2026 a apenas ~4 anios para TODA la estrategia (no solo para CEG,
porque el backtest necesita que todos los activos tengan datos en el
mismo periodo).

BENCHMARKS: se reportan DOS, porque la Estrategia 9 usaba equal-weight
del grupo como buy & hold (heredado de multi_asset_reversion.py), pero
la convencion general del proyecto (ver README) es comparar siempre
contra buy & hold de NEE especificamente. Se muestran ambos para que la
comparacion sea honesta en los dos sentidos.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt

from data import descargar_precios_mensuales
from multi_cluster_reversion import calcular_pesos_multicluster
from multi_asset_reversion import backtest_multi_activo
from metrics import calcular_metricas, imprimir_metricas

TICKERS_CLUSTER_A = ["NEE", "DUK", "SO"]
TICKERS_CLUSTER_B = ["XEL", "WEC", "AEP"]
CLUSTERS = {t: "A" for t in TICKERS_CLUSTER_A}
CLUSTERS.update({t: "B" for t in TICKERS_CLUSTER_B})

VENTANA_FORMACION = 1  # mismo criterio que Estrategia 9: 1 mes de formacion


def main():
    precios = {}
    for ticker in CLUSTERS:
        precios[ticker] = descargar_precios_mensuales(ticker, start="2005-01-01")
    precios_df = pd.DataFrame(precios).dropna()

    print(f"Periodo con datos completos para los 6 activos: "
          f"{precios_df.index.min().date()} a {precios_df.index.max().date()} "
          f"({len(precios_df)} meses)")

    pesos = calcular_pesos_multicluster(precios_df, CLUSTERS, VENTANA_FORMACION)
    resultado = backtest_multi_activo(precios_df, pesos, costo_bps=10)

    metricas_estrategia = calcular_metricas(
        resultado, "retorno_estrategia_neto", "equity_estrategia", periods_per_year=12
    )
    metricas_bh_basket = calcular_metricas(
        resultado, "retorno_mensual_activo", "equity_buy_and_hold", periods_per_year=12
    )
    imprimir_metricas("Estrategia 21 - Mean-reversion multi-cluster (A+B)", metricas_estrategia)
    imprimir_metricas("Buy & Hold equal-weight (los 6 activos)", metricas_bh_basket)

    # Benchmark adicional: NEE solo, para mantener la convencion general
    # del proyecto de comparar siempre contra buy & hold NEE especificamente.
    ret_nee = precios_df["NEE"].pct_change().dropna()
    idx_comun = resultado.index.intersection(ret_nee.index)
    equity_nee = (1 + ret_nee.loc[idx_comun]).cumprod()
    metricas_bh_nee = calcular_metricas(
        pd.DataFrame({"r": ret_nee.loc[idx_comun], "eq": equity_nee}),
        "r", "eq", periods_per_year=12,
    )
    imprimir_metricas("Buy & Hold NEE (referencia general del proyecto)", metricas_bh_nee)

    print("\nPeso promedio (valor absoluto) por activo:")
    print(pesos.abs().mean().sort_values(ascending=False).round(4))

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia 21 (multi-cluster A+B)")
    plt.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold equal-weight (6 activos)", linestyle="--")
    plt.plot(idx_comun, equity_nee, label="Buy & Hold NEE", linestyle=":")
    plt.title("Estrategia 21: Mean-reversion multi-cluster vs benchmarks")
    plt.xlabel("Fecha")
    plt.ylabel("Valor de $1 invertido")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/strat_21_mean_reversion_multicluster.png", dpi=150)
    print("\nGrafico guardado en outputs/figures/strat_21_mean_reversion_multicluster.png")

    resultado.to_csv("outputs/results/strat_21_mean_reversion_multicluster.csv")
    print("Detalle mensual guardado en outputs/results/strat_21_mean_reversion_multicluster.csv")


if __name__ == "__main__":
    main()
