"""
strat_06_multifactor.py
--------------------------
Estrategia 6 (Seccion 3.6 del paper): Multifactor portfolio.

Combina los "factores" clasicos YA implementados en este proyecto
(Estrategias 1, 2, 3, 4, 7 -- momentum, earnings-momentum, value,
low-volatility, residual momentum) con pesos w_A (Eq. 275 del paper).
Se diferencia de la Estrategia 20 (Alpha combos) en dos cosas:
  1) Aqui solo se combinan los 5 factores "clasicos" de estilo, no TODAS
     las estrategias del proyecto (Estrategia 20 tambien mete tecnicas
     de precio y stat-arb).
  2) El esquema de pesos es mucho mas simple (uniforme o inverso a la
     volatilidad), en vez del procedimiento de regresion de 11 pasos de
     la Estrategia 20.

DOS ESQUEMAS DE PESOS (ambos mencionados en el paper, Seccion 3.6 y
nota al pie 47):
  - Uniforme: w_A = 1/F (Eq. 275) -- la opcion "simple" que el propio
    paper describe primero.
  - Inverso a la volatilidad: w_A ~ 1/sigma_A (nota 47) -- se calcula
    con volatilidad TRAILING de 36 meses (solo datos pasados, para no
    tener look-ahead) y se re-normaliza cada mes.

REQUISITO: correr DESPUES de las Estrategias 1, 2, 3, 4 y 7 (lee sus
CSVs en outputs/results/, igual que hace la Estrategia 20). Si falta
alguno de esos 5 archivos, la estrategia se combina igual con los que
esten disponibles (minimo 2), y se avisa cuales faltaron.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import descargar_precios_mensuales
from metrics import calcular_metricas, imprimir_metricas

TICKER = "NEE"
CARPETA_RESULTADOS = "outputs/results"
VENTANA_VOL = 36  # meses, para el esquema inverso a la volatilidad

# Los 5 "factores" clasicos de la Estrategia 6, con sus archivos de
# resultados esperados (ajustar aqui si en tu repo tienen otro nombre)
FACTORES = {
    "strat_01_momentum_NEE": "Momentum",
    "strat_02_earnings_momentum_NEE": "Earnings-momentum",
    "strat_03_value_NEE": "Value",
    "strat_04_low_vol_NEE": "Low-volatility",
    "strat_07_residual_momentum_NEE": "Residual momentum",
}


def cargar_retornos_factores() -> pd.DataFrame:
    series = {}
    for nombre in FACTORES:
        ruta = os.path.join(CARPETA_RESULTADOS, f"{nombre}.csv")
        if not os.path.exists(ruta):
            print(f"  AVISO: no se encontro {ruta}, se omite ese factor.")
            continue
        df = pd.read_csv(ruta, index_col=0, parse_dates=True)
        if "retorno_estrategia_neto" not in df.columns:
            print(f"  AVISO: {nombre} sin columna 'retorno_estrategia_neto', se omite.")
            continue
        ret = df["retorno_estrategia_neto"].dropna()
        ret.name = nombre
        series[nombre] = ret

    if len(series) < 2:
        raise FileNotFoundError(
            "Se necesitan al menos 2 estrategias factor disponibles en "
            f"{CARPETA_RESULTADOS} para construir el portafolio multifactor."
        )
    return pd.concat(series.values(), axis=1)


def pesos_uniformes(retornos_factores: pd.DataFrame) -> pd.DataFrame:
    F = retornos_factores.shape[1]
    return pd.DataFrame(1.0 / F, index=retornos_factores.index, columns=retornos_factores.columns)


def pesos_inverso_vol(retornos_factores: pd.DataFrame, ventana: int = VENTANA_VOL) -> pd.DataFrame:
    vol_trailing = retornos_factores.rolling(ventana).std()
    inv_vol = (1.0 / vol_trailing).replace([np.inf, -np.inf], np.nan)
    pesos = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return pesos


def backtest_combo(retornos_factores: pd.DataFrame, pesos: pd.DataFrame, costo_bps: float = 10) -> pd.DataFrame:
    pesos_aplicados = pesos.shift(1).reindex(retornos_factores.index).fillna(0)
    ret_bruto = (pesos_aplicados * retornos_factores.fillna(0)).sum(axis=1)

    cambios = pesos_aplicados.diff().abs().sum(axis=1).fillna(0)
    costos = cambios * (costo_bps / 10000)
    ret_neto = ret_bruto - costos

    resultado = pd.DataFrame({
        "retorno_estrategia_bruto": ret_bruto,
        "retorno_estrategia_neto": ret_neto,
    })
    resultado["equity_estrategia"] = (1 + resultado["retorno_estrategia_neto"]).cumprod()
    return resultado


def main():
    print("Cargando retornos de estrategias factor (1, 2, 3, 4, 7)...")
    retornos_factores = cargar_retornos_factores()
    print(f"Factores incluidos ({retornos_factores.shape[1]}): {list(retornos_factores.columns)}")

    precios_nee = descargar_precios_mensuales(TICKER, start="2005-01-01")
    ret_bh = precios_nee.pct_change().dropna()

    resultados = {}
    plt.figure(figsize=(10, 6))

    for nombre_esquema, funcion_pesos in [("uniforme", pesos_uniformes), ("inverso_vol", pesos_inverso_vol)]:
        pesos = funcion_pesos(retornos_factores)
        resultado = backtest_combo(retornos_factores, pesos)

        idx_comun = resultado.index.intersection(ret_bh.index)
        resultado = resultado.loc[idx_comun].copy()
        resultado["retorno_mensual_activo"] = ret_bh.reindex(idx_comun)
        resultado["equity_buy_and_hold"] = (1 + resultado["retorno_mensual_activo"]).cumprod()
        resultados[nombre_esquema] = resultado

        m_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto", "equity_estrategia", periods_per_year=12)
        imprimir_metricas(f"Estrategia 6 - Multifactor ({nombre_esquema})", m_estrategia)

        plt.plot(resultado.index, resultado["equity_estrategia"], label=f"Multifactor ({nombre_esquema})")

    m_bh = calcular_metricas(resultados["uniforme"], "retorno_mensual_activo", "equity_buy_and_hold", periods_per_year=12)
    imprimir_metricas("Buy & Hold NEE", m_bh)

    plt.plot(
        resultados["uniforme"].index, resultados["uniforme"]["equity_buy_and_hold"],
        label="Buy & Hold NEE", linestyle="--", color="black",
    )
    plt.title("Estrategia 6: Multifactor portfolio vs Buy & Hold - NEE")
    plt.xlabel("Fecha")
    plt.ylabel("Valor de $1 invertido")
    plt.legend()
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    plt.savefig("outputs/figures/strat_06_multifactor.png", dpi=150)
    print("\nGrafico guardado en outputs/figures/strat_06_multifactor.png")

    for nombre_esquema, resultado in resultados.items():
        resultado.to_csv(f"outputs/results/strat_06_multifactor_{nombre_esquema}.csv")
    print("CSVs guardados en outputs/results/ (uno por esquema de pesos)")


if __name__ == "__main__":
    main()
