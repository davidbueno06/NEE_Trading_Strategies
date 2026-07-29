"""
strat_19_market_making.py
----------------------------
Estrategia 19 (Seccion 3.19 del paper): Market-making.

ADVERTENCIA METODOLOGICA (leer antes de interpretar resultados, e
incluir una version de esto en el reporte final):
Esta NO es una implementacion de market-making real. Market-making real
requiere datos de nivel de libro (bid/ask, profundidad, prioridad en la
cola de ordenes) y opera en milisegundos/microsegundos. No existen datos
gratuitos con esa granularidad e historia larga, y el propio paper
reconoce que esto es esencialmente un negocio de HFT (velocidad,
infraestructura, cancel-replace).

Lo que se implementa aqui es un PROXY de barra diaria, honesto sobre sus
limitaciones, que recoge la idea central de la seccion 3.19: "modular la
senal de corto plazo con una senal de mas largo plazo" para intentar
capturar flujo "dumb" (no informado) y evitar flujo "smart"/toxico
(parrafo "Another possibility..." del paper):

  - Senal de reversion de muy corto plazo (1 dia): contraria al retorno
    de ayer -> intenta "comprar barato / vender caro" a corto plazo,
    como haria un market maker.
  - Senal de tendencia de mas largo plazo (cruce de medias moviles
    20/100 dias) -> evita tomar el lado equivocado de una tendencia
    sostenida (que es precisamente lo que penalizo a las estrategias
    long-short anteriores en NEE).
  - Solo se toma posicion cuando AMBAS coinciden en direccion. Si no
    coinciden, no se opera ese dia (se interpreta como posible flujo
    toxico / senal ambigua, y se evita).

MEJORA DE PRECISION vs. usar 10 bps fijos (como las demas 18 estrategias
del proyecto): el costo de cruzar el spread se estima dia a dia con el
estimador de Corwin & Schultz (2012) a partir de High/Low (ver
spread_estimator.py), en vez de un costo generico. Esto es mas apropiado
aqui porque, a diferencia de las estrategias que rebalancean una vez al
mes, esta es sensible dia a dia al costo de cruzar el spread, y ese
costo varia en el tiempo (mas ancho en episodios de estres/baja
liquidez, mas angosto en mercados tranquilos) -- un 10 bps fijo
subestimaria o sobreestimaria sistematicamente segun el periodo.

Fuente de precios: OHLC DIARIO (data.descargar_precios_diarios_ohlc), no
mensual -- el spread solo se puede estimar con High/Low diario, y una
senal de "corto plazo" no tiene sentido a frecuencia mensual.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import descargar_precios_diarios_ohlc
from spread_estimator import estimar_spread_corwin_schultz
from metrics import calcular_metricas, imprimir_metricas

TICKER = "NEE"
VENTANA_CORTA = 20
VENTANA_LARGA = 100
COSTO_MINIMO_BPS = 2.0  # piso de costo por operacion, por si el spread estimado sale ~0 en algun dia


def calcular_senales(precios: pd.DataFrame) -> pd.DataFrame:
    close = precios["Close"]

    ret_1d = close.pct_change()
    senal_reversion = np.sign(-ret_1d)

    ma_corta = close.rolling(VENTANA_CORTA).mean()
    ma_larga = close.rolling(VENTANA_LARGA).mean()
    senal_tendencia = np.sign(ma_corta - ma_larga)

    coincide = (senal_reversion == senal_tendencia) & (senal_reversion != 0)
    posicion = pd.Series(np.where(coincide, senal_tendencia, 0.0), index=precios.index)

    return pd.DataFrame({
        "senal_reversion": senal_reversion,
        "senal_tendencia": senal_tendencia,
        "posicion": posicion,
    })


def backtest_market_making(precios: pd.DataFrame, senales: pd.DataFrame) -> pd.DataFrame:
    close = precios["Close"]
    ret_diario = close.pct_change().dropna()

    spread = estimar_spread_corwin_schultz(precios).reindex(ret_diario.index)
    # aproximacion estandar en la literatura de costos de transaccion:
    # cruzar el spread cuesta ~ la mitad del spread cotizado por operacion
    costo_variable = (spread / 2).clip(lower=COSTO_MINIMO_BPS / 10000)
    costo_variable = costo_variable.fillna(costo_variable.median())

    pos_aplicada = senales["posicion"].shift(1).reindex(ret_diario.index).fillna(0)

    ret_bruto = pos_aplicada * ret_diario

    cambios = pos_aplicada.diff().abs().fillna(0)
    costos = cambios * costo_variable

    ret_neto = ret_bruto - costos

    resultado = pd.DataFrame({
        "retorno_diario_activo": ret_diario,
        "posicion": pos_aplicada,
        "spread_estimado": spread,
        "retorno_estrategia_bruto": ret_bruto,
        "retorno_estrategia_neto": ret_neto,
    })
    resultado["equity_estrategia"] = (1 + resultado["retorno_estrategia_neto"]).cumprod()
    resultado["equity_buy_and_hold"] = (1 + resultado["retorno_diario_activo"]).cumprod()
    return resultado


def main():
    precios = descargar_precios_diarios_ohlc(TICKER, start="2005-01-01")
    senales = calcular_senales(precios)
    resultado = backtest_market_making(precios, senales)

    metricas_estrategia = calcular_metricas(
        resultado, "retorno_estrategia_neto", "equity_estrategia", periods_per_year=252
    )
    metricas_bh = calcular_metricas(
        resultado, "retorno_diario_activo", "equity_buy_and_hold", periods_per_year=252
    )

    imprimir_metricas("Estrategia 19 - Market-making (proxy diario, costo variable Corwin-Schultz)", metricas_estrategia)
    imprimir_metricas("Buy & Hold NEE (diario)", metricas_bh)

    pct_dias_activos = (resultado["posicion"] != 0).mean()
    spread_prom_bps = resultado["spread_estimado"].mean() * 10000
    print(f"\n% de dias con posicion activa: {pct_dias_activos:.2%}")
    print(f"Spread estimado promedio (Corwin-Schultz): {spread_prom_bps:.1f} bps")
    print("\nADVERTENCIA: proxy de barra diaria, NO market-making real. "
          "Ver docstring del script y seccion correspondiente del reporte.")

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia 19 (Market-making proxy)")
    plt.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    plt.title("Estrategia 19: Market-making (proxy diario) vs Buy & Hold - NEE")
    plt.xlabel("Fecha")
    plt.ylabel("Valor de $1 invertido")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/strat_19_market_making.png", dpi=150)
    print("\nGrafico guardado en outputs/figures/strat_19_market_making.png")

    resultado.to_csv("outputs/results/strat_19_market_making.csv")
    print("Detalle diario guardado en outputs/results/strat_19_market_making.csv")


if __name__ == "__main__":
    main()
