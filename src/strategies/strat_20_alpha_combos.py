"""
strat_20_alpha_combos.py
---------------------------
Estrategia 20 (Seccion 3.20 del paper): Alpha combos.

Combina las N estrategias YA implementadas y corridas en este proyecto
(cada una tratada como un "alpha" en el sentido del paper -- ver
docstring de alpha_combos.py para la adaptacion a un solo activo) en un
unico "mega-alpha", usando el procedimiento de 11 pasos de Kakushadze &
Yu (2017b) citado en la Seccion 3.20, aplicado en ventana movil para
evitar look-ahead bias.

REQUISITO: este script debe correrse DESPUES de haber corrido las demas
estrategias, porque lee sus resultados guardados en outputs/results/.
No vuelve a descargar precios ni recalcula senales de cada estrategia.

QUE ESTRATEGIAS SE INCLUYEN:
Se incluye automaticamente cualquier outputs/results/strat_NN_*.csv que
tenga una columna 'retorno_estrategia_neto', EXCEPTO:
  - strat_20_*.csv (esta misma estrategia, para evitar auto-referencia)
  - las que se listen explicitamente en EXCLUIR_DE_LA_LISTA abajo

Por default se deja fuera la Estrategia 19 (market-making proxy) del
combo: es un proxy de barra diaria con supuestos fuertes (ver docstring
de strat_19_market_making.py), y mezclarlo sin mas con estrategias
mensuales "normales" le daria un peso implicito no justificado dentro
del combo. Se puede incluir cambiando INCLUIR_ESTRATEGIA_19 a True una
vez que se haya evaluado si su resultado es razonable.

MANEJO DE FRECUENCIA MIXTA (mensual vs diaria):
Las estrategias 1-9 y 15+ (salvo 10-14 y 19) son mensuales. Las
estrategias 10-14 (medias moviles, soporte/resistencia, canal) y 19
(market-making) son diarias. Para combinarlas todas junto, aqui se
reconstruye la curva de equity de cada estrategia a partir de sus
retornos netos diarios/mensuales guardados, y se remuestrea TODO a
frecuencia mensual (fin de mes) antes de correr el procedimiento de
Alpha combos -- que, siguiendo el resto del proyecto, opera a frecuencia
mensual. Esto es una decision metodologica explicita: se pierde
granularidad de las estrategias diarias, pero mezclar frecuencias
directamente en el procedimiento de pesos no seria valido.
"""

import sys
import os
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import descargar_precios_mensuales
from metrics import calcular_metricas, imprimir_metricas
from alpha_combos import calcular_pesos_alpha_combo, backtest_alpha_combo

TICKER = "NEE"
CARPETA_RESULTADOS = "outputs/results"
VENTANA_M = 36          # meses de historia usados para fijar pesos en cada punto del tiempo
DIAS_PROMEDIO_E = 12    # meses usados para el retorno esperado E_i (paso 8, Eq. 360)
MIN_ALPHAS_ACTIVOS = 3  # minimo de estrategias con historia suficiente para calcular pesos

INCLUIR_ESTRATEGIA_19 = False
EXCLUIR_DE_LA_LISTA = []  # ej. ["strat_05_implied_vol"] si se decide omitir la 5


def _nombre_estrategia(ruta_csv: str) -> str:
    base = os.path.basename(ruta_csv)
    return os.path.splitext(base)[0]


def cargar_retornos_estrategias(carpeta: str = CARPETA_RESULTADOS) -> pd.DataFrame:
    """
    Lee todos los CSV de estrategias ya corridas, reconstruye su curva de
    equity neta y la remuestrea a retornos MENSUALES (fin de mes), sin
    importar si el CSV original era diario o mensual.
    """
    rutas = sorted(glob.glob(os.path.join(carpeta, "strat_*.csv")))
    series_mensuales = {}

    for ruta in rutas:
        nombre = _nombre_estrategia(ruta)

        if nombre.startswith("strat_20"):
            continue
        if not INCLUIR_ESTRATEGIA_19 and nombre.startswith("strat_19"):
            print(f"  (omitida de la lista de alphas por config: {nombre})")
            continue
        if nombre in EXCLUIR_DE_LA_LISTA:
            print(f"  (excluida explicitamente: {nombre})")
            continue

        df = pd.read_csv(ruta, index_col=0, parse_dates=True)
        if "retorno_estrategia_neto" not in df.columns:
            print(f"  (ignorada, sin columna 'retorno_estrategia_neto': {nombre})")
            continue

        ret = df["retorno_estrategia_neto"].dropna()
        if ret.empty:
            continue

        equity = (1 + ret).cumprod()
        equity_m = equity.resample("ME").last()
        ret_m = equity_m.pct_change().dropna()
        ret_m.name = nombre
        series_mensuales[nombre] = ret_m

    if not series_mensuales:
        raise FileNotFoundError(
            f"No se encontraron resultados validos en {carpeta}. "
            "Corre las estrategias 1-19 (las que apliquen) antes de la 20."
        )

    retornos_alphas = pd.concat(series_mensuales.values(), axis=1)
    retornos_alphas.columns = list(series_mensuales.keys())
    return retornos_alphas.sort_index()


def main():
    print("Cargando resultados de estrategias previas...")
    retornos_alphas = cargar_retornos_estrategias()
    print(f"\nEstrategias incluidas en el combo ({retornos_alphas.shape[1]}):")
    for col in retornos_alphas.columns:
        n_obs = retornos_alphas[col].notna().sum()
        print(f"  - {col}: {n_obs} meses con dato")

    if retornos_alphas.shape[1] < MIN_ALPHAS_ACTIVOS:
        raise ValueError(
            f"Solo hay {retornos_alphas.shape[1]} estrategias disponibles; "
            f"se necesitan al menos {MIN_ALPHAS_ACTIVOS} para el combo. "
            "Corre mas estrategias antes de continuar con la 20."
        )

    print(f"\nCalculando pesos del combo (ventana movil de {VENTANA_M} meses, sin look-ahead)...")
    pesos = calcular_pesos_alpha_combo(
        retornos_alphas,
        ventana_M=VENTANA_M,
        dias_promedio_E=DIAS_PROMEDIO_E,
        min_alphas_activos=MIN_ALPHAS_ACTIVOS,
    )

    resultado = backtest_alpha_combo(retornos_alphas, pesos, costo_bps=10)

    precios_nee = descargar_precios_mensuales(TICKER, start="2005-01-01")
    ret_bh = precios_nee.pct_change().dropna()
    equity_bh = (1 + ret_bh).cumprod()

    idx_comun = resultado.index.intersection(equity_bh.index)
    resultado = resultado.loc[idx_comun]
    resultado["retorno_mensual_activo"] = ret_bh.reindex(idx_comun)
    resultado["equity_buy_and_hold"] = (1 + resultado["retorno_mensual_activo"]).cumprod()

    metricas_estrategia = calcular_metricas(
        resultado, "retorno_estrategia_neto", "equity_estrategia", periods_per_year=12
    )
    metricas_bh = calcular_metricas(
        resultado, "retorno_mensual_activo", "equity_buy_and_hold", periods_per_year=12
    )

    imprimir_metricas("Estrategia 20 - Alpha combos", metricas_estrategia)
    imprimir_metricas("Buy & Hold NEE", metricas_bh)

    print("\nPeso promedio (valor absoluto) por estrategia dentro del combo:")
    print(pesos.abs().mean().sort_values(ascending=False).round(4))

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia 20 (Alpha combo)")
    plt.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    plt.title("Estrategia 20: Alpha combos vs Buy & Hold - NEE")
    plt.xlabel("Fecha")
    plt.ylabel("Valor de $1 invertido")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/strat_20_alpha_combos.png", dpi=150)
    print("\nGrafico guardado en outputs/figures/strat_20_alpha_combos.png")

    resultado.to_csv("outputs/results/strat_20_alpha_combos.csv")
    pesos.to_csv("outputs/results/strat_20_pesos.csv")
    print("Detalle mensual y pesos guardados en outputs/results/")


if __name__ == "__main__":
    main()
