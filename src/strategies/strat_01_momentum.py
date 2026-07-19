"""
Estrategia 1 (Sección 3.1 del paper "151 Trading Strategies",
Kakushadze & Serur, 2018): Price-momentum.

Aplicada a: NextEra Energy (NEE), componente del S&P 500 (sector energia
renovable / utilities).

ADAPTACION NECESARIA:
----------------------
En el paper, la estrategia se define para un universo de N acciones: en cada
mes se ordenan las N acciones por su retorno acumulado de formacion (Ricum) y
se compra el decil superior (ganadoras) y se vende en corto el decil inferior
(perdedoras) -> es una estrategia CROSS-SECTIONAL (seccion transversal).

Con una sola accion (NEE) no existe "seccion transversal" contra la cual
rankear. La adaptacion estandar y honesta de este mismo factor a un solo
activo es la version TIME-SERIES del momentum (Moskowitz, Ooi & Pedersen,
2012, "Time Series Momentum", que es la extension natural de este mismo
efecto): en lugar de comparar NEE contra otras acciones, se compara el signo
(y magnitud) del retorno acumulado de formacion de NEE contra si mismo (contra
cero / contra su propia media historica). Las formulas (266)-(270) del paper
se mantienen identicas; lo unico que cambia es la regla de decision al final
(sort-and-decile -> señal de signo sobre un solo activo).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) PARAMETROS DE LA ESTRATEGIA (notacion identica al paper, Eqs. 266-270)
# ----------------------------------------------------------------------
TICKER = "NEE"
T_FORMATION = 12   # T: meses de la "formation period" (t=0 es el mes mas reciente)
S_SKIP = 1         # S: meses de "skip period" (se salta el mes mas reciente)
HOLDING = 1         # meses que se mantiene la posicion antes de re-evaluar
COST_BPS = 10       # costo de transaccion por operacion, en basis points (0.10%)


# ----------------------------------------------------------------------
# 2) DESCARGA DE DATOS (se ejecuta en tu maquina local, requiere yfinance)
# ----------------------------------------------------------------------
def descargar_precios_mensuales(ticker: str, start: str = "2005-01-01") -> pd.Series:
    """
    Descarga precios mensuales AJUSTADOS (por splits y dividendos) -> esto
    corresponde exactamente a Pi(t) en la Eq. (266)-(267) del paper, que pide
    'prices fully adjusted for splits and dividends'.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}.")

    # yfinance a veces devuelve columnas con multi-indice (Price, Ticker)
    # incluso pidiendo un solo ticker -> aplanamos para asegurar una Serie 1D.
    if isinstance(df.columns, pd.MultiIndex):
        cierre = df["Close"][ticker] if ticker in df["Close"].columns else df["Close"].iloc[:, 0]
    else:
        cierre = df["Close"]

    cierre = pd.Series(np.asarray(cierre).ravel(), index=df.index)

    precios_m = cierre.resample("ME").last()
    precios_m.name = ticker
    return precios_m.dropna()


# ----------------------------------------------------------------------
# 3) CALCULO DEL FACTOR DE MOMENTUM (Eqs. 266-270 del paper)
# ----------------------------------------------------------------------
def calcular_senales_momentum(precios: pd.Series, T: int, S: int) -> pd.DataFrame:
    """
    Para cada mes t0 (tratado como "t=0", el mes mas reciente de cada
    ventana), calcula:

      - retornos mensuales R_i(t)                      Eq. (266)
      - retorno acumulado del periodo de formacion Ricum Eq. (267)
      - retorno medio mensual Rimean                     Eq. (268)
      - retorno ajustado por riesgo Ririsk.adj            Eq. (269)
      - volatilidad mensual sigma_i                       Eq. (270)

    saltando el mes mas reciente (S) y usando T meses de formacion.
    """
    ret = precios.pct_change().dropna()
    ret.name = "R"

    filas = []
    # ventana total necesaria: T meses de formacion + S meses de skip
    for i in range(T + S, len(ret) + 1):
        ventana_total = ret.iloc[i - (T + S): i]          # incluye skip
        ventana_formacion = ventana_total.iloc[: T]         # excluye los ultimos S meses (skip)

        R_mean = ventana_formacion.mean()                  # Eq. (268)
        sigma = ventana_formacion.std(ddof=1)               # Eq. (270)
        R_riskadj = R_mean / sigma if sigma > 0 else np.nan  # Eq. (269)

        # Ricum: retorno acumulado compuesto durante la formation period, Eq. (267)
        R_cum = (1 + ventana_formacion).prod() - 1

        fecha_decision = ventana_total.index[-1]  # fin del skip period = mes en que se decide
        filas.append({
            "fecha": fecha_decision,
            "R_cum": R_cum,
            "R_mean": R_mean,
            "R_riskadj": R_riskadj,
            "sigma": sigma,
        })

    return pd.DataFrame(filas).set_index("fecha")


# ----------------------------------------------------------------------
# 4) REGLA DE DECISION (adaptacion a un solo activo)
# ----------------------------------------------------------------------
def generar_posiciones(senales: pd.DataFrame) -> pd.Series:
    """
    Version long/short basada en el SIGNO de R_cum (equivalente, para un solo
    activo, a "comprar si esta en el decil ganador / vender si esta en el
    decil perdedor" del paper):

        R_cum > 0  -> posicion larga  (+1)
        R_cum < 0  -> posicion corta  (-1)
        R_cum = 0  -> sin posicion    ( 0)
    """
    posiciones = np.sign(senales["R_cum"]).fillna(0)
    posiciones.name = "posicion"
    return posiciones


# ----------------------------------------------------------------------
# 5) BACKTEST (long-short, rebalanceo mensual, holding period H meses)
# ----------------------------------------------------------------------
def backtest(precios: pd.Series, posiciones: pd.Series, holding: int, costo_bps: float):
    ret_mensual = precios.pct_change().dropna()

    # la posicion decidida en el mes t se APLICA al retorno del mes t+1
    # (evita look-ahead bias: no se usa informacion futura)
    pos_aplicada = posiciones.shift(1).reindex(ret_mensual.index)
    if holding > 1:
        pos_aplicada = pos_aplicada.ffill(limit=holding - 1)
    pos_aplicada = pos_aplicada.fillna(0)

    ret_estrategia_bruto = pos_aplicada * ret_mensual

    # costos de transaccion: se paga cada vez que la posicion cambia
    cambios = pos_aplicada.diff().abs().fillna(0)
    costos = cambios * (costo_bps / 10000)
    ret_estrategia_neto = ret_estrategia_bruto - costos

    resultado = pd.DataFrame({
        "retorno_mensual_activo": ret_mensual,
        "posicion": pos_aplicada,
        "retorno_estrategia_bruto": ret_estrategia_bruto,
        "retorno_estrategia_neto": ret_estrategia_neto,
    })
    resultado["equity_estrategia"] = (1 + resultado["retorno_estrategia_neto"]).cumprod()
    resultado["equity_buy_and_hold"] = (1 + resultado["retorno_mensual_activo"]).cumprod()
    return resultado


# ----------------------------------------------------------------------
# 6) METRICAS DE DESEMPENO
# ----------------------------------------------------------------------
def calcular_metricas(resultado: pd.DataFrame, columna_retorno: str) -> dict:
    r = resultado[columna_retorno].dropna()
    n_meses = len(r)
    ret_total = resultado[f"equity_{'estrategia' if 'estrategia' in columna_retorno else 'buy_and_hold'}"].iloc[-1] - 1
    ret_anual = (1 + ret_total) ** (12 / n_meses) - 1 if n_meses > 0 else np.nan
    vol_anual = r.std(ddof=1) * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol_anual if vol_anual > 0 else np.nan

    equity = (1 + r).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "retorno_total": ret_total,
        "retorno_anualizado": ret_anual,
        "volatilidad_anualizada": vol_anual,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "meses": n_meses,
    }


# ----------------------------------------------------------------------
# 7) EJECUCION PRINCIPAL
# ----------------------------------------------------------------------
def main():
    precios = descargar_precios_mensuales(TICKER, start="2005-01-01")
    senales = calcular_senales_momentum(precios, T=T_FORMATION, S=S_SKIP)
    posiciones = generar_posiciones(senales)
    resultado = backtest(precios, posiciones, holding=HOLDING, costo_bps=COST_BPS)

    metricas_estrategia = calcular_metricas(resultado, "retorno_estrategia_neto")
    metricas_bh = calcular_metricas(resultado, "retorno_mensual_activo")

    print(f"\n=== Estrategia 1: Price-momentum (adaptada, time-series) sobre {TICKER} ===")
    print(f"Periodo: {resultado.index[0].date()} a {resultado.index[-1].date()}  ({metricas_estrategia['meses']} meses)\n")

    print(">> Estrategia de momentum (neta de costos):")
    for k, v in metricas_estrategia.items():
        if k != "meses":
            print(f"   {k:25s}: {v:.4f}")

    print("\n>> Buy & Hold (comprar y mantener NEE):")
    for k, v in metricas_bh.items():
        if k != "meses":
            print(f"   {k:25s}: {v:.4f}")

    # Grafico
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado.index, resultado["equity_estrategia"], label="Estrategia momentum (neta)")
    ax.plot(resultado.index, resultado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 1 (Price-momentum) vs Buy & Hold — {TICKER}")
    ax.set_ylabel("Valor de $1 invertido")
    ax.set_xlabel("Fecha")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    carpeta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_png = os.path.join(carpeta_script, "backtest_momentum_NEE.png")
    ruta_csv = os.path.join(carpeta_script, "resultado_backtest_NEE.csv")

    fig.savefig(ruta_png, dpi=150)
    print(f"\nGrafico guardado en: {ruta_png}")

    resultado.to_csv(ruta_csv)
    print(f"Detalle mensual guardado en: {ruta_csv}")


if __name__ == "__main__":
    main()