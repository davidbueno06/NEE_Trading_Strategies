"""
Estrategia 17 (Seccion 3.17): Machine Learning -- K-Nearest Neighbors.

Adaptacion a un solo activo: KNN como clasificador binario que predice el
signo del retorno del dia siguiente de NEE, usando features tecnicos
calculados solo con informacion disponible hasta el dia t (sin
look-ahead). El modelo se reentrena en WALK-FORWARD (ventana de
entrenamiento expansiva, nunca usa datos futuros):

  - Ventana minima de entrenamiento: 756 dias habiles (~3 anos), elegida
    por el usuario.
  - Reentrenamiento: cada trimestre (~63 dias habiles), elegido por el
    usuario. Entre reentrenamientos, el modelo entrenado se usa tal cual
    para predecir dia a dia (out-of-sample).

Features (todos calculados con datos <= dia t, para no usar el retorno
que se quiere predecir):
  - retornos rezagados de 1, 5, 10 y 20 dias
  - volatilidad realizada de 20 dias
  - distancia relativa (P/SMA - 1) a SMA de 10, 20 y 50 dias
  - RSI de 14 dias

Label: signo del retorno de NEE en t+1 (1 = sube, 0 = baja/igual).
Senal aplicada: +1 si el modelo predice sube, -1 si predice baja
(se reusa el motor generico backtest() de backtest_engine.py, que ya
aplica el rezago de 1 dia adicional para evitar look-ahead).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import descargar_precios_diarios_ohlc
from metrics import calcular_metricas, imprimir_metricas
from backtest_engine import backtest

TICKER = "NEE"
COST_BPS = 10
PERIODS_PER_YEAR = 252

VENTANA_MINIMA = 756   # ~3 anos, elegido por el usuario
FRECUENCIA_REENTRENO = 63  # ~1 trimestre, elegido por el usuario
K_VECINOS = 15


def calcular_rsi(precios: pd.Series, periodo: int = 14) -> pd.Series:
    delta = precios.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    avg_ganancia = ganancia.rolling(periodo).mean()
    avg_perdida = perdida.rolling(periodo).mean()
    rs = avg_ganancia / avg_perdida.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def construir_features(precios: pd.Series) -> pd.DataFrame:
    ret1 = precios.pct_change(1)
    ret5 = precios.pct_change(5)
    ret10 = precios.pct_change(10)
    ret20 = precios.pct_change(20)
    vol20 = ret1.rolling(20).std()

    sma10 = precios.rolling(10).mean()
    sma20 = precios.rolling(20).mean()
    sma50 = precios.rolling(50).mean()

    dist_sma10 = precios / sma10 - 1
    dist_sma20 = precios / sma20 - 1
    dist_sma50 = precios / sma50 - 1

    rsi14 = calcular_rsi(precios, 14)

    df = pd.DataFrame({
        "ret1": ret1, "ret5": ret5, "ret10": ret10, "ret20": ret20,
        "vol20": vol20,
        "dist_sma10": dist_sma10, "dist_sma20": dist_sma20, "dist_sma50": dist_sma50,
        "rsi14": rsi14,
    })
    return df


def generar_predicciones_walk_forward(precios: pd.Series, features: pd.DataFrame,
                                       ventana_minima: int, frecuencia_reentreno: int,
                                       k: int) -> pd.Series:
    """
    Entrena un KNeighborsClassifier con ventana expansiva y reentrena cada
    'frecuencia_reentreno' dias. Devuelve una serie de predicciones
    (+1 / -1) indexada igual que 'precios', con 0/NaN->0 antes de que
    haya suficiente historia para el primer entrenamiento.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    # label: signo del retorno de manana (t+1), disponible solo hasta ayer
    ret_manana = precios.pct_change().shift(-1)
    label = (ret_manana > 0).astype(int)

    datos = features.join(label.rename("label")).dropna()
    idx_datos = datos.index

    predicciones = pd.Series(0, index=precios.index, dtype=int)

    if len(idx_datos) <= ventana_minima:
        return predicciones  # no hay suficiente historia para entrenar ni una vez

    i = ventana_minima
    while i < len(idx_datos):
        train = datos.iloc[:i]
        fin_bloque = min(i + frecuencia_reentreno, len(idx_datos))
        bloque_test_idx = idx_datos[i:fin_bloque]

        X_train = train.drop(columns="label").values
        y_train = train["label"].values

        scaler = StandardScaler()
        X_train_esc = scaler.fit_transform(X_train)

        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train_esc, y_train)

        X_test = datos.loc[bloque_test_idx].drop(columns="label").values
        X_test_esc = scaler.transform(X_test)
        pred = modelo.predict(X_test_esc)  # 0 o 1

        senal = np.where(pred == 1, 1, -1)
        predicciones.loc[bloque_test_idx] = senal

        i = fin_bloque

    return predicciones


def main():
    ohlc = descargar_precios_diarios_ohlc(TICKER, start="2005-01-01")
    precios = ohlc["Close"]
    features = construir_features(precios)

    posiciones = generar_predicciones_walk_forward(
        precios, features, VENTANA_MINIMA, FRECUENCIA_REENTRENO, K_VECINOS
    )
    posiciones.name = "posicion"

    resultado = backtest(precios, posiciones, holding=1, costo_bps=COST_BPS)

    # recortar el periodo sin senal (antes del primer entrenamiento) para
    # que las metricas no esten infladas/desinfladas por ceros iniciales
    primer_senal = posiciones[posiciones != 0].index.min()
    resultado_evaluado = resultado.loc[resultado.index >= primer_senal]

    m_estrategia = calcular_metricas(resultado_evaluado, "retorno_estrategia_neto", "equity_estrategia", periods_per_year=PERIODS_PER_YEAR)
    m_bh = calcular_metricas(resultado_evaluado, "retorno_mensual_activo", "equity_buy_and_hold", periods_per_year=PERIODS_PER_YEAR)

    print(f"\n=== Estrategia 17: ML KNN (walk-forward trimestral, ventana min. {VENTANA_MINIMA}d) sobre {TICKER} ===")
    print(f"Periodo evaluado (post primer entrenamiento): {resultado_evaluado.index[0].date()} a {resultado_evaluado.index[-1].date()}  ({m_estrategia['periodos']} dias)\n")
    imprimir_metricas("Estrategia KNN (neta de costos)", m_estrategia)
    print()
    imprimir_metricas("Buy & Hold", m_bh)

    # tasa de acierto direccional, para diagnostico (no es una metrica financiera)
    ret_real = precios.pct_change().reindex(resultado_evaluado.index)
    acierto = (np.sign(ret_real) == np.sign(resultado_evaluado["posicion"])).mean()
    print(f"\n   tasa_acierto_direccional   : {acierto:.4f}  (0.50 = azar)")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(resultado_evaluado.index, resultado_evaluado["equity_estrategia"], label="Estrategia KNN (neta)")
    ax.plot(resultado_evaluado.index, resultado_evaluado["equity_buy_and_hold"], label="Buy & Hold NEE", linestyle="--")
    ax.set_title(f"Estrategia 17 (ML KNN walk-forward) vs Buy & Hold — {TICKER}")
    ax.set_ylabel("Valor de $1 invertido")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    carpeta_figuras = os.path.join(carpeta_raiz, "outputs", "figures")
    carpeta_resultados = os.path.join(carpeta_raiz, "outputs", "results")
    os.makedirs(carpeta_figuras, exist_ok=True)
    os.makedirs(carpeta_resultados, exist_ok=True)
    fig.savefig(os.path.join(carpeta_figuras, "strat_17_ml_knn_NEE.png"), dpi=150)
    resultado_evaluado.to_csv(os.path.join(carpeta_resultados, "strat_17_ml_knn_NEE.csv"))
    print(f"\nArchivos guardados en outputs/figures y outputs/results.")


if __name__ == "__main__":
    main()
