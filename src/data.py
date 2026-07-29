"""
data.py
-------
Descarga y preparacion de precios. Reutilizable por todas las estrategias
(1 a 21) para que la logica de obtener datos no se repita en cada script.
"""

import numpy as np
import pandas as pd


def descargar_precios_mensuales(ticker: str, start: str = "2005-01-01") -> pd.Series:
    """
    Descarga precios mensuales AJUSTADOS (por splits y dividendos).
    Devuelve una pandas Series 1D indexada por fecha, sin importar si
    yfinance regresa columnas con multi-indice (Price, Ticker).
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        cierre = df["Close"][ticker] if ticker in df["Close"].columns else df["Close"].iloc[:, 0]
    else:
        cierre = df["Close"]

    cierre = pd.Series(np.asarray(cierre).ravel(), index=df.index)

    precios_m = cierre.resample("ME").last()
    precios_m.name = ticker
    return precios_m.dropna()


def descargar_precios_diarios_ohlc(ticker: str, start: str = "2005-01-01") -> pd.DataFrame:
    """
    Descarga precios diarios AJUSTADOS con Open/High/Low/Close, para
    estrategias que operan en frecuencia diaria (medias moviles, soporte
    y resistencia, canal).
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        columnas = {}
        for col in ["Open", "High", "Low", "Close"]:
            serie = df[col][ticker] if ticker in df[col].columns else df[col].iloc[:, 0]
            columnas[col] = np.asarray(serie).ravel()
        salida = pd.DataFrame(columnas, index=df.index)
    else:
        salida = df[["Open", "High", "Low", "Close"]].copy()

    return salida.dropna()
