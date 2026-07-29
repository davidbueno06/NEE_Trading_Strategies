"""
factors_famafrench.py
-----------------------
Descarga los 3 factores de Fama-French (MKT, SMB, HML) mensuales desde la
Ken French Data Library (Dartmouth) -> fuente academica estandar, gratuita,
con historia completa desde 1926. Usada por la Estrategia 7 (Residual
momentum).
"""

import pandas as pd


def descargar_factores_ff3_mensuales() -> pd.DataFrame:
    """
    Devuelve un DataFrame mensual con columnas Mkt-RF, SMB, HML, RF, en
    fraccion decimal (no en %), indexado por fecha de fin de mes.
    """
    import pandas_datareader.data as web

    datos = web.DataReader("F-F_Research_Data_Factors", "famafrench")
    factores = datos[0].copy()  # [0] = datos mensuales; [1] = anuales
    factores.index = factores.index.to_timestamp(how="end").normalize()
    factores = factores / 100.0  # la fuente reporta en porcentaje
    return factores
