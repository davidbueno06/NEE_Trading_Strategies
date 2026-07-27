"""
fundamentals_edgar.py
----------------------
Descarga datos fundamentales de NEE directamente desde SEC EDGAR (fuente
oficial y gratuita, con historia completa desde que la empresa reporta en
formato XBRL, ~2009 en adelante). Reemplaza el uso de yfinance para EPS y
balance, que solo da 4-5 trimestres de historia.

CIK de NextEra Energy Inc: 0000753308

Documentacion de la API: https://www.sec.gov/edgar/sec-api-documentation
"""

import time
import requests
import pandas as pd

CIK_NEE = "0000753308"

# La SEC EXIGE un User-Agent identificable (nombre + correo). Sin esto,
# la API devuelve error 403. Cambia esto por tus datos reales.
HEADERS = {
    "User-Agent": "Investigacion academica ITESM - tu_correo@ejemplo.com"
}


def descargar_companyfacts(cik: str = CIK_NEE) -> dict:
    """Descarga TODOS los conceptos XBRL reportados por la empresa (JSON crudo)."""
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def _extraer_serie(facts: dict, concepto: str, unidad: str, solo_trimestral: bool = True) -> pd.DataFrame:
    """
    Extrae un concepto US-GAAP (ej. 'EarningsPerShareDiluted',
    'StockholdersEquity') de los company facts, y regresa un DataFrame con
    una fila por periodo reportado (end date), quedandose con el valor mas
    reciente presentado para cada fecha (evita duplicados por refilings).
    """
    try:
        bloque = facts["facts"]["us-gaap"][concepto]["units"][unidad]
    except KeyError:
        return pd.DataFrame(columns=["end", "val", "filed", "form"])

    df = pd.DataFrame(bloque)
    if df.empty:
        return df

    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])

    if solo_trimestral and "start" in df.columns:
        # conceptos de flujo (EPS, ingresos): quedarse solo con periodos de
        # ~1 trimestre (entre 70 y 100 dias), para excluir acumulados YTD/anuales
        df["start"] = pd.to_datetime(df["start"])
        dias = (df["end"] - df["start"]).dt.days
        df = df[(dias >= 70) & (dias <= 100)]

    # si la misma fecha 'end' se reporto varias veces (correcciones), usar
    # el valor mas reciente segun fecha de presentacion ('filed')
    df = df.sort_values("filed").drop_duplicates(subset="end", keep="last")
    df = df.sort_values("end").reset_index(drop=True)
    return df[["end", "val", "filed", "form"]]


def descargar_eps_trimestral_edgar(cik: str = CIK_NEE) -> pd.Series:
    """EPS diluido trimestral (concepto EarningsPerShareDiluted, unidad USD/shares)."""
    facts = descargar_companyfacts(cik)
    df = _extraer_serie(facts, "EarningsPerShareDiluted", "USD/shares", solo_trimestral=True)
    if df.empty:
        raise ValueError("No se encontraron datos de EPS diluido en SEC EDGAR para este CIK.")
    eps = df.set_index("end")["val"]
    eps.name = "EPS"
    return eps


def descargar_book_value_trimestral_edgar(cik: str = CIK_NEE) -> pd.Series:
    """
    Book value por accion = StockholdersEquity (instantaneo, fin de trimestre)
    / acciones en circulacion (CommonStockSharesOutstanding, tambien instantaneo).
    """
    facts = descargar_companyfacts(cik)

    equity = _extraer_serie(facts, "StockholdersEquity", "USD", solo_trimestral=False)
    if equity.empty:
        raise ValueError("No se encontro StockholdersEquity en SEC EDGAR.")

    acciones = _extraer_serie(facts, "CommonStockSharesOutstanding", "shares", solo_trimestral=False)
    if acciones.empty:
        acciones = _extraer_serie(facts, "CommonStockSharesIssued", "shares", solo_trimestral=False)
    if acciones.empty:
        raise ValueError("No se encontraron acciones en circulacion en SEC EDGAR.")

    equity_s = equity.set_index("end")["val"]
    acciones_s = acciones.set_index("end")["val"]

    # emparejar por fecha de cierre de trimestre (tolerancia de unos dias)
    df = pd.merge_asof(
        equity_s.reset_index().rename(columns={"val": "equity"}).sort_values("end"),
        acciones_s.reset_index().rename(columns={"val": "acciones"}).sort_values("end"),
        on="end", tolerance=pd.Timedelta("10D"), direction="nearest",
    ).dropna()

    bvps = (df["equity"] / df["acciones"])
    bvps.index = df["end"]
    bvps.name = "BVPS"
    return bvps.sort_index()
