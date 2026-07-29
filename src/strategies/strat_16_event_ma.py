"""
Estrategia 16 (Seccion 3.16): Event-driven M&A -- adaptada a un solo activo.

IMPORTANTE (nota metodologica): el paper original aplica esta estrategia a
un universo amplio de deals (long target / short acquirer, o variantes de
arbitraje de fusion). NEE nunca fue TARGET de una adquisicion en su
historia como empresa publica; siempre actuo como ACQUIRER (o acquirer
fallido). Por lo tanto, esta no es una senal continua como las demas 20
estrategias: son solo 3 situaciones de M&A distintas (7 fechas de evento)
en ~21 anos de historia.

Por eso esto se implementa como un EVENT STUDY clasico (retorno acumulado
en una ventana post-evento + prueba de significancia por bootstrap),
NO como un backtest con curva de equity comparable a las demas
estrategias. El resultado debe reportarse en el reporte final como caso
de estudio cualitativo, no en la tabla de Sharpe/retorno anualizado junto
a las demas 20 estrategias -- mezclarlos seria enganoso dado el n tan
pequeno.

EVENTOS (verificados via SEC 8-K, comunicados de prensa y cobertura de
Utility Dive / Bloomberg):

  1. 2014-12-03  Anuncio acuerdo NEE-Hawaiian Electric (HEI)      [ANUNCIO]
  2. 2016-07-18  Terminacion del acuerdo NEE-HEI (rechazo PUC Hawai, 2016-07-15) [FRACASO]
  3. 2016-07-29  Anuncio acuerdo NEE-Oncor (via EFH)               [ANUNCIO]
  4. 2017-04-13  Primer rechazo PUCT Texas al deal Oncor            [FRACASO]
  5. 2017-06-07  Rechazo final PUCT (reconsideracion) -- deal muerto [FRACASO]
  6. 2018-05-21  Anuncio acuerdo NEE-Gulf Power (Southern Co.)      [ANUNCIO]
  7. 2019-01-01  Cierre/completacion adquisicion Gulf Power         [COMPLETADO]

Hipotesis (literatura estandar de M&A): anuncios de adquisicion del
ACQUIRER suelen tener reaccion neutra/negativa (mercado castiga la prima
pagada y el riesgo de ejecucion); fracasos de deals grandes suelen tener
reaccion neutra/positiva (alivio, la empresa se queda con el capital);
completaciones exitosas suelen ser neutras (ya estaban descontadas).

Metodologia:
  - CAR (retorno acumulado) de NEE en los N dias habiles siguientes al
    evento (ventana = [t, t+N], N=5 por defecto).
  - Significancia: se compara cada CAR observado contra una distribucion
    empirica de 5000 ventanas aleatorias de la misma longitud, tomadas
    del resto de la serie (bootstrap), y se reporta el percentil en el
    que cae el CAR real dentro de esa distribucion.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import descargar_precios_diarios_ohlc

TICKER = "NEE"
VENTANA_DIAS = 5
N_BOOTSTRAP = 5000
SEED = 42

EVENTOS = [
    {"fecha": "2014-12-03", "tipo": "ANUNCIO",    "descripcion": "Anuncio acuerdo NEE-Hawaiian Electric (HEI)"},
    {"fecha": "2016-07-18", "tipo": "FRACASO",    "descripcion": "Terminacion acuerdo NEE-HEI (rechazo PUC Hawai)"},
    {"fecha": "2016-07-29", "tipo": "ANUNCIO",    "descripcion": "Anuncio acuerdo NEE-Oncor (via EFH)"},
    {"fecha": "2017-04-13", "tipo": "FRACASO",    "descripcion": "Primer rechazo PUCT Texas al deal Oncor"},
    {"fecha": "2017-06-07", "tipo": "FRACASO",    "descripcion": "Rechazo final PUCT (reconsideracion), deal Oncor muerto"},
    {"fecha": "2018-05-21", "tipo": "ANUNCIO",    "descripcion": "Anuncio acuerdo NEE-Gulf Power (Southern Co.)"},
    {"fecha": "2019-01-01", "tipo": "COMPLETADO", "descripcion": "Cierre adquisicion Gulf Power"},
]


def calcular_car(precios: pd.Series, fecha_evento: pd.Timestamp, ventana: int) -> float:
    """
    Retorno acumulado (compuesto) de 'precios' en los 'ventana' dias
    habiles de trading siguientes a la primera fecha disponible >= fecha_evento.
    Devuelve np.nan si no hay suficientes datos despues del evento.
    """
    idx = precios.index
    pos = idx.searchsorted(fecha_evento)
    if pos >= len(idx) or pos + ventana >= len(idx):
        return np.nan
    p_inicio = precios.iloc[pos]
    p_fin = precios.iloc[pos + ventana]
    return p_fin / p_inicio - 1


def bootstrap_percentil(precios: pd.Series, car_observado: float, ventana: int,
                         n_boot: int, rng: np.random.Generator) -> float:
    """
    Genera n_boot ventanas aleatorias de longitud 'ventana' sobre toda la
    serie de precios, calcula su retorno compuesto, y devuelve el
    percentil (0-100) en el que cae car_observado dentro de esa
    distribucion empirica (0 = el mas bajo posible, 100 = el mas alto).
    """
    n = len(precios)
    if n <= ventana + 1 or np.isnan(car_observado):
        return np.nan

    inicios = rng.integers(0, n - ventana - 1, size=n_boot)
    valores = precios.values
    car_boot = valores[inicios + ventana] / valores[inicios] - 1

    return float((car_boot < car_observado).mean() * 100)


def main():
    ohlc = descargar_precios_diarios_ohlc(TICKER, start="2005-01-01")
    precios = ohlc["Close"]
    rng = np.random.default_rng(SEED)

    print(f"\n=== Estrategia 16: Event study M&A sobre {TICKER} (adaptada, n={len(EVENTOS)} eventos) ===")
    print(f"Ventana post-evento: {VENTANA_DIAS} dias habiles | Bootstrap: {N_BOOTSTRAP} muestras\n")

    filas = []
    for ev in EVENTOS:
        fecha = pd.Timestamp(ev["fecha"])
        car = calcular_car(precios, fecha, VENTANA_DIAS)
        percentil = bootstrap_percentil(precios, car, VENTANA_DIAS, N_BOOTSTRAP, rng)
        filas.append({
            "fecha": ev["fecha"],
            "tipo": ev["tipo"],
            "descripcion": ev["descripcion"],
            f"CAR_{VENTANA_DIAS}d": car,
            "percentil_bootstrap": percentil,
        })
        car_str = f"{car:+.4%}" if not np.isnan(car) else "N/D (fuera de rango de datos)"
        pct_str = f"{percentil:.1f}" if not np.isnan(percentil) else "N/D"
        print(f"[{ev['tipo']:10s}] {ev['fecha']}  {ev['descripcion']}")
        print(f"             CAR {VENTANA_DIAS}d: {car_str}   |  percentil vs bootstrap: {pct_str}\n")

    df = pd.DataFrame(filas)

    # Resumen por tipo de evento (promedio simple; n muy chico, solo referencia)
    print("--- Resumen por tipo de evento (promedio simple, n pequeno -> solo orientativo) ---")
    resumen = df.groupby("tipo")[f"CAR_{VENTANA_DIAS}d"].agg(["mean", "count"])
    print(resumen)

    print("\nADVERTENCIA: con n=7 fechas de evento (3 situaciones de M&A distintas),")
    print("esto NO tiene poder estadistico para conclusiones fuertes. Se reporta como")
    print("caso de estudio cualitativo, no como estrategia comparable en Sharpe/retorno")
    print("anualizado frente a las demas 20 estrategias.")

    # Grafico: precio de NEE con lineas verticales en cada evento
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(precios.index, precios.values, color="steelblue", linewidth=1)
    colores = {"ANUNCIO": "green", "FRACASO": "red", "COMPLETADO": "gray"}
    for ev in EVENTOS:
        fecha = pd.Timestamp(ev["fecha"])
        if precios.index[0] <= fecha <= precios.index[-1]:
            ax.axvline(fecha, color=colores.get(ev["tipo"], "black"), linestyle="--", alpha=0.6)
    ax.set_title(f"Estrategia 16 (Event study M&A) — {TICKER}: precio con eventos marcados")
    ax.set_ylabel("Precio de cierre ajustado (USD)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    carpeta_figuras = os.path.join(carpeta_raiz, "outputs", "figures")
    carpeta_resultados = os.path.join(carpeta_raiz, "outputs", "results")
    os.makedirs(carpeta_figuras, exist_ok=True)
    os.makedirs(carpeta_resultados, exist_ok=True)
    fig.savefig(os.path.join(carpeta_figuras, "strat_16_event_ma_NEE.png"), dpi=150)
    df.to_csv(os.path.join(carpeta_resultados, "strat_16_event_ma_NEE.csv"), index=False)
    print(f"\nArchivos guardados en outputs/figures y outputs/results.")


if __name__ == "__main__":
    main()
