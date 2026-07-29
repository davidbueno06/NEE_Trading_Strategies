"""
multi_cluster_reversion.py
------------------------------
Motor para la Estrategia 21 (Seccion 3.9.1 del paper): Mean-reversion -
multiple clusters. Generalizacion de la Estrategia 9 (single cluster,
motor en multi_asset_reversion.py) a K > 1 clusters de acciones
historicamente correlacionadas.

DIFERENCIA CLAVE vs. Estrategia 9: en vez de "demeanar" cada retorno
respecto al promedio de TODO el grupo, aqui cada retorno se demeana
respecto al promedio de SU PROPIO cluster (Eqs. 301-309 del paper, via
la matriz de loadings binaria Lambda_iA). El propio paper senala que
esto es matematicamente equivalente a correr la Estrategia 9 de forma
independiente en cada cluster ("we can simply treat clusters
independently... construct a mean-reversion strategy following the
above procedure in each cluster") -- la version "unificada" con Lambda
solo importa porque permite fijar una UNICA escala global de inversion I
(Eq. 295) sobre todos los clusters juntos, en vez de una escala separada
por cluster.
"""

import numpy as np
import pandas as pd


def calcular_pesos_multicluster(
    precios: pd.DataFrame,
    clusters: dict,
    ventana_formacion: int,
) -> pd.DataFrame:
    """
    precios: DataFrame con una columna por activo (TODAS las acciones de
    TODOS los clusters juntas), precios mensuales.
    clusters: dict {ticker: nombre_cluster}. Debe cubrir todas las
    columnas de 'precios'; cada ticker pertenece a un solo cluster
    (consistente con la Eq. 299-300 del paper: sin clusters vacios,
    cada accion en un solo cluster).
    ventana_formacion: meses de la ventana de formacion (igual criterio
    que en Estrategia 9).

    Devuelve un DataFrame de pesos (una columna por activo). Los pesos
    son dollar-neutral DENTRO de cada cluster (Eq. 310-311 del paper),
    y la exposicion bruta total (suma de |pesos|) esta normalizada a 1
    a nivel global (los 2+ clusters juntos).
    """
    columnas = list(precios.columns)
    faltantes = [c for c in columnas if c not in clusters]
    if faltantes:
        raise ValueError(f"Faltan clusters para: {faltantes}")

    log_ret = np.log(precios / precios.shift(1)).dropna()
    ret_ventana = log_ret.rolling(ventana_formacion).sum().dropna()

    # Demeaning POR CLUSTER (Eq. 309): a cada retorno se le resta el
    # promedio de SU cluster, no el promedio de todo el grupo.
    demeaned = pd.DataFrame(index=ret_ventana.index, columns=columnas, dtype=float)
    for nombre_cluster in sorted(set(clusters.values())):
        miembros = [c for c in columnas if clusters[c] == nombre_cluster]
        if len(miembros) < 2:
            raise ValueError(
                f"El cluster '{nombre_cluster}' tiene menos de 2 activos; "
                "mean-reversion dentro del cluster no tiene sentido con 1 solo activo."
            )
        promedio_cluster = ret_ventana[miembros].mean(axis=1)
        demeaned[miembros] = ret_ventana[miembros].sub(promedio_cluster, axis=0)

    # Normalizacion GLOBAL (Eq. 295 y 298): la escala de inversion I se
    # fija sobre TODOS los activos de TODOS los clusters juntos.
    suma_abs = demeaned.abs().sum(axis=1)
    pesos = demeaned.div(-suma_abs, axis=0)  # signo negativo: corto al "rico" del cluster, largo al "barato"
    pesos = pesos.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return pesos
