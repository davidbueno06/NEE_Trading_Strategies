"""
state_machine_signals.py
--------------------------
Motor compartido para estrategias tecnicas cuyo Signal (Eqs. 321-331 del
paper) no es un simple signo puntual, sino una regla de "establecer
posicion / liquidar posicion" que crea HISTERESIS: una vez que se entra
largo o corto, la posicion se mantiene hasta que se cumple explicitamente
la condicion de liquidacion, aunque la condicion de entrada ya no se
cumpla.

Se usa en las Estrategias 12 (tres medias moviles), 13 (soporte y
resistencia) y 14 (canal / Donchian).
"""

import numpy as np
import pandas as pd


def aplicar_maquina_estados(
    entra_largo: pd.Series, sale_largo: pd.Series,
    entra_corto: pd.Series, sale_corto: pd.Series,
) -> pd.Series:
    """
    Las 4 series son booleanas, mismo indice. Devuelve una serie de
    posiciones (-1, 0, +1) dia a dia, con histeresis:
      - si esta flat y entra_largo -> pasa a largo
      - si esta largo y sale_largo -> pasa a flat
      - si esta flat y entra_corto -> pasa a corto
      - si esta corto y sale_corto -> pasa a flat
    """
    n = len(entra_largo)
    posiciones = np.zeros(n, dtype=int)
    estado = 0  # 0 = flat, 1 = largo, -1 = corto

    el = entra_largo.values
    sl = sale_largo.values
    ec = entra_corto.values
    sc = sale_corto.values

    for i in range(n):
        if estado == 0:
            if el[i]:
                estado = 1
            elif ec[i]:
                estado = -1
        elif estado == 1:
            if sl[i]:
                estado = 0
        elif estado == -1:
            if sc[i]:
                estado = 0
        posiciones[i] = estado

    return pd.Series(posiciones, index=entra_largo.index, name="posicion")
