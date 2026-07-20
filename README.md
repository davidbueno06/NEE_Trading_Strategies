# NEE Trading Strategies

Implementacion de las 21 estrategias del Capitulo 3 ("Stocks") de
*"151 Trading Strategies"* (Kakushadze & Serur, 2018), aplicadas a
NextEra Energy (NEE), componente del S&P 500.

## Estructura

```
src/
  data.py                 -> descarga y limpieza de precios (comun a todas las estrategias)
  metrics.py              -> calculo de metricas de desempeno (comun)
  backtest_engine.py      -> motor de backtest generico (comun)
  strategies/
    strat_01_momentum.py  -> Estrategia 1: Price-momentum
    ...
outputs/
  figures/                -> graficos (.png) generados por cada estrategia
  results/                -> detalle mensual (.csv) generado por cada estrategia
reports/
  reporte_final.docx      -> interpretacion y comparacion de resultados
```

## Instalacion

```
pip install -r requirements.txt
```

## Ejecutar una estrategia

```
python src/strategies/strat_01_momentum.py
```

Cada script genera su grafico en `outputs/figures/` y su CSV de detalle
mensual en `outputs/results/`.

## Metodologia comun a todas las estrategias

- Precios mensuales ajustados por splits y dividendos.
- Senal de la estrategia calculada con informacion hasta el mes t;
  aplicada al retorno del mes t+1 (sin look-ahead bias).
- Costos de transaccion de 10 puntos basicos por cambio de posicion.
- Comparacion siempre contra comprar-y-mantener (buy & hold) NEE.
