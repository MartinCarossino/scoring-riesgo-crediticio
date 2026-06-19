# Scoring de Riesgo Crediticio

Modelo de regresión logística para predecir el riesgo de default en clientes de tarjetas de crédito, a partir del dataset público de UCI Machine Learning Repository (clientes de Taiwán, 2005).

## Objetivo

Predecir si un cliente va a caer en default (no pago) el próximo mes, a partir de su historial de pagos, datos demográficos e información de facturación — y entender qué variables son las que más influyen en ese riesgo.

## Dataset

- **Fuente:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Tamaño:** 30.000 clientes, 23 variables predictoras + 1 variable objetivo
- **Variable objetivo:** `default` (1 = cayó en default el mes siguiente, 0 = no)

## Proceso

1. **Carga de datos:** lectura del CSV directo desde este repositorio (sin dependencias externas de APIs o librerías de terceros).
2. **Limpieza:** corrección de valores no documentados en `EDUCATION` y `MARRIAGE`, renombrado de columnas, eliminación de `ID`.
3. **Análisis exploratorio (EDA):** balance de clases, tasa de default por grupo demográfico, distribución de edad, matriz de correlación.
4. **Modelo:** regresión logística con `scikit-learn`, evaluación con métricas más allá de accuracy (precision, recall, F1), y análisis del trade-off al ajustar el umbral de decisión.
5. **Interpretación:** análisis de coeficientes para identificar qué variables más influyen en el riesgo de default, y discusión de sus limitaciones.

## Principales hallazgos

- El **historial de pago reciente** es el predictor más fuerte de default — mucho más que cualquier variable demográfica.
- El dataset está moderadamente desbalanceado (22% default / 78% no default), lo que requiere mirar métricas más allá de accuracy.
- Existe un trade-off claro entre precision y recall según el umbral de decisión elegido — una decisión que depende del costo relativo de cada tipo de error para el negocio.

## Cómo reproducirlo

El notebook está pensado para correr de punta a punta en Google Colab sin configuración adicional:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MartinCarossino/scoring-riesgo-crediticio/blob/main/notebook/scoring_riesgo_crediticio.ipynb)

## Tecnologías

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Autor

Martín Carossino — [GitHub](https://github.com/MartinCarossino)
