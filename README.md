# Scoring de Riesgo Crediticio — UCI Default of Credit Card Clients

Modelo de regresión logística para predecir el riesgo de default en clientes de tarjetas de
crédito, a partir del dataset público de UCI Machine Learning Repository (clientes de Taiwán, 2005).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MartinCarossino/scoring-riesgo-crediticio/blob/main/notebook/scoring_riesgo_crediticio.ipynb)

## Preguntas que guían el análisis

- ¿Qué variables predicen mejor el riesgo de default: el historial de pago, los datos
  demográficos o la información de facturación?
- ¿Qué tan desbalanceadas están las clases, y cómo afecta eso a la elección de métricas?
- ¿Qué trade-off existe entre precision y recall según el umbral de decisión, y qué umbral
  tiene más sentido para un caso de uso real de scoring crediticio?

## Dataset

| Atributo | Detalle |
|---|---|
| Fuente | [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Tamaño | 30.000 clientes · 23 variables predictoras + 1 variable objetivo |
| Variable objetivo | `default` (1 = cayó en default el mes siguiente, 0 = no) |

El CSV está alojado en este repo para que el notebook sea 100% reproducible, sin necesidad de
descargar nada manualmente.

## Estructura del notebook

El análisis está dividido en 5 bloques, cada uno con su justificación en celdas de markdown:

1. **Carga y limpieza** — lectura del CSV, corrección de valores no documentados en
   `EDUCATION` y `MARRIAGE`, renombrado de columnas y eliminación de `ID`.
2. **Análisis exploratorio (EDA)** — balance de clases, tasa de default por grupo
   demográfico, distribución de edad y matriz de correlación.
3. **Modelado** — regresión logística con `scikit-learn`, split train/test, y entrenamiento
   del modelo base.
4. **Evaluación** — métricas más allá de accuracy (precision, recall, F1, AUC-ROC), y análisis
   del trade-off entre precision y recall al mover el umbral de decisión.
5. **Interpretación** — análisis de coeficientes para identificar qué variables más influyen
   en el riesgo de default, y discusión de las limitaciones del modelo.

## Principales hallazgos

**Sobre los datos**

- El dataset está desbalanceado: 22,1% de los clientes cayó en default, 77,9% no — esto exige
  mirar métricas más allá de accuracy.
- Se detectaron valores no documentados en `EDUCATION` y `MARRIAGE`, agrupados sin pérdida
  de filas.
- La edad de los clientes está sesgada hacia adultos jóvenes (concentración entre 25 y 40 años).

**Sobre los factores de riesgo**

- El **historial de pago reciente (`PAY_1`)** es, por lejos, el predictor más fuerte de
  default — mucho más que cualquier variable demográfica (género, educación, estado civil).
- El **límite de crédito otorgado** correlaciona negativamente con el riesgo: los clientes con
  mayor límite tienden a tener menor probabilidad de default, posiblemente porque el banco ya
  los consideraba de bajo riesgo al otorgárselo.

**Sobre el modelo**

- Con el umbral por defecto (0.5), el modelo logra **80,8% de accuracy** pero solo **24,2% de
  recall** — detecta menos de 1 de cada 4 clientes que efectivamente caen en default.
- Bajando el umbral a 0.3, el recall sube a **46,0%** (casi el doble), a costa de bajar la
  precision de 0.69 a 0.55. Este trade-off debería decidirlo el negocio según qué error sea más
  costoso: dejar pasar un cliente riesgoso, o generar fricción con uno bueno.

## Stack utilizado

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Estadística aplicada`

## Cómo reproducirlo

**Opción A — Google Colab (sin instalar nada)**

Hacer clic en el badge de arriba o abrir directamente:

```
https://colab.research.google.com/github/MartinCarossino/scoring-riesgo-crediticio/blob/main/notebook/scoring_riesgo_crediticio.ipynb
```

La primera celda carga el CSV directo desde el repo. Ejecutar Runtime → Run all.

**Opción B — Local**

```bash
git clone https://github.com/MartinCarossino/scoring-riesgo-crediticio
cd scoring-riesgo-crediticio
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook notebook/scoring_riesgo_crediticio.ipynb
```

## Limitaciones

- Los coeficientes de las variables `BILL_AMT` muestran signos contraintuitivos, probablemente
  por multicolinealidad entre los seis meses de facturación.
- El dataset corresponde a clientes de Taiwán en 2005; los patrones de comportamiento
  crediticio pueden no generalizar a otros países o períodos.
- El modelo es un punto de partida interpretable, no una solución de producción: un siguiente
  paso lógico sería probar modelos de árboles (Random Forest, XGBoost) y comparar performance.

## Autor

Martín Carossino — [Portfolio](https://github.com/MartinCarossino)
