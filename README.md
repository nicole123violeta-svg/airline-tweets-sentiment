# Análisis de sentimiento de tuits de aerolíneas con Machine Learning

Proyecto de clasificación de texto que identifica si un tuit sobre una aerolínea es negativo, neutral o positivo.

## Dataset
Twitter US Airline Sentiment (Kaggle): 14.640 tuits en inglés sobre aerolíneas de EE. UU., con tres categorías. Las clases están desbalanceadas: 62,7% negativos, 21,2% neutrales y 16,1% positivos.

- Kaggle: https://www.kaggle.com/crowdflower/twitter-airline-sentiment
- Licencia del dataset: CC BY-NC-SA (uso no comercial)

## Herramientas
Python, pandas, scikit-learn, matplotlib, Google Colab.

## Metodología
1. Exploración de datos y análisis del desbalance de clases.
2. Limpieza del texto: minúsculas, y eliminación de links, menciones y símbolos.
3. División en entrenamiento (80%) y prueba (20%) estratificada.
4. Vectorización con TF-IDF usando palabras sueltas y pares de palabras (para conservar el sentido de expresiones como "not good").
5. Entrenamiento y comparación de dos modelos: Naive Bayes y Regresión Logística (con class_weight="balanced").

## Resultados
| Modelo | Precisión general | Recall negative | Recall neutral | Recall positive |
|---|---|---|---|---|
| Naive Bayes | 69,9% | 1,00 | 0,15 | 0,26 |
| Regresión Logística | 78,8% | 0,85 | 0,69 | 0,69 |

- Un modelo que clasificara todo como negativo acertaría el 62,7%, así que ese es el punto de referencia.
- Naive Bayes casi no reconoce tuits neutrales ni positivos: tiende a predecir siempre "negativo".
- La Regresión Logística mejora la detección de las clases minoritarias y es el modelo final.
- Los errores más frecuentes se dan con la clase neutral: 220 tuits negativos fueron clasificados como neutrales.

## Demo
| Tuit | Predicción |
|---|---|
| Thanks for the great service, the crew was amazing! | positive |
| My flight was delayed three hours and nobody told us anything | negative |
| Does the flight to Boston leave from gate 12? | neutral |

## Aprendizajes
- Con clases desbalanceadas, la precisión general no alcanza: hay que mirar el recall de cada clase.
- Ajustar el peso de las clases mejora la detección de las categorías minoritarias.
- En análisis de sentimiento conviene no eliminar palabras como "not", porque cambian el significado.

## Posibles mejoras
Probar otros modelos (SVM, redes neuronales), ajustar hiperparámetros y usar modelos de lenguaje preentrenados.

## Autora
Nicole Ramirez - [LinkedIn](https://www.linkedin.com/in/nicole-ramirez-0a2a06294)
