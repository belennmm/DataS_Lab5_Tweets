# Laboratorio 5 – Clasificación de Tweets usando Minería de Texto

**CC3084 – Data Science**
Universidad del Valle de Guatemala
Semestre II – 2026

## Descripción

Este laboratorio implementa un pipeline completo de minería de texto y análisis de sentimiento sobre el dataset [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) de Kaggle. El objetivo es clasificar si un tweet se refiere a un desastre real (`target = 1`) o no (`target = 0`), e incorporar un análisis de sentimiento como variable adicional para el modelo.

## Dataset

- **Filas:** ~10,500 (train.csv)
- **Columnas:**
  - `id`: identificador del tweet
  - `keyword`: palabra clave del tweet (puede estar vacía)
  - `location`: ubicación desde donde fue enviado el tweet
  - `text`: texto del tweet
  - `target`: 1 = desastre real, 0 = no desastre

## Estructura del repositorio

```
├── Lab5_Teewts.ipynb                  # Preprocesamiento y análisis exploratorio (puntos 1–5)
├── Lab5_modelos_clasificacion.ipynb   # Modelado, clasificación y sentimiento (puntos 6–10)
├── train.csv / test.csv               # Datos originales de Kaggle
├── train_clean.csv / test_clean.csv   # Datos ya preprocesados (salida del primer notebook)
└── README.md
```

Los dos notebooks están pensados para ejecutarse **en orden**: `Lab5_Teewts.ipynb` primero (genera `train_clean.csv`), y luego `Lab5_modelos_clasificacion.ipynb` (lo consume).

## 1. `Lab5_Teewts.ipynb` — Preprocesamiento y análisis exploratorio

### Limpieza y preprocesamiento (punto 3)

Se creó la columna `text_clean` a partir de `text`, aplicando en orden:

1. Conversión a minúsculas.
2. Eliminación de URLs (`http`, `www`).
3. Eliminación de menciones (`@usuario`, se descartan completas porque no aportan información sobre si el tweet es un desastre real) y del símbolo `#` (conservando la palabra del hashtag, ya que suele traer información relevante, p. ej. "wildfires").
4. Revisión y eliminación de emojis Unicode (rangos de emoticones, pictogramas, banderas, etc.). **Resultado:** no se encontró ningún tweet con emoji Unicode en el dataset (0 de 7,613).
5. Eliminación de signos de puntuación.
6. Eliminación de números, **conservando "911"** por su relevancia semántica en contexto de emergencias. **Resultado:** se encontraron 7 tweets con "911" en la clase desastre real y solo 3 en la clase no-desastre, lo que confirma que valía la pena conservarlo en lugar de eliminarlo junto con el resto de números.
7. Eliminación de stopwords en inglés (NLTK), dado que el dataset está compuesto por tweets en inglés.
8. Normalización de espacios extra generados por los pasos anteriores.
9. Decodificación de entidades HTML (`&amp;`, `&lt;`, `&gt;`, etc.), aplicada antes de la eliminación de menciones y puntuación para evitar que el símbolo `&` quedara convertido en basura tipo "ampamp". **Resultado:** se detectaron varios tweets con entidades sin decodificar (p. ej. "Rene Ablaze &amp; Jacinta").
10. Revisión de emoticones hechos con puntuación (`:)`, `:(`, `;)`, etc.), ya que estos no los cubre la limpieza de emojis Unicode y se destruyen silenciosamente al quitar la puntuación. **Resultado:** se encontraron 4,059 tweets (más de la mitad del dataset) con este tipo de emoticones de texto. Se decidió no conservarlos como señal aparte en este notebook porque el objetivo aquí es la clasificación por contenido, no el sentimiento, pero se documentó el hallazgo porque es clave para el notebook de sentimiento (punto 8), donde sí se usa el texto crudo en lugar del preprocesado.
11. Normalización de letras repetidas (p. ej. "fiiiire" → "fiire"), dejando máximo dos repeticiones para conservar la raíz de la palabra sin distorsionarla.
12. Eliminación de palabras de una sola letra sueltas, residuales de los pasos anteriores (p. ej. la "s" que puede quedar de un posesivo mal cortado).
13. Lematización con POS tagging (`WordNetLemmatizer` + `averaged_perceptron_tagger_eng`), en vez de stemming, para obtener palabras reales en su forma base (p. ej. "disasters" → "disaster") en lugar de raíces truncadas no siempre válidas (p. ej. "disastrous" → "disastr" con stemming).
14. Revisión de duplicados y filas vacías tras la limpieza. **Resultado:** se encontraron 110 tweets duplicados en el texto original y 0 tweets que quedaran vacíos después de toda la limpieza, por lo que no fue necesario descartar ninguna fila por ese motivo.

**Paquetes usados:** `re`, `string`, `html`, `nltk` (`stopwords`, `wordnet`, `omw-1.4`, `averaged_perceptron_tagger_eng`).

### Frecuencia de palabras y n-gramas (punto 4)

- Frecuencia de palabras (unigramas) calculada por separado para tweets de desastre (`target=1`) y no desastre (`target=0`) usando `collections.Counter`.
- Identificación de palabras "distintivas" de cada clase (diferencia entre los top-50 de cada categoría).
- Cálculo de frecuencia relativa (proporción sobre el total de tweets de cada clase), no solo conteo absoluto.
- Generación de **bigramas y trigramas** con `nltk.bigrams` / `nltk.trigrams` para capturar contexto (p. ej. combinaciones como "suicide bomber" o "northern california" que no se detectan con unigramas).

**Resultados obtenidos:**

- La palabra más frecuente en tweets de desastre es **"fire"** (268 apariciones, 8.19% de esos tweets), seguida de "kill", "bomb", "news" y "disaster".
- La palabra más frecuente en tweets que no son de desastre es **"get"** (305 apariciones, 7.02% de esos tweets), seguida de "like", "im" y "go".
- Al comparar los top-50 de cada clase, se identificaron 33 palabras exclusivas de desastre (p. ej. "collapse", "wildfire", "evacuate", "nuclear", "nlm") y 33 palabras exclusivas de no-desastre (p. ej. "lol", "love", "cant", "scream", "fuck"), lo que confirma un vocabulario claramente distinto entre ambas clases.
- Se encontraron 17 palabras presentes en ambos top-50 (p. ej. "fire", "get", "like", "amp", "via", "emergency"), pero con pesos muy distintos según la clase: por ejemplo "fire" representa 8.19% en desastre vs. solo 2.07% en no-desastre, y "like" representa 6.68% en no-desastre vs. 3.03% en desastre. Esto muestra que aunque una palabra aparezca en ambas categorías, su frecuencia relativa puede seguir siendo un buen indicador para el modelo.

**Conclusión:** las palabras más frecuentes en desastre real están asociadas a eventos concretos (fire, flood, storm, killed, emergency), mientras que en no-desastre predominan palabras de uso cotidiano/coloquial. Los bigramas/trigramas sí aportan valor porque desambiguan contexto (una palabra sola como "fire" es ambigua, pero "wild fire spreading" no lo es).

### Análisis exploratorio (punto 5)

- **Palabra más repetida por categoría:** "fire" (268) en desastre y "get" (305) en no-desastre, lo cual ya anticipa que el vocabulario de desastre es más específico y el de no-desastre más genérico.
- **Nube de palabras (wordcloud):** en la nube de desastre destacan "fire", "flood", "storm", "new" y "suicide bomber", todo vocabulario ligado a eventos reales. En la nube de no-desastre destacan "im", "amp", "one", "go" y expresiones informales como "love" y "want", reflejando un lenguaje más casual o figurado.
- **Histograma de palabras más repetidas:** confirma numéricamente lo anterior. En desastre, "fire" está muy por encima del resto (casi el doble que la segunda palabra, "kill"). En no-desastre, "get" y "like" dominan claramente sobre el resto.
- **Palabras presentes en ambas categorías:** se identificaron 17 palabras compartidas entre los top-50 de cada clase (detalle y porcentajes en la sección de n-gramas arriba). La discusión relevante es que, aunque compartidas, palabras como "fire" siguen siendo mucho más características de desastre real, mientras que palabras como "get" o "like" son mucho más características de no-desastre, y solo unas pocas (como "still", "video", "time") tienen proporciones casi idénticas entre clases, por lo que aportan poco valor discriminativo.
- **Balance de clases:** el dataset tiene 4,342 tweets de no-desastre (57%) y 3,271 tweets de desastre real (43%). El desbalance es moderado y no debería representar un problema grave para el entrenamiento de los modelos.
- **Longitud de los tweets:** tanto en palabras como en caracteres, las distribuciones de ambas clases son similares en forma. La mayoría de tweets tiene entre 5 y 12 palabras y entre 100 y 145 caracteres (con un pico marcado cerca del límite histórico de 140 caracteres de Twitter). Los tweets de desastre real tienden a ser levemente más largos en palabras que los de no-desastre.
- **Relación entre `keyword` y `target`:** contrario a lo esperado, los tweets sin `keyword` asignado tienen mayor probabilidad de ser desastre real (68.9%) que los que sí tienen `keyword` (42.8%). Esto se explica porque varias keywords del dataset (p. ej. "armageddon", "explode") se usan frecuentemente en sentido figurado.
- **Top keywords por clase:** en desastre real predominan keywords graves y específicas como "derailment", "outbreak", "wreckage" y "suicide%20bombing". En no-desastre predominan keywords que suenan graves pero se usan de forma coloquial o exagerada, como "armageddon", "harm" y "twister".
- **Hashtags y menciones:** los tweets de desastre usan en promedio más hashtags (0.50 vs. 0.39), consistente con su función de categorizar noticias reales, mientras que los tweets de no-desastre usan más menciones a otros usuarios (0.42 vs. 0.27), consistente con un uso más conversacional.

## 2. `Lab5_modelos_clasificacion.ipynb` — Modelado y análisis de sentimiento

### Modelos de clasificación (punto 6)

- **Vectorización:** TF-IDF con `ngram_range=(1,2)` sobre `text_clean`, para capturar tanto palabras individuales como contexto de pares de palabras.
- **Split:** 70% entrenamiento / 30% prueba, estratificado por `target` (`random_state=42`).
- **Algoritmos probados:**
  - Regresión Logística (`max_iter=200`)
  - Naive Bayes Multinomial
  - Random Forest (`n_estimators=100`)
- **Métricas de evaluación:** Accuracy, Precision, Recall, F1-Score, y matriz de confusión para cada modelo.
- Se seleccionó el mejor modelo según F1-Score (columna `resultados_modelos` ordenada de forma descendente).

**Manejo de contexto:** se abordó incluyendo bigramas en el vectorizador TF-IDF, de manera que el modelo no solo ve palabras sueltas sino también pares de palabras consecutivas, lo cual ayuda a capturar frases con significado distinto al de sus palabras por separado.

### Función de clasificación (punto 7)

Se implementó `clasificar_tweet(tweet)`, que:
1. Recibe un tweet **sin preprocesar**.
2. Aplica internamente la misma cadena de preprocesamiento del punto 3 (`preprocesar_tweet`).
3. Vectoriza el texto limpio con el `vectorizer` ya entrenado.
4. Predice con el modelo entrenado (`modelo_lr`) y devuelve `"Desastre real"` o `"No desastre"`.

También se incluyó un modo interactivo por consola para probar tweets manualmente.

### Análisis de sentimiento (punto 8)

- Se usó **VADER** (`nltk.sentiment.SentimentIntensityAnalyzer`), un analizador de sentimiento léxico-basado en reglas, optimizado para texto corto de redes sociales.
- El análisis se aplicó sobre `train['text']` (texto **original**, no `text_clean`), ya que VADER aprovecha mayúsculas, signos de puntuación y emoticones de texto (`:)`, `:(`) como señales de sentimiento — información que ya había sido eliminada en el preprocesamiento del punto 3.
- Se generaron las columnas `sent_neg`, `sent_neu`, `sent_pos` y `sent_compound` (puntaje agregado entre -1 y 1).
- Clasificación final en `positivo` / `negativo` / `neutro` según el umbral estándar de VADER sobre `sent_compound` (≥0.05 positivo, ≤-0.05 negativo, resto neutro).

**Conclusión:** sí vale la pena conservar los emoticones y signos de puntuación para este análisis específico, por lo que se usó el texto crudo en lugar del preprocesado para el módulo de sentimiento.

### Resultados de sentimiento (punto 9)

- **9.1 / 9.2:** se identificaron los 10 tweets más negativos y los 10 más positivos según `sent_compound`, junto con su categoría (`target`).
- **9.3:** se comparó el promedio de `sent_compound` entre categorías (`groupby('target')`) y se visualizó con un boxplot. Se corrió además una prueba t de Welch (`scipy.stats.ttest_ind`) para verificar si la diferencia de sentimiento entre tweets de desastre real y no-desastre es estadísticamente significativa.

### Variable de negatividad y reentrenamiento (punto 10)

- Se creó la variable `negatividad` = `sent_neg` (proporción de contenido negativo detectado por VADER en cada tweet).
- Se combinó con la matriz TF-IDF existente usando `scipy.sparse.hstack`, agregando **una columna numérica adicional** a las ~42,000 columnas del vocabulario TF-IDF.
- Se reentrenó el modelo (Regresión Logística) con esta nueva matriz y se comparó contra el modelo original mediante Accuracy, Precision, Recall y F1-Score.

**Resultado obtenido:** la inclusión de la variable de negatividad **no produjo una mejora sustancial** en el desempeño del modelo (diferencias de F1-Score del orden de 0.005, prácticamente imperceptibles). Esto se explica porque el vocabulario TF-IDF ya captura de forma implícita gran parte de la señal de sentimiento a través de palabras como "fire", "help", "dead", etc., por lo que una única variable adicional de sentimiento aporta poca información nueva frente a las más de 42,000 features ya existentes.

## Paquetes y módulos utilizados

| Librería | Uso |
|---|---|
| `pandas`, `numpy` | Manejo de datos |
| `re`, `string`, `html` | Limpieza de texto |
| `nltk` (`stopwords`, `wordnet`, `omw-1.4`, `averaged_perceptron_tagger`, `vader_lexicon`) | Stopwords, lematización, POS tagging, análisis de sentimiento (VADER) |
| `scikit-learn` | TF-IDF, split de datos, modelos (Logistic Regression, Naive Bayes, Random Forest), métricas |
| `scipy` | Combinación de matrices dispersas (`hstack`) y prueba estadística (`ttest_ind`) |
| `matplotlib`, `seaborn` | Visualizaciones (matrices de confusión, boxplots, barplots) |
| `wordcloud` | Nube de palabras |
| `collections.Counter` | Frecuencia de palabras y n-gramas |

## Cómo ejecutar

```bash
# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install pandas numpy scikit-learn nltk scipy matplotlib seaborn wordcloud

# Ejecutar en orden
jupyter notebook Lab5_Teewts.ipynb                  # Genera train_clean.csv / test_clean.csv
jupyter notebook Lab5_modelos_clasificacion.ipynb   # Consume train_clean.csv
```

La primera ejecución descargará automáticamente los recursos necesarios de NLTK (`stopwords`, `wordnet`, `omw-1.4`, `averaged_perceptron_tagger`, `vader_lexicon`).
