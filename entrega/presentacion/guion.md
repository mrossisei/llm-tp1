# Guión de la presentación — TP1: Predicción de BTR con Transformers

**Duración total: ~27 minutos** (rango pedido: 25–30). Los tiempos por diapositiva están
calibrados para hablar tranquilo; si el reloj aprieta, los recortes seguros son la mitad B de
la diapositiva 14 (palabras/w2v) y la 17 (dejar solo la curva de aprendizaje).

La presentación es `presentacion.html` (abrir en un navegador; flechas ← → para navegar,
imprimir a PDF desde el navegador si Campus pide archivo). Cada sección de este guión indica
**[tiempo]**, el texto sugerido para decir, y `→ en pantalla` con qué señalar.

---

## 1 · Portada — [0.5 min]

Buenas. Vamos a presentar el TP1: predicción de Buy Through Rate en un e-commerce de
supermercado, con un sistema basado en Transformers. La estructura sigue el enunciado:
formulación y EDA, el diseño del sistema con sus experimentos, y al final el ejercicio teórico
de personalización.

## 2 · El problema y la formulación — [1.5 min]

El BTR es una métrica de negocio: compras sobre impresiones en la página de resultados — la
probabilidad de que un producto mostrado se compre. El objetivo es identificar qué productos
promocionar.

La primera decisión del TP es la formulación, y conviene explicitarla: el dataset es un log de
**eventos** — cada fila es un producto impreso en una búsqueda concreta. Entonces formulamos
**clasificación binaria por impresión**: el modelo estima p(bought | producto, contexto de
búsqueda). El BTR de negocio por producto sale de **agregar** esas probabilidades sobre sus
impresiones, y "qué promociono" es rankear por esa agregación. Dos consecuencias: las métricas
correctas son las de clasificación binaria por fila (PR-AUC/ROC-AUC, como sugiere el enunciado),
y no hace falta ningún umbral — el uso final es un ranking.

`→ en pantalla`: el diagrama fila→p→agregación.

## 3 · EDA (1): la estructura y la primera trampa — [2 min]

10.000 impresiones, 2.012 búsquedas de 1 a 8 productos, 13% de positivos. Los filtros de la
búsqueda son constantes dentro de cada query — son contexto de la búsqueda, no del producto — y
verificamos que el 100% de los productos impresos cumple su filtro: eso significa que
`filter_category` y `filter_storage_type` no informan nada por sí solos a nivel fila.

La primera trampa: `cart`. Comprado implica carrito en el **100%** de las filas — es el funnel
impresión→carrito→compra, o sea, es *parte del resultado del mismo evento*, no un atributo
disponible al momento de decidir qué promocionar. Usarlo sería leakage: p(bought|cart=False) es
exactamente 0 y el modelo degeneraría a mirar solo eso. Queda afuera. (Spoiler: más adelante lo
probamos como *label auxiliar* de multi-task — que no es leakage porque nunca es input — y
tampoco aportó.)

`→ en pantalla`: tabla de estructura + el 100% de bought⟹cart.

## 4 · EDA (2): la señal dominante está escondida en el texto — [2.5 min]

El hallazgo central del EDA. El título termina en un sufijo entre paréntesis — "(Best Seller)",
"(Top Rated)" — en el 95% de las filas, y ese sufijo define **tiers de BTR brutales**: cuatro
sufijos compran entre 0.63 y 0.68; otros cuatro, 0.02–0.04; los once restantes y el sin-sufijo,
**0.000 exacto**. Y ojo: no es sentimiento — "Highly Rated" suena tan positivo como "Top Rated"
y compra 50 veces menos. Hay que aprender la partición exacta, no la valencia. La última oración
de la descripción repite lo mismo en prosa (verificado por crosstab).

Segundo efecto: el precio tiene una **U invertida** dentro del rango filtrado — se compra más en
el medio del rango que en los extremos — y está condicionada al tier. Eso motiva `price_rel`: la
posición relativa del precio dentro del rango que pidió el usuario. Es una señal *relacional*
producto×búsqueda.

Calidad de datos: los timestamps están rotos — una misma búsqueda tiene eventos separados por
hasta dos años — así que ni split temporal ni features de tiempo (después lo verificamos:
agregar hora/día empeora). Y hay redundancias: package_size ≈ peso, dimensiones ≈ envase,
descripción ≈ sufijo.

`→ en pantalla`: la tabla de tiers; el gráfico de la U.

## 5 · Features y preprocesamiento — [2 min]

Con ese EDA, la selección queda así. Entran **7 categóricas** — incluida `listing_status`, que
*derivamos* parseando el sufijo del título con una regex — y **6 numéricas**, incluida
`price_rel` derivada. Preprocesamiento por tipo: categóricas → codificación (era LA decisión
abierta: el enunciado sugiere investigar one-hot y alternativas — le dedicamos un experimento
entero, diapositiva 13); numéricas → z-score con estadísticos de train, con log1p previo en las
dos sesgadas a derecha (price, peso). Todo se ajusta **solo con train**; los niveles no vistos
van a un índice UNK.

Quedan afuera: `cart` (leakage), `query_id` (solo particiona), timestamp (roto), los filtros
redundantes, package/dimensiones/ingredientes (redundantes con el peso). Punto metodológico que
nos importa: **cada descarte fue después verificado por ablación** — reintroducir esos campos
midió delta ≈ 0. No descartamos en papel; descartamos y comprobamos.

`→ en pantalla`: la tabla feature→preprocesamiento→justificación.

## 6 · ¿Es válido usar "Best Seller"? Dos familias — [1.5 min]

Discusión conceptual que tuvimos en el equipo: esos badges se asignan *después* de vender mucho
— ¿no es circular usarlos? Nuestra resolución: distinguir tres clases de información. Outcome
del mismo evento (`cart`): leakage estricto, nunca. **Estado del producto al momento de la
impresión** (los badges): disponible al predecir y plausiblemente causal — prueba social — así
que es válido para *predecir*, pero es circular para *decidir promociones* y ciego al producto
nuevo. Atributos intrínsecos: válidos siempre.

Entonces entrenamos **dos familias**: *catálogo* (con estado — responde qué promocionar hoy) e
*intrínseca* (sin el estado, ni parseado ni escondido en el texto — responde qué esperar de un
producto nuevo). Adelanto el resultado porque encuadra todo: con estado se llega a ~0.82; sin
estado, **nadie** — ni el GBM — pasa de ~0.16, porque el 61% de las filas vive en tiers de BTR
exactamente cero. Eso no es un bug del modelo: es un hallazgo sobre el dataset.

## 7 · Métricas y protocolo experimental — [2.5 min]

**Métricas.** Con 13% de positivos, accuracy es inútil (el 87% se consigue diciendo siempre
"no"). La principal es **PR-AUC** — el azar da 0.131, así que hay mucho rango — y **ROC-AUC**
como complementaria, tal como sugiere el enunciado. Sin umbral: el uso es ranking; igual
medimos y guardamos **las 16 métricas** en cada época y corrida — y una de ellas lo confirma: el
F1 máximo se alcanza en umbral ~0.40, no 0.5 — cualquier umbral fijo hubiera sido arbitrario.
Overfitting/underfitting: las curvas train/val por época están guardadas para todas las
corridas; el early stopping corta por PR-AUC de **validación** con paciencia 20.

**Partición.** Por `query_id`, 70/15/15: una búsqueda entera cae del mismo lado — un split
aleatorio por fila filtraría información de la página. ¿Por qué no temporal? Timestamps rotos.
¿Por producto? Lo verificamos: 99% de los títulos son únicos, y restringir test a productos
jamás vistos da métricas idénticas — el modelo no recibe identidad del producto, no hay canal
para memorizar.

**Varianza.** Seguimos la priorización de la cátedra: **promedio de ejecuciones** — 6 seeds por
configuración — antes que cross-validation. Y como cierre de robustez igual corrimos GroupKFold
5×6: da 0.821 ± 0.012, consistente. Disciplina en todo el TP: los hiperparámetros se eligen
mirando validación; test se reporta al final.

## 8 · La arquitectura: dónde va el transformer y por qué — [3 min]

La pregunta del enunciado: ¿dónde, cómo y por qué un transformer? Nuestra respuesta sale del
EDA: la señal de este problema es **relacional** — el precio importa *en relación al rango del
filtro*; la U del precio está condicionada *al tier*. Un modelo lineal tiene que recibir esos
cruces hechos a mano; la **self-attention los computa de a pares, en forma aprendida** — es la
generalización de las feature crosses. Por eso la arquitectura base es un **FT-Transformer**:
cada feature es un token.

¿Cómo puede la atención comparar un precio con una marca? Porque no hace falta que los tokens
sean "del mismo tipo" — hace falta que vivan en el **mismo espacio ℝ^d**, y eso es exactamente
lo que hace el tokenizador de features: cada numérica entra como x·w+b con vectores aprendidos,
cada categórica con su codificación. Es el mismo principio por el que en un LLM conviven
"perro", una coma y un número.

El resto es un encoder estándar, y acá está el mapeo con la teoría de las clases 1 y 2:
usamos los **mismos bloques de la demo de la cátedra** — Head, MultiHeadAttention, FeedForward,
Block pre-LN con residuales — con dos adaptaciones justificadas. Uno: **sin máscara causal** —
esto es clasificación de un conjunto, no generación autoregresiva; atención bidireccional como
BERT. Dos: el escalado por √d_k se hace **una** vez, como el paper (la demo escalaba dos veces).
Agregamos un token **[CLS]** de lectura — no aporta información, la *recolecta*: como atiende a
todo en cada capa, su estado final es el resumen para clasificar (clase 2, BERT). Y **sin
positional encoding**: un conjunto de features no tiene orden — la identidad de cada columna ya
vive en sus parámetros propios. No lo asumimos: lo medimos, agregar PE da delta ≈ 0. Tamaño
inicial siguiendo la sugerencia del enunciado: d_model=32 (<100), 4 cabezas, 2 bloques.

`→ en pantalla`: el diagrama completo de la arquitectura (del CSV a p(bought)).

## 9 · Alternativas consideradas — y medidas — [1.5 min]

Antes de comprometernos evaluamos dónde MÁS podía ir el transformer, porque "que use un
transformer" no fija dónde. Las tres alternativas serias: (a) **el texto crudo como tokens** —
la demo literal, caracteres de título+descripción; la señal está ahí, pero el EDA mostró que se
extrae con una regex — gastar atención cuadrática sobre 257 tokens para eso es caro; (b)
**los productos de la página como tokens** (listwise) — modela la competencia dentro de la
búsqueda, pero el EDA midió competencia débil (se compran varios productos por página); (c)
**el transformer solo como encoder de texto** + un MLP clasificador. Elegimos features-como-
tokens como base… pero no descartamos en papel: **implementamos y corrimos las tres** (y dos
variantes más). Los resultados les dan la razón a los diagnósticos del EDA — los vemos en la
diapositiva 14 y en la tabla final.

## 10 · Baselines: la vara — [1 min]

Escalera de complejidad antes del transformer, con el mismo split y las mismas métricas:
regresión **logística** 0.698 — la vara lineal; **GBM** 0.762 — la vara no lineal fuerte,
árboles con interacciones; y un **MLP** denso con exactamente los mismos embeddings de entrada
que el transformer: 0.746. Regla de honestidad que nos pusimos: si el transformer no supera
esto, la capa de atención no se justifica.

## 11 · Experimento 1: ¿la atención aporta? — [2 min]

La comparación central, apareada por seed (mismo split): transformer vs MLP **con la misma
entrada** — la única diferencia es qué mezcla los tokens. Resultado: **+0.048, gana en 5 de 6
seeds**, con el MLP teniendo 4,5 veces más parámetros. La atención aporta y no es cuestión de
tamaño.

Dos refinamientos honestos que nos parecieron importantes. Uno: al MLP le probamos también
one-hot crudo en vez de embeddings y mejoró a 0.797 — parte del déficit del MLP era su entrada;
contra el *mejor* MLP posible, la ventaja del mejor transformer es +0.027. Sigue ganando, con la
vara más alta. Dos: ¿la atención solo "descubrió" el cruce precio×tier que ya sabíamos del EDA?
Le dimos ese cruce a mano a la logística: mejora +0.015, pero eso explica **solo el 12%** del
gap — la atención aprende bastante más que esa única interacción.

## 12 · Experimento 2: capacidad y entrenamiento — [1.5 min]

Grilla de capacidad estilo paper chico: d_model {8,16,32,64} × cabezas {1,2,4} × bloques
{1,2,4}, más el protocolo de entrenamiento. Hallazgos: **una cabeza grande le gana a cuatro
chicas** (+0.018) — coherente con que la señal dominante es una sola: una consulta "rica" vale
más que cuatro pobres; d64 suma poco; más paciencia en el early stopping (8→20) mejora en 21 de
24 configs tabulares. Pero al combinar los ganadores individuales, no suman: meseta en ~0.816.
La lección que nos llevó al experimento decisivo: **el eje ganador no era capacidad**.

## 13 · Experimento 3: la codificación de las categóricas — [2.5 min]

La sugerencia del enunciado era investigar codificaciones — one-hot y alternativas. El menú que
implementamos y corrimos, todo lo demás fijo: primero, un resultado teórico que ahorra un
experimento: **one-hot seguido de proyección lineal aprende exactamente la misma matriz que un
embedding** — son el mismo modelo para el transformer; por eso one-hot solo se prueba como
entrada cruda al MLP. Después: **embedding aprendido** por columna (el estándar FT-Transformer),
**target encoding** (nivel → BTR promedio suavizado de train), **ordinal** (nivel → su *rango*
al ordenar por ese BTR, normalizado), **frequency** y **hashing**.

Resultado — y acá está la sorpresa del TP: **ordinal global gana: 0.824**, por encima del
embedding (0.798) y del target (0.813). Los contraejemplos calibran el porqué: frequency (0.22)
y hashing (0.50) *destruyen* la señal — la frecuencia no correlaciona con comprar, y las
colisiones del módulo mezclan tiers. La regla: la codificación debe **preservar la relación
nivel→propensión**. ¿Y por qué ordinal le gana al embedding, si el embedding puede aprender
cualquier cosa? Porque en 10k filas, "poder aprender cualquier cosa" es overfitting: el rango
inyecta como *prior* el orden que el embedding tendría que aprender, con un escalar en vez de
una tabla — 26k parámetros totales. Y le gana a target porque los rangos equiespaciados están
mejor condicionados que las magnitudes apelmazadas (0.65/0.03/0.000). Reflexión honesta: nuestra
hipótesis previa era exactamente la inversa (embedding ≥ target ≥ ordinal) — los datos nos
corrigieron, y esa corrección es el mejor modelo del TP.

`→ en pantalla`: tabla del menú con resultados; el campeón resaltado.

## 14 · Experimento 4: ¿y el texto? — [2.5 min]

La señal nace en el texto — ¿el transformer puede leerla solo, sin nuestra regex? Corrimos el
arco completo. **Texto puro** (caracteres como tokens, la demo adaptada de decoder a encoder):
0.652 — muy por encima del techo sin-señal (0.16): *encontró el sufijo solo*; muy por debajo del
tabular: leer caracteres con 36k parámetros cuesta. **Híbrido** (features + 256 caracteres en
una secuencia): 0.705 — ¡peor que tabular solo! Los 256 tokens de texto *diluyen* la atención
sobre los 13 que importan. Pero el dato fino: al híbrido, sacarle el token parseado no le cuesta
nada — **recupera desde los caracteres crudos lo que extraía la regex**; a la torre (texto
comprimido a un solo embedding + MLP) sí le cuesta −0.04: su cuello de botella de 32 dims no
deja pasar la señal entera. Y la **fusión** — comprimir el texto a UN token que entra a la
secuencia tabular — cura la dilución por completo (+0.069 sobre el híbrido, 6 de 6) pero empata
exacto con la torre: una vez comprimido, cruzar por atención o por concatenación da igual.

Conclusión de diseño: el texto crudo es *recuperable* pero *redundante* cuando el EDA ya parseó
la señal — y la moraleja del tokenizador: probamos word-level y word2vec pre-entrenado (la
conexión clase 1→2: el pre-training regulariza, +0.010) y aún así el tokenizador chico de
caracteres de la demo resultó el correcto a esta escala.

`→ en pantalla`: el gráfico de barras del arco textual.

## 15 · Desafíos encontrados — [2 min]

Cuatro que valen la pena contar. **El causal degenerado**: la ablación "¿importa la máscara
causal?" dio ROC exactamente 0.500. No era "causal es peor": con máscara causal, nuestro [CLS]
en la posición 0 solo podía verse a sí mismo — el modelo predecía una constante (verificado:
p=0.2214 para todo test). El fix es ponerlo al final, como lee GPT; con eso la respuesta real
es "la bidireccionalidad da igual acá". Nos llevamos la lección de arquitectura: los decoders
leen desde el último token, los encoders pueden poner el CLS adelante. **Las trampas del
dataset**: cart y los timestamps rotos — el EDA los cazó antes de que muerdan. **El cómputo**:
la familia de texto es inviable en CPU (atención 257²); armamos una suite de experimentos
resumible que corre todo en una GPU consumer y deja cada corrida registrada con sus 16 métricas.
**Hipótesis refutadas con datos** — las contamos porque el método importa: bins por cuantiles
para la U del precio (el FFN ya la captura), pos_weight para el desbalance (daña: PR-AUC es de
ranking), mean pooling (CLS gana 6/6), positional en features (Δ≈0, como predice la teoría).

## 16 · El modelo final — [2 min]

**FT-Transformer con encoding ordinal**: 13 feature-tokens + [CLS], d_model 32, 4 cabezas, 2
bloques pre-LN, sin positional, BCE, AdamW, early stopping por validación con paciencia 20.
**26.177 parámetros** — el más chico del top 5.

Cómo lo elegimos, con disciplina: por validación había un empate técnico entre cuatro configs
(Δ < 0.002 — validación no puede distinguirlas). Desempatamos por **parsimonia**: menos
parámetros, menor desvío entre seeds, y el menor gap val→test. Recién después miramos test, que
confirma. Números finales: **PR-AUC test 0.824 ± 0.018** (6 seeds) · GroupKFold 5×6 **0.821 ±
0.012** · ROC 0.975 · F1 máximo 0.784 en umbral 0.40 · y con ensemble — de configuraciones o de
inicializaciones, dos rutas independientes que convergen — **0.834**. Contra las varas: GBM
0.762, mejor MLP 0.797.

## 17 · Robustez — [1.5 min]

Tres verificaciones sobre el modelo final. **Curva de aprendizaje**: 0.758 → 0.780 → 0.817 →
0.824 al 25/50/75/100% de los datos — casi saturada (el último cuarto aporta +0.007), y con el
75% de los datos ya le gana al GBM entrenado con todo. **Varianza**: una grilla de 5
inicializaciones × 6 splits mostró que el ±0.018 es mitad lotería del split, mitad
inicialización — valida el protocolo de promediar seeds, y esa mitad de init es justo lo que el
ensemble elimina. **Calibración**: ECE ~0.01 y temperatura ≈ 1 — importa porque el BTR de
negocio es el *promedio* de las probabilidades: nuestro promedio es un estimador confiable, sin
corrección.

`→ en pantalla`: la curva de aprendizaje.

## 18 · ¿El modelo mira donde debe? — [2 min]

Con 14 tokens, la matriz de atención se puede *mirar*. En la capa 1, el [CLS] pone el **75% de
su atención en el token de estado**, y la familia de precio se consulta entre sí — `price` y
`filter_min` atienden a `price_rel`: la señal relacional, literal en el mapa. La capa 2 mezcla.
Y como "attention is not explanation", lo contrastamos con un diagnóstico independiente basado
en resultados: **importancia por permutación** — destruir `listing_status` cuesta 0.68 de
PR-AUC; `price_rel`, 0.14; `allergens`, 0.05; el resto, ~0. Dos métodos independientes, la misma
historia — que es exactamente la del EDA. El círculo cierra.

Y la traducción al negocio: en las páginas de test con al menos una compra, el producto que el
modelo rankea primero fue efectivamente comprado el **91%** de las veces (azar: 27%).

`→ en pantalla`: mapa de atención + barras de importancia, lado a lado.

## 19 · Ejercicio 3: personalización (teórico) — [2.5 min]

¿Cómo haríamos que el BTR dependa de *quién* busca? Hoy nuestro modelo es
p(bought | producto, búsqueda); personalizar es condicionar también al usuario:
p(bought | producto, búsqueda, **usuario**). Hace falta primero **dato nuevo**: `user_id` y su
historial de eventos — el dataset actual no lo trae.

La extensión natural de NUESTRA arquitectura: así como el texto entra como tokens, **el
historial del usuario entra como tokens** — sus últimas compras/búsquedas, cada una codificada
con el mismo tokenizador de productos — y la secuencia los atiende (estilo BST/SASRec, que es
exactamente esto en producción). El usuario que compra comida de perro todos los meses tiene
esos productos en su historial; cuando aparecen en la página, la atención cruza historial ↔
candidato y sube su probabilidad. La alternativa clásica que conecta con la clase 2:
**embeddings de usuario y producto entrenados con negative sampling** — item2vec/two-tower,
skipgram donde el "contexto" son los productos con los que el usuario interactuó — útil como
*retrieval* si el catálogo fuera enorme, con nuestro modelo como *ranker* encima. Y el detalle
que este TP nos dejó bien aprendido: el usuario nuevo sin historial es el mismo problema que el
producto nuevo sin estado — cold-start — y el fallback es exactamente el modelo que ya tenemos,
que no depende del usuario.

## 20 · Conclusiones — [1 min]

Cinco, cortas. **El EDA mandó**: la señal estaba escondida en un sufijo de texto, y todo el
diseño sale de haberla encontrado. **La formulación correcta valió más que el modelo grande**:
el campeón tiene 26k parámetros. **La atención aporta, medida con vara honesta**: +0.048 contra
su gemelo sin atención, +0.027 contra el mejor MLP. **La mejor codificación fue un prior
simple**: el encoding ordinal derivado de los datos le ganó al embedding — y refutó nuestra
propia hipótesis, que es lo que uno quiere de un experimento. **Y el modelo es auditable**: la
atención mira el estado, la permutación lo confirma, la calibración permite leer el promedio de
p como BTR, y elige bien el producto a promocionar el 91% de las veces. Número final: PR-AUC
0.824 ± 0.018 — 0.834 en ensemble — contra 0.762 del GBM. Gracias — preguntas.

---

### Apéndice para preguntas (no se presenta, se defiende)

- **¿Por qué PR-AUC y no F1?** F1 requiere umbral; el uso es ranking. Igual: F1 máx 0.784 @ 0.40.
- **¿Overfitting?** Curvas train/val por época en cada corrida (panel); early stopping por val;
  gap val→test ≈ +0.01/0.03 (selección normal); dropout 0.1; el encoding ordinal es en sí
  regularización (y MLM sobre features, probado, regulariza a los embeddings +0.011 pero no
  agrega sobre ordinal).
- **¿Por qué no CV desde el inicio?** La cátedra pidió priorizar promedio de corridas; CV 5×6 al
  final: 0.821 ± 0.012, consistente.
- **¿El split por producto?** Verificado: métricas idénticas sobre productos jamás vistos.
- **¿El estado no es hacer trampa?** Familia intrínseca medida: sin estado, techo ~0.16 para
  cualquier modelo (61% de filas en tiers de BTR = 0). Es información de estado del catálogo,
  válida al predecir, circular para promover — por eso las dos familias.
- **¿Multi-task con cart?** Probado (λ ∈ {0.1, 0.3, 0.5} como label auxiliar): dentro del ruido.
- **¿Probaron especializar los pesos por feature? ¿O achicar más?** Sí (5ª tanda, exploratoria,
  posterior al cierre de la selección): desatar W_q/W_k/W_v/FFN por posición empata con el
  campeón (y SÍ ayuda +0.02, 6/6, sobre embeddings — el beneficio existe pero es redundante con
  el prior ordinal); y el modelo admite compresión 7×: `min_d16l1` (d16, 1 bloque, **3.713
  parámetros**) empata al final. Bonus metodológico: la config con mejor validación de todo el
  proyecto (pf_ffn, 245k params) NO es mejor en test — sobreajuste de selección en vivo, la
  razón por la que la selección se cerró con procedimiento pre-registrado.
- **¿Cuántas corridas hay detrás?** 604 (77 configuraciones × 6 seeds + grillas), todas con las
  16 métricas por época, reproducibles con la suite del repo.
