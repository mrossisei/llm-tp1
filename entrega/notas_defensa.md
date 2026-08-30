# Notas de defensa — fundamentos técnicos

Apunte de estudio para la defensa oral. No es parte del entregable: es el material conceptual
detrás de cada decisión, con los números propios que la respaldan.

Cada punto trae una **respuesta corta** lista para decir.

---

# A · Atención y el bloque transformer

## A0 · Qué es una secuencia acá

En un LLM los tokens son palabras. Acá son **features**: 13 columnas → 13 tokens, más el `[CLS]`
adelante = **14 tokens**. Cada token es un vector de **32 números** (`d_model`). Toda la fila del
CSV, dentro del modelo, es una tabla de 14 × 32.

| número | qué es | valor |
|---|---|---|
| n | tokens de la secuencia | 14 (1 CLS + 13 features) |
| d_model | números por token | 32 |
| num_heads | cabezas de atención | 4 |
| head_size | a cuánto comprime cada cabeza | 8 |
| n_layer | bloques apilados | 2 |

Relación clave: `head_size × num_heads = d_model`. Las cabezas **reparten** d_model, no lo
multiplican.

## A1 · Q, K, V

La atención decide, **para cada fila**, cuánta importancia darle a cada dato. Los pesos cambian
producto a producto — eso es lo que la separa de una regresión logística, que tiene pesos fijos.

| | analogía de la mesa | qué es |
|---|---|---|
| **query (q)** | la pregunta que hace el token | qué está buscando |
| **key (k)** | el cartel que levanta cada token | qué ofrece |
| **value (v)** | lo que dice cuando lo escuchan | el contenido que se copia |

`wei = q @ k.transpose(-2,-1)` compara cada pregunta contra cada cartel (producto punto:
multiplicar posición por posición y sumar). `wei[i][j]` = cuánto le interesa a i el token j.

Q y K deciden **cuánto**. V es **qué**.

## A2 · Por qué dividir por √d_k

`k.shape[-1] ** -0.5` = 1/√8 ≈ 0.354 (8 = head_size).

Sin dividir, con puntajes `[8, 2, 1, 0]` el softmax da `99.6% / 0.25% / 0.09% / 0.03%`.
Dividiendo: `79% / 9.5% / 6.7% / 4.7%`.

Dos problemas del primero:
1. **Tira información**: mira una sola feature y descarta las otras doce.
2. **Deja de aprender**: en la zona saturada, mover el puntaje casi no cambia el porcentaje →
   gradiente ≈ 0 → los pesos de Q y K quedan congelados en su inicialización.

Por qué √d: el producto punto suma d términos que se cancelan parcialmente, así que el total
típico crece como √d. Dividir por √d lo deshace.

**Matiz:** la división no impide que el modelo esté seguro — puede aprender puntajes más grandes.
Evita que se sature **por accidente, antes de aprender nada**.

> **Respuesta corta:** "Sin el escalado los puntajes crecen con la dimensión de la cabeza y el
> softmax se satura: la atención colapsa a una feature y el gradiente se va a cero. Dividir por
> √d_k normaliza la varianza a 1 y deja el softmax donde hay señal para aprender."

**Bug de la cátedra:** la demo escalaba **dos** veces (= dividir por 8), lo que aplana la
atención a `44/21/18/16` — casi uniforme. Nuestro código lo aplica una sola vez, como el paper.

## A3 · El softmax y la tabla 14 × 14

`F.softmax(wei, dim=-1)` normaliza **por filas**. Cada fila suma 100%.

**Preguntan los 14, no solo el CLS.** Por eso la tabla es de 14 × 14: una fila por token. Eso es
lo que permite que `price` mire a `filter_price_max` y se entere de si $4.50 es caro *en esta
búsqueda*. Después de la atención, el token del precio ya no significa "cuesta 4.50" sino "está
en el tercio bajo de lo que buscaban".

Es lo mismo que en un LLM: después de la atención, `banco` significa "banco de plaza" o "banco de
dinero" según el contexto.

## A4 · Cuatro cabezas

Una cabeza = **un solo presupuesto de 100%**. Si le das 65% al tier, no te queda para el precio.
Cuatro cabezas = cuatro presupuestos independientes, cada uno con su criterio.

**Las cabezas NO se reparten los tokens ni cortan el embedding.** Cada cabeza lee los 32 números
completos de los 14 tokens y **fabrica sus propios 8** con `Linear(32, 8)` — pesos propios. Como
cuatro personas leyendo el mismo informe de 32 páginas y escribiendo cada una su resumen de 8
líneas.

Costo en parámetros: **idéntico.**
```
1 cabeza de 32:   3 × 32 × 32 = 3.072
4 cabezas de 8:   4 × 3 × 32 × 8 = 3.072
```

Medido en el TP (ambos con 26.177 parámetros exactos):

| | test PR-AUC |
|---|---|
| 4 cabezas (`feat_ordinal`) | **0.824 ± 0.018** |
| 1 cabeza (`camp_ordinal_h1`) | 0.800 ± 0.031 |

+0.024 gratis, **y la mitad de varianza** — con una cabeza mal inicializada no hay redundancia.

`self.proj` (Linear 32→32) mezcla las cuatro cabezas después de concatenarlas. Sin eso serían
cuatro tuberías paralelas que nunca se cruzan.

## A5 · El FeedForward (= un MLP)

```python
nn.Linear(32,128), nn.ReLU(), nn.Linear(128,32)
```

Es literalmente un MLP de dos capas. El docstring del código lo dice: *"MLP interno del bloque"*.
En GPT-2 la clase se llama `MLP`.

**El límite que resuelve.** La atención produce un promedio ponderado → cada entrada empuja
siempre en la misma dirección. Solo puede aprender "más es mejor" o "menos es mejor".

**El contraejemplo de nuestros datos**, medido:

| price_rel | tasa de compra |
|---|---|
| 0.0 – 0.2 | 0.075 |
| 0.2 – 0.4 | 0.134 |
| 0.4 – 0.6 | **0.176** |
| 0.6 – 0.8 | 0.134 |
| 0.8 – 1.0 | 0.090 |

U invertida. Ningún peso `w` produce "sube y después baja".

El ReLU (`max(0, z)`) introduce un **codo**. Con dos ReLU ya armás la U:
`f(p) = ReLU(p) − 2·ReLU(p − 0.5)` → 0, 0.25, 0.50, 0.25, 0. El FFN tiene 128 codos.

**División del trabajo (frase para la defensa):**
> **La atención mezcla información entre tokens. El FeedForward la transforma dentro de cada
> token.** Sin la primera no hay contexto; sin la segunda no hay no linealidad.

Se aplica **token por token**, con los mismos pesos (*position-wise*).

## A6 · El `x +` (residual)

`x = x + f(x)` en vez de `x = f(x)`. Tres efectos:

1. **Lo que el token ya era no se pierde** — acumula en vez de sobrescribir.
2. **Cada capa aprende solo el delta.** Una capa inútil aprende cero y la suma la deja pasar. Sin
   residual tendría que reconstruir la identidad.
3. **El gradiente baja intacto** por el camino directo de la suma. Sin esto, un modelo de 96
   capas no entrena (gradiente desvaneciente).

En el modelo: el CLS arranca como embedding aprendido y va **acumulando** aportes bloque a bloque.
Es lo que en interpretabilidad se llama *residual stream*.

## A7 · LayerNorm y pre-LN

**Qué normaliza:** las **activaciones** (los 14×32 números que circulan), no los pesos. Y **cada
token por separado** — 14 normalizaciones independientes, cada una con su propio promedio y
desvío. Por eso se llama *Layer*Norm.

```
post-LN (paper 2017):    x = LayerNorm(x + atención(x))     ← norm ENCIMA de la suma
pre-LN (nuestro, GPT):   x = x + atención(LayerNorm(x))     ← norm DENTRO de la rama
```

En post-LN el gradiente tiene que atravesar una normalización en cada capa: la autopista del A6
queda con peajes. Por eso los transformers post-LN **no entrenan sin warmup** del learning rate.

Pre-LN entrena estable, sin warmup, y tolera lr más altos. Por eso GPT-2 en adelante lo usa.

Hay un tercer LayerNorm, `ln_f`, después de los dos bloques: con pre-LN la salida del último
bloque nunca pasó por una normalización.

**Este es el segundo apartamiento documentado respecto de la demo**, junto con el escalado.

## A8 · El costo O(n²)

Cada token compara contra los 14 → 14² = **196 comparaciones**. Duplicar los tokens cuadruplica
las comparaciones. Y la tabla hay que **guardarla en memoria** para el backprop, por cabeza y por
capa.

n = 14 y es **fijo**: toda fila del CSV produce 14 tokens.

Con 100.000 tokens de contexto son 10.000 millones de celdas. Es el cuello de botella central de
los LLM, y el motivo de FlashAttention, atención con ventana, atención dispersa y Mamba.

---

# B · Encoder vs decoder

## B1 · Quién ve a quién

| | tabla de atención | ejemplo | para |
|---|---|---|---|
| **encoder** | completa, todos ven a todos | BERT, **nuestro modelo** | entender / clasificar |
| **decoder** | solo hacia atrás (máscara causal) | GPT | generar |

GPT necesita la máscara para que la tarea (predecir la próxima palabra) no sea trivial. Nosotros
no tenemos "siguiente token" que adivinar, y necesitamos las dos direcciones: el precio tiene que
ver el rango y el rango tiene que ver el precio.

Implementación: los puntajes prohibidos van a −infinito antes del softmax → `e^-inf = 0`.

## B2 · El causal degenerado

| | test PR-AUC |
|---|---|
| sin máscara (encoder) | 0.794 |
| con máscara, **CLS en posición 0** | **0.124** |
| con máscara, CLS al final | 0.795 |

Tasa base = 0.131. **0.124 es peor que el azar.**

Causa: en posición 0, con máscara causal, el CLS solo se ve a sí mismo. Nunca ve una feature. Y
como es el mismo vector aprendido para todas las filas, **la predicción es una constante**. ROC
exactamente 0.500, que es lo que da una predicción constante por definición.

Las otras 13 posiciones sí aprendían — pero al final solo se lee el CLS, así que se tira todo.

**La conclusión precisa:** la máscara no aporta (0.795 vs 0.794) ni daña si el token de lectura
está bien ubicado. Lo catastrófico es la **combinación incoherente**.

## B3 · Por qué el `[CLS]` y no un promedio

| forma de resumir 14 tokens → 1 | ¿pesos adaptativos? | PR-AUC |
|---|---|---|
| **token `[CLS]`** | **sí, vía atención** | **0.794** |
| promedio simple de los 13 | no (todos iguales) | 0.764 |
| promedio ponderado por posición | no (fijos, aprendidos) | 0.732 |

El promedio ponderado da **peor** que el simple: agrega parámetros que codifican la posición, que
acá es arbitraria.

Segundo motivo: como el CLS es el único token que llega a la salida, el entrenamiento **obliga** a
empujar la información útil hacia él.

Tercer motivo: su fila de atención es **legible** — "42% a listing_status" es una explicación.

## B4 · Por qué no hay positional encoding

Las 13 features son un **conjunto**, no una secuencia. `price` está en la posición 4 porque
alguien escribió la lista así. Sin positional, el modelo es **invariante a permutaciones**, que es
la propiedad correcta.

| | PR-AUC |
|---|---|
| sin positional (`feat_base`) | **0.794** |
| con positional (`feat_pos`) | 0.762 |

**Agregarlo empeora.** Es una señal vacía en la que el modelo encuentra patrones espurios.

## Resumen B

| | nuestro modelo | GPT |
|---|---|---|
| quién ve a quién | todos | solo hacia atrás |
| máscara causal | no | sí |
| cómo resume | token `[CLS]` | el último token |
| positional | no | sí |
| familia | BERT | GPT |

**Tiene arquitectura transformer pero no es un language model:** no genera, no predice el
siguiente token, no tiene noción de orden. Es un clasificador encoder-only.

---

# C · Tokenización y encoding

## C1 · Cómo un dato se vuelve un embedding

**Numéricas (6):** z-score (con `log1p` previo en `price` y `net_weight_oz`, sesgadas a derecha) →
`x.unsqueeze(-1) * num_weight + num_bias` → 32 números.

**Categóricas (7):** rango ordinal en [0,1] → misma proyección afín → 32 números.

O sea: **una vez convertida a número, una categórica se trata igual que una numérica.**

Costo: 13 × (32 pesos + 32 sesgos) = **832 parámetros**, el 3% del modelo.

Dos detalles: las tablas se ajustan **solo con train**, y el índice 0 es **UNK**.

## C2 · Ordinal vs embedding

| | embedding clásico | ordinal |
|---|---|---|
| parámetros por columna | 32 × niveles | **64** |
| qué aprende cada nivel | 32 números libres | nada, hereda su posición |
| relación entre niveles | ninguna a priori | ordenados por propensión |

Con 200 marcas: 6.400 parámetros contra 64.

| | test PR-AUC |
|---|---|
| **ordinal** | **0.824 ± 0.018** |
| embedding (`feat_base`) | 0.794 ± 0.034 |

**Por qué gana:** con ~7.000 filas de train, una marca que aparece 12 veces no alcanza para
aprender 32 números — aprende ruido. Estimar **un** número con suavizado (`SUAVIZADO_M = 50`) es
robusto. Y el rango le inyecta como prior el orden que el embedding tendría que descubrir solo.

**Matiz honesto:** el ordinal impone una recta. Pierde expresividad si dos niveles muy distintos
tuvieran la misma tasa. Se acepta porque con 7.000 filas el trade-off es claramente favorable.
Un LLM hace lo contrario porque tiene miles de millones de ejemplos.

## C3 · Los cinco encodings

| encoding | qué número le pone a un nivel | PR-AUC | ROC |
|---|---|---|---|
| **ordinal** | su puesto en el ranking de compra | **0.824** | 0.975 |
| target | su tasa de compra directa | 0.813 | 0.973 |
| embedding | 32 números libres | 0.794 | 0.974 |
| hashing | el grupo que le tocó (1 de 8) | 0.498 | 0.880 |
| frecuencia | cuántas veces aparece | 0.218 | 0.642 |

- **target pierde** porque la tasa cruda se dispara con pocos datos (3 apariciones, 3 compras →
  1.00). El ranking la pone "primera" sin exagerar la distancia.
- **hashing colapsa** porque agrupa niveles al azar y destruye los 21 niveles de `listing_status`.
- **frecuencia es la peor** porque codifica **popularidad**, que no tiene relación con propensión
  de compra. Señal sistemáticamente engañosa. Desvío 0.095 — ni siquiera falla consistente.

**El dato que justifica PR-AUC:** hashing tiene ROC 0.880 (suena razonable) y PR-AUC 0.498 (se
rompió). Y el ROC separa 0.001 entre el 1° y el 2°, mientras el PR-AUC separa 0.030.

**Rango completo: 0.218 → 0.824. Más que cualquier decisión de arquitectura del TP.**

## C4 · one-hot + lineal ≡ embedding

`[1,0,0,0] × W` = la fila 1 de W. Que es exactamente lo que hace el lookup de un embedding.
Mismos parámetros, mismo resultado; el embedding solo evita multiplicar por ceros.

Por eso no hizo falta correr one-hot en el transformer: se sabe *a priori* que daría 0.794.

## C5 · Todo al mismo espacio = FT-Transformer

Los dos caminos (ordinal y z-score) convergen en la misma proyección afín, así que **después del
tokenizador el modelo no distingue tipos**. Eso es lo que habilita que la atención compare el
token de `listing_status` con el de `price_rel`.

La arquitectura tiene nombre: **FT-Transformer** (Gorishniy et al., 2021). Idea central: *cada
feature es un token*.

---

# D · Datos y leakage

## D1 · La unidad de análisis

Cada fila es una **impresión**: un producto mostrado en una búsqueda concreta.

```
10.000 impresiones · 2.012 búsquedas · 1 a 8 productos por búsqueda
bought = true: 1.301 (13,01%)   ← la tasa base
```

Compras por búsqueda: 1.058 con 0 · 670 con 1 · 226 con 2 · 53 con 3 · 5 con 4.

**284 búsquedas tienen 2 o más compras** → no es "elegí uno entre 8". Por eso se predice cada
impresión de forma independiente, y por eso la variante *listwise* dio 0.740.

## D2 · El leakage de `cart`

```
cart = true          3.007 (30%)
bought = true        1.301 (13%)
cart Y bought        1.301
cart=false Y bought=true    →  0 filas
```

**De las 6.993 filas con `cart=false`, ninguna tiene `bought=true`.** El 70% del dataset queda
resuelto de forma determinista.

`p(bought | cart=true) = 43%` → **1.706 carritos abandonados**. Así que `cart` no resuelve todo,
pero resuelve el 70% trivialmente.

**El motivo real de la exclusión es temporal:** el carrito ocurre *después* del momento en que hay
que rankear. En producción esa columna llega vacía.

Cuantificado con un modelo simple sobre los datos reales:

| features | average precision |
|---|---|
| azar | 0.138 |
| solo `listing_status` | 0.659 |
| solo `cart` | 0.454 |
| **`cart` + `listing_status`** | **0.924** |
| nuestro modelo honesto | 0.824 |

Los 0.10 de diferencia son artificiales.

**Los tres tipos de leakage, y cómo los cubrimos:**

| tipo | ejemplo | cómo se evita |
|---|---|---|
| temporal | `cart` | excluida |
| del target | calcular el ranking ordinal con todo el dataset | tablas ajustadas solo con train |
| de partición | filas de la misma búsqueda en train y test | split por `query_id` |

## D3 · Las dos features derivadas

**`listing_status`** — regex al sufijo del título. 20 niveles, tres tiers netos:

```
#1 Pick 0.625 · Customer Favorite 0.677 · Top Rated 0.627 · Best Seller 0.657
Well Reviewed 0.038 · Shopper Favorite 0.028 · Highly Rated 0.021 · Popular Choice 0.019
los otros 12 niveles:  0.000 exacto  (~500 filas cada uno)
```

**El hallazgo:** "Top Rated" 0.627 contra "Highly Rated" 0.021 — suenan igual, difieren 30×.
Lo mismo "Customer Favorite" 0.677 vs "Shopper Favorite" 0.028.

Esto justifica el encoding ordinal (**el orden se saca contando, no leyendo**) y explica por qué
un modelo de lenguaje pre-entrenado falla acá: BERT vería esas frases como sinónimas.

**`price_rel`** = `(price − filter_min) / (filter_max − filter_min)`. Señal **relacional**: $4.50
es barato si buscaste de $3 a $10 y caro si buscaste de $1 a $5. Produce la U invertida de A5.

**Ninguna de las dos venía en el CSV, y las dos valen más que cualquier ajuste de arquitectura.**

## D4 · Exclusiones

| columna | motivo |
|---|---|
| `cart` | leakage temporal estricto |
| `query_id` | solo particiona |
| `timestamp` | **roto**: dentro de una misma búsqueda hay hasta 2 años de diferencia |
| `package_size`, `dimensions_in`, `ingredients` | redundantes — ablación con Δ ≈ 0 |
| `filter_category`, `filter_storage_type` | los productos mostrados siempre los cumplen |

Sobre el timestamp: el rango global es exactamente 2 años (2024-07-08 a 2026-07-08) y las fechas
están repartidas al azar. Es ruido del generador. Verificado derivando hora y día de la semana:
delta cero.

**Una feature de puro ruido no es neutra:** el modelo le busca patrones y encuentra coincidencias
falsas. Mismo fenómeno que el positional encoding de B4.

## D5 · `cart` como target auxiliar

`cart` no puede ser entrada, pero sí **segunda salida**: `loss = loss_bought + λ · loss_cart`.
No es leakage porque el modelo la **predice**, no la recibe.

| λ | test PR-AUC |
|---|---|
| sin tarea auxiliar | **0.824** |
| 0.3 | 0.809 |
| 0.1 | 0.796 |
| 0.5 | 0.791 |

**Ninguno mejora.** Explicación probable: `bought ⟹ cart`, así que las dos tareas son casi la
misma y la auxiliar no agrega información — solo le resta peso a la principal.

---

# E · Métricas y evaluación

## E1 · Por qué no accuracy

13% de positivos → decir "nadie compra" acierta el 87%. Y para el caso de uso (rankear) el
accuracy ni siquiera mide lo que importa.

## E2 · PR-AUC

Área bajo la curva precision-recall: resume el ranking **en todos los puntos de corte a la vez**,
sin elegir umbral.

**El PR-AUC de un modelo aleatorio es exactamente la proporción de positivos.** Por eso siempre se
reporta contra la tasa base: azar 0.131, modelo 0.824.

## E3 · Por qué no ROC como principal

Nuestro ROC es 0.975 y suena mejor. Pero con desbalance el ROC **engaña**, y tenemos la prueba:

| encoding | ROC | PR-AUC |
|---|---|---|
| ordinal | 0.975 | 0.824 |
| embedding | 0.974 | 0.794 |
| **hashing** | **0.880** | **0.498** |
| frecuencia | 0.642 | 0.218 |

El ROC premia acertar negativos, y hay 8.699 contra 1.301 positivos. El PR-AUC **ignora los
verdaderos negativos**.

## E4 · El split por `query_id`

70/15/15 **por búsqueda**, no por fila. Los productos de una misma búsqueda comparten el rango de
precio filtrado; repartirlos entre train y test filtra el contexto de la página.

Verificado además por producto: 99% de los títulos son únicos, y restringir test a productos nunca
vistos da métricas idénticas. El modelo no recibe identidad de producto, no hay canal para
memorizar.

## E5 · Seis semillas y comparaciones pareadas

Con desvíos de 0.02–0.03, **una diferencia de 0.01 entre configuraciones no significa nada.** Solo
diferencias de ~0.03 para arriba son creíbles.

Las configuraciones se comparan **semilla contra semilla** (pareadas), lo que elimina la
variabilidad de inicialización y hace la comparación más sensible.

Cierre de robustez: GroupKFold 5×6 = 30 corridas → **0.821 ± 0.012**, consistente con 0.824.

---

# F · Entrenamiento y regularización

```
loss           binary_cross_entropy_with_logits   (sigmoid incorporada, por estabilidad)
optimizador    AdamW, lr 1e-3
batch          256
epochs         hasta 300, corta ~64
early stopping por PR-AUC de VALIDACIÓN, paciencia 20
```

Corta por **PR-AUC de validación, no por loss**: es la métrica que importa y no siempre se mueven
juntas.

## El resultado incómodo

| configuración | test PR-AUC |
|---|---|
| **dropout 0 (sin nada de dropout)** | **0.828 ± 0.022** |
| feature dropout 0.1 | 0.826 |
| **dropout 0.1 ← el campeón** | **0.824 ± 0.018** |
| **sin dropout ni weight decay** | **0.824 ± 0.022** |
| weight decay 0 | 0.823 |
| weight decay 0.1 | 0.817 |
| dropout 0.3 | 0.813 |
| weight decay 0.001 | 0.812 |
| label smoothing 0.1 | 0.809 |
| feature dropout 0.2 | 0.795 |

**Ninguna regularización aporta.** Las cuatro primeras están dentro de 0.005, muy por debajo del
desvío de 0.02. Interpretación: el early stopping ya evita el sobreajuste — cortando en la época
64 de 300, el modelo nunca llega a memorizar.

**Lo que sí se ve es el daño del exceso:** dropout 0.3 baja 0.011; feature dropout 0.2 baja 0.029
y triplica el desvío.

**Honestidad:** el campeón usa dropout 0.1 y da 0.824; sin dropout da 0.828. Son
estadísticamente iguales. No decir "elegimos dropout 0.1 porque era mejor" — el campeón se fijó
antes de este barrido.

---

# G · Transfer learning (clase 3)

## G1 · Pre-entrenamiento MLM

Tapar features y reconstruirlas, 20 épocas, sin etiquetas. Después entrenar para `bought`.

| | PR-AUC |
|---|---|
| sin pre-entrenamiento | **0.824** |
| con MLM 20 épocas | 0.817 |

No aporta. **El MLM sirve cuando hay muchos datos sin etiquetar y pocos etiquetados.** Acá son los
mismos 7.000, y todos etiquetados.

## G2 · Feature extraction

| congelar el modelo de origen y entrenar solo una lineal encima | PR-AUC |
|---|---|
| desde MLM | **0.138** ← la tasa base |
| desde un modelo supervisado | **0.827** |

El MLM aprendió a reconstruir features, no a predecir compras. Congelado, no hay forma de
corregirlo.

**Lo que importa no es que haya pre-entrenamiento, sino que la tarea de origen esté alineada con
la de destino.**

## G3 · Destilación — el mejor resultado del TP

El teacher transmite **probabilidades** en vez de etiquetas duras: "no se compró, pero casi" (0.42)
contra "ni cerca" (0.03). Esa información no está en el `0/1`.

| student | parámetros | PR-AUC |
|---|---|---|
| **d8, 2 capas** | **1.937** | **0.828** |
| d16, 1 capa | 3.713 | 0.827 |
| d32 (= teacher) | 26.177 | 0.827 |
| — teacher original — | 26.177 | 0.824 |
| d8, 1 capa | 1.089 | 0.821 |
| d4, 1 capa | 353 | 0.794 |

**1.937 parámetros igualan a 26.177. Compresión 13,5×.**

El teacher es un **ensamble de las 6 semillas**, así que promedia el ruido de inicialización. Es
el mecanismo real detrás de DistilBERT y de los modelos chicos que corren en un teléfono.

## G4 · Transfer externo (MiniLM)

| configuración | parámetros | PR-AUC |
|---|---|---|
| **nuestro modelo, sin BERT** | **26.177** | **0.824** |
| MiniLM fine-tuneado | 22.751.713 | 0.811 |
| MiniLM congelado + MLP | 3.294.401 | 0.773 |
| MiniLM congelado como token | 38.497 | 0.751 |
| solo MiniLM, sin nuestras features | 3.187.073 | 0.567 |

**22,7 millones de parámetros dan 0.811 contra 26 mil que dan 0.824.**

**Por qué falla:** el dataset es sintético y el wording no correlaciona con el comportamiento —
"Top Rated" 0.627 vs "Highly Rated" 0.021, que BERT considera sinónimas. Su conocimiento
lingüístico es **activamente engañoso** acá.

El fine-tuning **repara** parte del daño (0.751 → 0.811), lo que confirma que el problema son las
representaciones y no la arquitectura. Pero ni reparado alcanza.

## Conclusión transversal de G

**El transfer learning no es magia. Funciona cuando la tarea de origen se parece a la de destino.
Cuando no, resta.**

---

# H · Conexión con LLMs

## Qué se comparte

Todo el bloque A, sin excepción: atención con Q/K/V y escalado por √d_k, softmax, multi-head, el
MLP interno, residuales, LayerNorm en posición pre-LN, bloques apilados.

**Si tomás nuestro `Block` y lo pegás en el código de GPT-2, funciona.**

## Desglose de los 26.177 parámetros

| componente | parámetros | % |
|---|---|---|
| tokenizador (13 × 64) | 832 | 3% |
| token `[CLS]` | 32 | 0,1% |
| atención (2 bloques × 4.128) | 8.256 | 32% |
| **FeedForward (2 × 8.352)** | **16.704** | **64%** |
| LayerNorms (4 × 64 + 64) | 320 | 1% |
| cabeza `Linear(32→1)` | 33 | 0,1% |
| **total** | **26.177** | |

**El 64% del modelo son los FFN — la misma proporción que en GPT.** Si alguien dice "el
transformer es la atención", la respuesta es que dos tercios de los parámetros son feedforward.

## Qué separa

| | nuestro modelo | GPT-3 |
|---|---|---|
| parámetros | 26.177 | 175.000.000.000 |
| d_model | 32 | 12.288 |
| capas | 2 | 96 |
| cabezas | 4 | 96 |
| datos | 7.000 filas | ~300.000M tokens |
| **tarea** | clasificación binaria | predecir el siguiente token |
| **etiquetas** | necesita `bought` | **ninguna** — el texto se etiqueta solo |
| generación | no | sí |
| familia | encoder (BERT) | decoder (GPT) |

**La línea de las etiquetas es la más profunda.** Somos *supervisados*: sin la columna `bought` no
hay entrenamiento, y por eso hay 7.000 ejemplos. Un LLM es **auto-supervisado**: la etiqueta es la
palabra que sigue, y ya está en el texto. Eso es lo que hizo posible la escala.

(Y de ahí sale la lección de G1: el MLM es la técnica correcta para un problema que no tenemos.)

## Lo que el TP demuestra sobre el campo

1. **La arquitectura no manda.** El encoding movió de 0.218 a 0.824; los ejes de capacidad, ~0.02.
   Los LLM llegaron a lo mismo por otro camino: el transformer casi no cambió desde 2017 y todo el
   progreso vino de datos, escala y entrenamiento.
2. **El transfer no es magia.** MiniLM con 22,7M pierde contra 26k. Contraargumento empírico a
   "usemos un LLM para todo".
3. **Los modelos se comprimen muchísimo.** 1.937 parámetros igualan a 26.177.

## La respuesta precisa

> "Es un transformer encoder-only —un FT-Transformer— para datos tabulares. Comparte con un LLM la
> arquitectura completa: atención multi-cabeza, MLP interno, residuales, pre-LN. **No es un LLM**:
> no procesa lenguaje, no genera, no predice el siguiente token, es supervisado y no
> auto-supervisado, y tiene 26 mil parámetros contra miles de millones."

## Sobre "¿es un transformer puro o transformer + MLP?"

**Transformer puro.** El MLP que se ve es el feed-forward interno de cada bloque, que forma parte
de la definición del transformer desde el paper original. Lo único fuera de los bloques es el
tokenizador (análogo del embedding) y una cabeza `Linear(32→1)` de 33 parámetros (análogo de la LM
head) — una sola capa lineal, sin no linealidad, así que no es un MLP.

Lo que **sí** sería una composición es la arquitectura `tower`, que probamos: transformer como
encoder de texto + un MLP clasificador separado. Da 0.775 contra 0.824.
