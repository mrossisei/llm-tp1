# TP1 — Predicción de Buy Through Rate: propuesta de solución

> Documento de trabajo del equipo. Primera propuesta: formulación del problema, hallazgos del
> EDA inicial, decisiones de diseño con sus alternativas, plan de experimentos y temas abiertos.

## 0. Resumen de la propuesta (TL;DR)

- **Qué predecimos:** `bought` a nivel impresión (cada fila = un producto mostrado en una búsqueda).
  La probabilidad estimada `p(bought)` **es** el BTR del producto en ese contexto; el BTR de negocio
  por producto se obtiene agregando (promediando) sobre sus impresiones.
- **Dónde va el transformer:** es el clasificador de BTR. Formulación principal: **encoder-only
  sobre "feature-tokens"** — cada feature del producto/búsqueda se convierte en un token (un vector
  de dimensión `d_model`), la self-attention modela **interacciones entre features**, y un token
  `[CLS]` agrega todo para la cabeza de clasificación (1 logit → sigmoide). Es la idea de
  FT-Transformer / AutoInt, construida con los mismos bloques (`Head`, `MultiHeadAttention`,
  `FeedForward`, `Block`) de la demo de la cátedra.
- **Variantes a comparar (implementadas):** (i) **texto crudo como tokens** — cada carácter de
  `title+description` es un token, como en la demo; secuencia ~257 con atención cuadrática → acá
  el cómputo es real y se justifica la GPU. Es el experimento central del TP: *¿descubre el
  transformer solo, desde los caracteres, la señal que nosotros parseamos a mano?* (ii) **híbrido**
  ([CLS] + features + caracteres en una secuencia). (iii) transformer "listwise" donde los tokens
  son **los productos de la misma query** (efecto débil según el EDA, pero medirlo vale).
- **Hallazgo clave del EDA:** la señal dominante está **escondida en el texto**: el sufijo del
  título (p. ej. `(Best Seller)`, `(Limited Feedback)`) define tiers con BTR ≈ 0.65 / ≈ 0.03 / ≈ 0.
  Además: `cart` es **leakage** (comprado ⟹ carrito), el precio tiene un efecto **no lineal** (U
  invertida dentro del rango del filtro), y varios campos son redundantes o ruido.
- **Dos familias de modelos** (§2.3.1): *catálogo* (usa el estado del listing: responde qué
  conviene promocionar hoy) e *intrínseco* (sin estado, `--strip-status`: responde qué esperar de
  un producto nuevo; techo mucho más bajo — PR-AUC ≈ 0.16 vs 0.76 — y eso es un hallazgo, no un bug).
- **Split:** por `query_id` (group split) 70/15/15, promediando varias corridas con seeds distintas.
- **Métricas:** ROC-AUC y PR-AUC (desbalance 13% de positivos; PR-AUC de azar ≈ 0.13), más curvas
  de loss train/val para over/underfitting. Sin umbral (no lo pide el enunciado).
- **Bandas de referencia ya medidas** (ver §7): regresión logística lineal ROC 0.959 / PR 0.660;
  un modelo no lineal con interacciones llega a ROC 0.968 / PR 0.762 (mismo split por query que los
  modelos; reproducible con `eda/verificaciones.py`). El transformer debería quedar en esa banda
  alta; si no supera al lineal, la capa de atención no está aportando.

---

## 1. El problema

### 1.1 Qué es el BTR y qué nos piden

El Buy Through Rate se define como `compras / impresiones` en la página de resultados de búsqueda:
la probabilidad de que un producto **mostrado** sea comprado. El objetivo de negocio es identificar
los mejores productos para promocionarlos en otras áreas del e-commerce.

El dataset (`supermarket_products.csv`, 10.000 filas) es un log de eventos: cada fila es **un
producto impreso en una búsqueda** (`query_id`), con los filtros que usó el usuario y el resultado
de la interacción (`cart`, `bought`).

### 1.2 Qué se predice exactamente (variable objetivo)

Dos lecturas posibles, que conviene explicitar en la presentación:

1. **Nivel evento (la que adoptamos):** clasificación binaria por impresión. Target: `bought`
   (bool). El modelo emite `p(bought | producto, contexto de búsqueda)`. Es la lectura que habilita
   las métricas sugeridas (ROC-AUC / PR-AUC son métricas de clasificación binaria por fila).
2. **Nivel producto (la métrica de negocio):** `BTR(producto) = E[p(bought)]` promediado sobre las
   impresiones del producto. Se deriva de (1) agregando. Para "elegir qué promocionar" se rankea
   por esta agregación.

No hay que definir umbral: las dos métricas sugeridas son threshold-free y el uso de negocio es un
ranking de productos, no una decisión binaria.

### 1.3 Qué NO es el problema

- No es generación de texto: no hay autoregresión, no hay "siguiente token". Esto tiene
  consecuencias directas sobre qué partes de la demo se conservan y cuáles no (§5).
- No es (todavía) personalización: no hay `user_id` en el dataset. Eso es el Ejercicio 3 (§10).

---

## 2. Qué encontramos en el dataset (EDA inicial)

Números del primer pase exploratorio (a reproducir prolijo en el notebook de EDA del Ejercicio 1):

### 2.1 Estructura de eventos

| Aspecto | Valor |
|---|---|
| Filas (impresiones) | 10.000 |
| Búsquedas (`query_id`) | 2.012 |
| Productos por búsqueda | 1–8, mediana 5 |
| `bought` | **13.0%** positivos |
| `cart` | 30.1% positivos |
| Compras por búsqueda | 0 en 53% de las queries; hasta 4 |

- Los filtros (`filter_category`, `filter_price_min/max`, `filter_storage_type`) son **constantes
  dentro de cada query** (verificado): son contexto de la búsqueda, no del producto.
- **Todos** los productos impresos cumplen su filtro: `category == filter_category`,
  `storage_type == filter_storage_type` y `price ∈ [min, max]` en el 100% de las filas. Implicancia:
  esos campos de filtro no agregan información *por sí solos* a nivel fila; lo que sí informa es la
  **posición relativa del precio dentro del rango filtrado** (ver 2.4).

### 2.2 `cart` es parte del target, no un feature (leakage)

`bought=True ⟹ cart=True` en el 100% de los casos (0 filas compradas sin carrito). Es el funnel
impresión → carrito → compra. Usar `cart` como entrada del modelo sería **data leakage**: (a) no está
disponible al momento de decidir qué promocionar, (b) `p(bought|cart=False) = 0` exactamente, el
modelo degeneraría a mirar solo eso.

Decisión: **excluir `cart` como feature**. Alternativas interesantes para discutir (no bloqueantes):
usarlo como **target auxiliar** (multi-task: predecir `cart` y `bought` a la vez, el gradiente extra
puede regularizar) o modelar el funnel en dos etapas `p(buy) = p(cart) · p(buy|cart)`. Lo anotamos
como extensión opcional.

### 2.3 La señal dominante está escondida en el texto 🚨

El título termina en un sufijo entre paréntesis en el 94.9% de las filas (19 valores distintos +
"sin sufijo"), y la última oración de la descripción dice lo mismo con otras palabras (mapeo
casi 1-a-1 verificado por crosstab). Ese "estado del listing" define **tiers de BTR brutales**:

| Tier (sufijo del título) | BTR | cart rate |
|---|---|---|
| `#1 Pick`, `Top Rated`, `Best Seller`, `Customer Favorite` | **0.63–0.68** | 0.71–0.74 |
| `Popular Choice`, `Highly Rated`, `Shopper Favorite`, `Well Reviewed` | 0.02–0.04 | 0.20–0.25 |
| Los 11 restantes (`New Listing`, `Clearance`, `Discontinuing Soon`, …) y sin sufijo | **0.000 exacto** | 0.18–0.24 |

Observaciones:
- No es "sentimiento": `Highly Rated` y `Shopper Favorite` suenan tan positivos como `Top Rated`,
  pero tienen BTR ~50 veces menor. Hay que aprender la partición exacta, no la valencia.
- El grupo de BTR=0 igual tiene carritos (~20%): agregan al carrito pero nunca compran.
- Consecuencia de diseño: si no procesamos el texto de alguna forma, el modelo queda ciego a casi
  toda la señal. Lo verificamos con un baseline (§7): sin este feature, ROC-AUC ≈ 0.55 (casi azar).

**Decisión propuesta:** extraer el sufijo como feature categórico `listing_status` (20 niveles) en
el preprocesamiento, y tratar título/descripción crudos como *alternativa* a comparar (§4, opción C).
Pero atención: usar o no usar esta información **es una decisión con contenido conceptual**, no un
detalle técnico — ver 2.3.1, que es probablemente la discusión más importante del TP.

### 2.3.1 ¿Es válido usar el estado del listing? (leakage vs circularidad)

Objeción planteada en el equipo: *"esas etiquetas se las pusieron a los productos DESPUÉS de que
se vendieran mucho; cuando lancemos un producto nuevo no vamos a saber si es Best Seller. ¿No
habría que sacar esa información?"* La objeción apunta a algo real y merece una respuesta precisa.
Conviene separar los features en tres clases:

1. **Outcome del mismo evento** (`cart`): ocurre *después* de la impresión que estamos prediciendo
   y es (parte de) el desenlace. **Leakage estricto** → nunca entra al modelo. Sobre esto no hay
   discusión.
2. **Estado del producto al momento de la impresión** (`listing_status`, las frases de la
   descripción): esto NO es leakage en sentido estricto — el badge ya está puesto y es visible
   cuando el usuario ve la página, o sea que está **disponible al momento de predecir**. Más aún:
   es plausible que sea *causa* de la compra (prueba social: los tiers altos convierten carrito →
   compra a tasas altísimas mientras los bajos cargan el carrito ~20% y compran 0%). **Pero** el
   badge es a su vez *consecuencia de outcomes pasados*: un modelo que lo usa para decidir qué
   promocionar es parcialmente **circular** ("promociono lo que ya era popular"), realimenta la
   popularidad (feedback loop) y es **ciego al cold-start**: todo producto nuevo cae en `New
   Listing`/`Recently Added` → BTR predicho ≈ 0 → nunca se promociona → profecía autocumplida.
3. **Atributos intrínsecos** (categoría, precio, alérgenos, peso, …): válidos para predecir y
   para decidir, sin circularidad.

Los números que dimensionan el dilema (GBM no lineal, split por query, seed 42):

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
| Con estado (`listing_status` + resto) | 0.968 | 0.762 |
| **Sin estado (solo atributos intrínsecos)** | **0.556** | **0.162** |

Además, el 61% de las filas está en tiers con BTR = 0.000 exacto. Es decir: en este dataset la
compra está gobernada **casi por completo** por la señal de estado/popularidad; los atributos
intrínsecos apenas superan el azar (0.162 vs 0.134). Sacar la señal de estado por completo dejaría
un problema casi impredecible — y la cátedra claramente la puso en el texto para que el EDA la
encuentre y el modelo la use.

**Resolución adoptada: entrenar y presentar DOS familias de modelos, porque responden preguntas
de negocio distintas.**

- **Modelo de catálogo (estado completo)**: usa `listing_status` / texto completo. Responde la
  pregunta literal del enunciado: *¿cuál es el BTR de cada producto del catálogo tal como se
  muestra hoy?* — para promocionar lo que hoy convierte.
- **Modelo intrínseco ("producto nuevo")**: sin `listing_status` (`--drop-features`) y con el
  texto limpio de estado (`--strip-status` saca el sufijo del título y la oración de estado de la
  descripción — necesario: si solo sacáramos el feature parseado, la variante de texto lo
  re-aprendería de la descripción). Responde: *¿qué BTR puede esperar un producto por sus
  atributos, antes de tener historia?* Su techo es mucho más bajo, y eso mismo es un hallazgo
  presentable: en este e-commerce la compra la mueve la prueba social, no el atributo del producto.

Limitación a declarar: con los timestamps comprometidos (2.6) no podemos reconstruir el estado
histórico del badge ni validar temporalmente esta distinción; en un sistema real el estado del
listing en el momento exacto de cada impresión vendría del log. Conecta con el position bias (§9)
y con la personalización (Ej. 3): son todas caras del mismo problema — predecir ≠ decidir.

### 2.4 Efectos secundarios (condicionado al tier alto)

Dentro del tier alto (donde hay varianza), hay estructura adicional — esto es lo que un modelo
lineal no captura y donde el transformer tiene margen:

- **Precio, U invertida:** `p(bought | top)` según la posición del precio dentro del rango del
  filtro: 0.39 (piso del rango) → **0.86 (centro)** → 0.43 (techo). Efecto fuerte y **no
  monotónico** → motiva codificaciones no lineales de numéricas (§6.2).
- **Alérgenos:** Fish/Shellfish/Tree nuts/Peanuts ≈ 0.29–0.39 vs None/Milk/Soy/Wheat ≈ 0.69–0.72.
- **Categoría:** Seafood 0.30 … Bakery 0.74 (y Seafood ya era la peor incondicional).
- **Marca y país:** efectos suaves (~0.59–0.72). **Nutrition score:** casi plano (0.68 → 0.63).
- Los mismos patrones se repiten (a escala chica) en el tier medio: el proceso generador parece
  `tier(texto) × categoría × alérgenos × campana(precio relativo)`.

### 2.5 Interacción entre productos de la misma página: débil

Hipótesis natural: "que te compren depende de contra quién aparecés". Medido:
`p(bought | soy tier alto)` según cuántos tier-alto hay en mi query: 0.64, 0.65, 0.67, 0.63 para
1–4 competidores. **Casi constante** → en este dataset se pueden comprar varios productos de la
misma página (changuito de supermercado, no click exclusivo) y la competencia no penaliza.
Esto baja la prioridad de la formulación listwise (§4, opción B), pero la mantenemos como
experimento porque: (a) es la manera de *demostrar* esa conclusión, (b) ejercita máscaras de
padding, que es contenido central de la materia.

### 2.6 Calidad de datos y redundancias (para el informe de EDA)

- **Timestamps sospechosos:** dentro de una misma `query_id` los timestamps abarcan hasta 2 años
  (mediana ~16 meses). Una "búsqueda" no puede durar eso: o `query_id` agrupa un template de
  búsqueda recurrente, o el timestamp es ruido sintético. Además BTR por año/día de semana/hora ≈
  constante. Decisión: **no usar timestamp como feature** en v1 (probar features cíclicos como
  ablación de descarte) y reportarlo como hallazgo de calidad de datos.
- **Redundancias verificadas:** `net_weight_oz` ≈ número de `package_size` (corr 0.995, ratio
  mediano 1.00) → conservamos `net_weight_oz` + `unit_of_measure`. `filter_category` y
  `filter_storage_type` duplican atributos del producto (2.1). La descripción duplica el sufijo del
  título (2.3) y el resto de la descripción es template con los mismos atributos estructurados.
- **Sin señal detectada:** volumen físico (`dimensions_in`), cantidad de ingredientes,
  `nutrition_score` (muy débil). Candidatos a excluir de v1 y reincorporar solo si una ablación lo
  justifica.
- **Nulos:** solo `allergens` (44.6%) → categoría explícita `None` (semánticamente es "sin
  alérgenos declarados", no dato faltante clásico).
- 88 títulos aparecen en más de una query (posible fuga suave entre splits; el group split por
  query lo mitiga; opcional: agrupar por título también).
- Desbalance 87/13 → pesa la elección PR-AUC y sugiere probar `pos_weight` en la loss.

---

## 3. ¿Dónde y por qué va un transformer acá?

Pregunta explícita de la cátedra ("¿dónde, cómo y por qué?"). Nuestra respuesta propuesta:

**El transformer es el modelo que estima `p(bought)`**, y el mecanismo de atención es la pieza que
aprende **interacciones entre features** sin ingeniería manual. El EDA muestra que la señal es
composicional: el efecto del precio depende del tier, el del alérgeno depende del tier, etc. Un
modelo lineal sobre one-hot no puede representarlo (verificado: PR 0.72 vs 0.81 del no lineal);
un MLP lo puede aproximar mezclando todo; la self-attention lo hace **explícito e inspeccionable**
(los mapas de atención muestran qué feature "mira" a cuál — material de oro para la presentación).

Justificación teórica del recorte de arquitectura (conexión con las clases 1 y 2):

- **Encoder-only, sin máscara causal.** La máscara triangular del decoder existe para no ver el
  futuro al generar autoregresivamente. Acá no hay noción de futuro: los features de un producto
  son un **conjunto**, no una secuencia temporal. Corresponde atención bidireccional (como BERT).
- **Token `[CLS]` + cabeza de clasificación.** Igual que BERT para clasificación: un token
  aprendido que atiende a todos los features y cuyo embedding final se proyecta a 1 logit.
  Alternativa a comparar: mean-pooling sobre los tokens de salida.
- **Sin positional encoding (por defecto).** La self-attention es invariante a permutaciones; el
  orden de los features es arbitrario y cada feature ya recibe una identidad propia (embedding por
  columna). Agregar PE acá no debería aportar — **lo verificamos como ablación** en lugar de
  asumirlo (es una de las ablaciones que sugirió la cátedra).
- **Sin generación autoregresiva ni softmax sobre vocabulario.** La salida es 1 logit → sigmoide →
  `p(bought)`, con binary cross-entropy. (En la demo: `vocab_size` logits por posición + softmax +
  cross-entropy contra el próximo token.)

## 4. La pregunta conceptual central: ¿qué son los "tokens"?

En texto el token es la unidad discreta de la secuencia. Acá hay tres candidatos, y elegir es LA
decisión de diseño del TP:

### Opción A — cada feature es un token (elegida como principal)

Secuencia de entrada = `[CLS, listing_status, category, brand, storage, unit, country, allergens,
price, price_rel, filter_min, filter_max, net_weight, nutrition]` → largo fijo 14.
Cada categórica pasa por su `nn.Embedding(cardinalidad, d_model)`; cada numérica se proyecta a
`d_model` (§6.2). La atención relaciona features entre sí (interacciones de orden alto).

- ✅ Es exactamente la estructura de FT-Transformer/AutoInt (estado del arte tabular publicado).
- ✅ Reusa los bloques de la demo casi sin tocar; secuencia corta → barato de entrenar.
- ✅ La atención es interpretable a nivel feature (presentación).
- ⚠️ "Token" deja de ser una unidad discreta de un vocabulario compartido; hay que explicarlo bien:
  la analogía correcta es *token = unidad de información que emite/recibe atención*.

### Opción B — cada producto de la query es un token (variante a comparar)

Secuencia = los 1–8 productos de la misma página de resultados; cada producto se colapsa primero a
un vector `d_model` (con la misma tokenización de features de A, agregada), y la atención modela la
página como conjunto (¿me compran a mí dado lo que aparece al lado?). Salida: 1 logit **por token**
(por producto), como los logits por posición de la demo pero con sigmoide.

- ✅ Usa la estructura por `query_id` del dataset; análoga a los re-rankers listwise (PRM/SetRank).
- ✅ Obliga a implementar **padding + máscara de padding** (contenido central de la materia).
- ⚠️ El EDA (2.5) anticipa efecto débil: probablemente empate o pierda contra A. Ese resultado
  "esperado y explicado" es en sí presentable. Implementada (`--arch listwise`); primera corrida
  (seed 42): PR-AUC 0.664 vs 0.766 de A — consistente con la predicción.

### Opción C — el texto crudo como tokens (implementada como contrapunto de A)

Pasar `title + '\n' + description` por el transformer **a nivel caracteres, exactamente como la
demo**: vocabulario de chars de train, PAD/UNK, secuencia truncada a 256 (el título — que contiene
el sufijo al final — entra siempre entero: mide ≤ 81 chars; p95 de título+descripción = 243).
Acá, a diferencia de A, el **positional encoding es necesario** (el texto sí tiene orden) y la
**máscara de padding** también (longitudes variables) — dos piezas de la materia que A no ejercita.

- ✅ Es la formulación más fiel al pipeline de la materia (clase 2: texto → tokens → embeddings →
  positional encoding → transformer) y responde la pregunta más linda del TP: si NO le damos el
  `listing_status` parseado, ¿el modelo lo redescubre desde los caracteres crudos?
- ✅ "No enroscarse con el tokenizador" se respeta: caracteres, como la demo — cero ingeniería.
- ⚠️ Costo real: secuencia ~257 vs 14 → atención cuadrática ≈ 300× más cómputo. Medido: ~74
  s/época en CPU (~40–90 min por corrida) vs 28 s la corrida entera de A. **Acá se usa la GPU**
  (ver §7.5). Con ~7k filas de train quizás no alcance a aprender la partición exacta de 19
  sufijos: cualquiera de los dos desenlaces es un resultado presentable (capacidad del transformer
  vs valor del feature engineering guiado por EDA).

También queda el modo **híbrido**: `[CLS] + feature-tokens + caracteres` en una misma secuencia
(la atención puede cruzar texto ↔ features). La comparación clave del informe:
`features` (con `listing_status`) vs `text` vs `hybrid --drop-features listing_status`
(reemplazar el regex por la fuente cruda).

**Variante C2 — transformer solo como encoder de texto (`--arch tower`).** El transformer no tiene
por qué ser el clasificador entero: acá la torre de texto ([CLS] + chars → bloques → embedding)
comprime el texto a UN vector, que se concatena con los feature-tokens tabulares y clasifica un
MLP. Es el transformer usado como *módulo de embedding* (el encoder-only de BERT de la clase 2)
en lugar de como clasificador. La diferencia con `hybrid` es exactamente que acá la atención **no
puede cruzar** texto ↔ features — comparar ambos mide cuánto vale esa atención cruzada. También
implementada (`--arch mlp` es el otro extremo: nada de atención).

### Opción D — embeddings query↔producto estilo skipgram (analizada; se recicla en Ej. 3)

Idea surgida en el equipo: aprender embeddings de queries y productos al estilo del skipgram de la
clase de embeddings — pares positivos (query, producto comprado) empujan el producto interno hacia
arriba, negativos muestreados al azar lo empujan hacia abajo. Es una técnica real y muy usada: es
**item2vec / two-tower retrieval** (los embeddings de listings de Airbnb se entrenan exactamente
así sobre sesiones de clicks). Análisis para *este* problema:

- **Nuestra BCE ya optimiza ese mismo objetivo, con mejores negativos.** La loss de skipgram con
  negative sampling es `log σ(u·v)` para positivos + `log σ(−u·v_neg)` para negativos — la misma
  familia logística que nuestra binary cross-entropy. La diferencia: skipgram muestrea negativos
  *al azar* porque no tiene log de impresiones; nosotros tenemos los negativos **reales** (productos
  impresos en esa query y NO comprados) — negativos "difíciles" (alternativas reales que el usuario
  vio y descartó), estrictamente más informativos que un producto random que los filtros ya
  excluían. Reemplazar nuestros negativos por muestreo sería cambiar datos mejores por peores.
- **El producto interno u·v es una restricción de escalabilidad, no una virtud.** El two-tower
  factoriza el score en dot(query_emb, product_emb) para poder precomputar embeddings y buscar
  entre millones de ítems (etapa de *retrieval*). Nuestro problema es la etapa de *ranking*:
  puntuar con precisión los ≤8 candidatos ya impresos — ahí un modelo conjunto f(query, producto)
  como el transformer es estrictamente más expresivo (captura interacciones no bilineales, p. ej.
  la U del precio × tier).
- **Un embedding por `query_id` no tiene sentido acá**: cada id aparece ~5 veces (overfitting
  garantizado) y toda búsqueda nueva trae un id nuevo (cold-start en el 100% de la inferencia).
  El *contenido* de la query son solo sus filtros, que ya entran al modelo — `price_rel` ES el
  feature query-condicional, y es de los más fuertes (2.4). Con timestamps rotos y sin usuarios,
  no queda información residual de query para minar (la composición de la página ya la mide la
  formulación B, con efecto débil).
- **item2vec como pretraining de productos**: pares por co-compra dentro de la query — solo 284
  queries tienen ≥2 compras, muy poco; pares por co-impresión codificarían "compartimos filtros"
  (categoría/precio), que ya está explícito en los features. Valor marginal acá; descartado con
  esta justificación.

**Dónde SÍ brilla la idea: Ejercicio 3.** Cambiá "query" por "usuario" y es la respuesta clásica a
personalización: embeddings de usuario y de producto en un espacio compartido, entrenados con pares
(usuario, producto comprado) positivos y negativos muestreados — el mismo truco de skipgram de la
clase 2, a escala industrial (retrieval de YouTube/Airbnb). Queda incorporada en §10.

**Plan:** A como modelo de referencia rápido, **A vs C (y el híbrido sin `listing_status`) como
experimento central**, B (listwise) como tercer experimento.

## 5. Arquitectura propuesta (v1) y mapeo con la demo

```
entrada (por impresión)
  7 categóricas ──► nn.Embedding por feature ─┐
  6 numéricas   ──► proyección a d_model      ├─► secuencia (14 × d_model), con [CLS]
  [CLS] aprendido ────────────────────────────┘
        │
        ▼
  N × Block:   x = x + MultiHeadAttention(LayerNorm(x))   ← SIN máscara causal
               x = x + FeedForward(LayerNorm(x))          ← pre-LN + residuales, como la demo
        │
        ▼
  LayerNorm final → tomar salida del [CLS] → Linear(d_model → 1) → sigmoide = p(bought)
```

Hiperparámetros iniciales (siguiendo la sugerencia de arrancar chico, `d_model < 100`):
`d_model=32, n_heads=4, n_layers=2, dropout=0.1, batch=256, AdamW lr=1e-3`, BCEWithLogits
(con `pos_weight` como experimento). Con esto el modelo tiene ~30k parámetros: corre en CPU.

Las tres formulaciones comparten estos mismos bloques y se eligen con
`--formulation {features, text, hybrid}`: cambia solo la construcción de la secuencia
(features: 14 tokens sin PE; text: `[CLS]` + ~256 chars con PE aprendido y máscara de padding;
hybrid: `[CLS]` + features + chars). `--drop-features listing_status` permite el experimento de
reemplazo regex → texto crudo.

Mapeo demo → nuestro código (trazabilidad para poder defender cada línea):

| Demo (notebook) | Nuestro código | Cambio y por qué |
|---|---|---|
| `Head` | `btr/model.py::Head` | Igual, pero (1) la máscara causal pasa a ser **opcional** (`causal=False` por defecto, §3) y (2) se corrige el **doble escalado**: la demo multiplica por `k.shape[-1]**-0.5` y además divide por `head_size**0.5` — es decir divide dos veces por √d_k; el paper escala una sola vez. Lo dejamos anotado para mencionarlo (¡ablación graciosa posible: doble escala vs simple!). |
| `MultiHeadAttention` | idem | Igual (concat de heads + proyección + dropout). |
| `FeedFoward` (sic) | `FeedForward` | Igual (expansión ×4 + ReLU + proyección + dropout). |
| `Block` | idem | Igual (pre-LN + residuales). |
| `token_embedding_table` (vocab de 65 chars) | `FeatureTokenizer` | En vez de UN vocabulario compartido, un embedding POR feature categórica y una proyección POR feature numérica. Conceptualmente igual: lookup/afín que lleva la unidad de entrada a `d_model`. |
| `position_embedding_table` | opcional / ausente | Sin orden que codificar (§3); queda como flag para la ablación. |
| `lm_head` (d_model → vocab) + softmax + CE | `cls_head` (d_model → 1) + sigmoide + BCE | Clasificación binaria, no distribución sobre vocabulario. |
| `generate()` autoregresivo | — (se elimina) | No hay generación en este problema. |
| `get_batch` (ventanas del corpus) | `DataLoader` tabular | Muestreo de filas (A) o de queries con padding (B). |

### 5.1 De punta a punta con un ejemplo real

Primera fila del CSV: *"Cedar House Steamable Pepperoni Pizza - 10 oz (Well Reviewed)"*,
categoría Frozen, $8.30, búsqueda con filtro Frozen y rango $2.45–$15.45, `bought=false`.

1. **Qué entra:** no el texto crudo sino los 13 features ya preprocesados:
   7 categóricos (`listing_status='Well Reviewed'`, `category='Frozen'`, `brand='Cedar House'`,
   `storage='Frozen'`, `unit='oz'`, `country='United States'`, `allergens='Wheat'`) y
   6 numéricos estandarizados (`price=8.30`, `price_rel=(8.30−2.45)/(15.45−2.45)=0.45`,
   `filter_min=2.45`, `filter_max=15.45`, `net_weight=10.14`, `nutrition=36`).
2. **Tokenización:** cada feature se vuelve un vector de `d_model=32`: los categóricos por
   lookup en su tabla propia (p. ej. la fila "Well Reviewed" de la tabla de `listing_status`,
   21×32) y los numéricos por proyección afín propia (`0.45·w_i + b_i`). Se antepone el vector
   aprendido `[CLS]` → matriz de **14×32**. Todo aprendido por backprop, como los embeddings de
   caracteres de la demo.
3. **Bloques (×2):** self-attention de 4 cabezas + MLP, con residuales y pre-LN. Acá cada
   feature "mira" a los demás: el token de `price_rel` puede atender al de `listing_status`
   ("estoy en el centro del rango **y** soy tier alto") — la interacción que un modelo lineal
   no puede representar. Sin máscara causal y sin positional encoding (§3).
4. **Qué sale:** el vector final de `[CLS]` (32 números) → `Linear(32→1)` → un logit →
   sigmoide → **`p(bought) ≈ 0.10`** (corrida seed 42; coherente: "Well Reviewed" es tier
   medio, BTR real del grupo ≈ 0.04, y esta fila no fue comprada).
5. **Entrenamiento:** BCE entre esa probabilidad y el `bought` real de cada fila ajusta todo
   (tablas, proyecciones, W_Q/W_K/W_V, MLPs, `[CLS]`, cabeza). En inferencia el mismo forward
   devuelve el BTR estimado para cualquier producto descripto por sus features — esté o no en
   el catálogo actual.

## 6. Codificación de features: opciones investigadas y decisión

(Ejercicio 1.4; la sugerencia del enunciado era "investigar técnicas de codificación, p. ej.
one-hot". Resumen de lo investigado y qué usamos para cada tipo.)

### 6.1 Categóricas

| Técnica | Idea | Pros / contras para este TP |
|---|---|---|
| **One-hot** | vector binario de dimensión = cardinalidad | Simple, sin supuestos de orden; explota con cardinalidad alta; no captura similitud entre niveles. Perfecto para los **baselines** (logística / MLP). |
| **Embedding aprendido** (entity embeddings) | lookup `nn.Embedding` entrenado con el modelo | Denso, aprende similitud entre niveles (p. ej. los 4 sufijos "top" deberían terminar cerca); es lo que usa la demo para caracteres y FT-Transformer para columnas. **Elegida para el transformer.** |
| Target encoding | reemplazar nivel por media (suavizada) del target | Compacto, pero riesgo alto de leakage/overfit en 10k filas; requiere esquema out-of-fold. Lo descartamos con esta justificación (es el tipo de alternativa que está bueno discutir). |
| Ordinal / label | entero arbitrario | Impone un orden falso a nominales. Descartada. |
| Hashing / frequency | para cardinalidad enorme | Innecesario: nuestra máxima cardinalidad es 20. |

Punto fino para la presentación: **one-hot seguido de una capa lineal ES un embedding aprendido**
(`W · onehot(i) = W[:, i]` = fila de la tabla). O sea que la elección real no es "one-hot vs
embedding" sino *dónde* está la no linealidad y si la tabla se comparte. Con eso respondemos
"¿por qué no one-hot?" sin descartarlo: lo usamos, factorizado adentro del `nn.Embedding`.

Cardinalidades: `listing_status` 20, `category` 12, `brand` 15, `storage_type` 3,
`unit_of_measure` 5, `country_of_origin` 10, `allergens` 8 (con `None`).

### 6.2 Numéricas

`price`, `price_rel = (price - filter_min)/(filter_max - filter_min)` (feature derivado nuevo,
motivado por 2.4), `filter_price_min`, `filter_price_max`, `net_weight_oz`, `nutrition_score`.

1. **Estandarización** (z-score con estadísticos de train; log1p previo para las sesgadas como
   `price` y `net_weight_oz`) — siempre, para estabilidad numérica.
2. **¿Cómo se vuelve un token?** Opciones (Gorishniy et al. 2022):
   - **Proyección lineal por feature**: `token_i = x_i · w_i + b_i` con `w_i, b_i ∈ R^d_model`
     (FT-Transformer básico). Default v1.
   - **Binning por cuantiles + embedding** (piecewise): discretizar en Q bins y hacer lookup como
     categórica. Captura no monotonías (¡la U invertida del precio!) sin depender del MLP.
   - Periodic/Fourier embeddings: senos y cosenos de frecuencias aprendidas (une esto con la
     motivación del positional encoding sinusoidal visto en clase — mismo truco, otro uso).
   - **Ablación planeada:** lineal vs bins (vs periodic si hay tiempo). Es de las ablaciones más
     interesantes porque el EDA predice que bins debería ganar en `price_rel`.

### 6.3 Texto, tiempo y descartes

- `title`/`description` → `listing_status` (parseo por regex, §2.3). Alternativas documentadas en §4C.
- `timestamp` → fuera de v1 (2.6); ablación de descarte con hora/día/mes cíclicos (sin/cos).
- `cart` → excluido (leakage, §2.2). `query_id` → no es feature (identificador; úsase para split y
  para la formulación B). `package_size`, `filter_category`, `filter_storage_type`,
  `dimensions_in`, `ingredients` → redundantes o sin señal (2.6); se documenta y se pueden
  reintroducir vía ablación si sobra tiempo.

## 7. Protocolo experimental

### 7.1 Partición

**GroupShuffleSplit por `query_id`**: 70% train / 15% val / 15% test. Motivo: filas de la misma
query comparten contexto (filtros idénticos, outcome correlacionado); partir por fila contaminaría
val/test con queries vistas (y rompería la formulación B). El split temporal —lo canónico en
sistemas reales— queda descartado porque el timestamp está comprometido (2.6); lo discutimos como
limitación. Siguiendo la recomendación de la cátedra: **promedio de ≥3 corridas** (seed de split y
de inicialización) con media ± desvío, antes que cross-validation compleja.

**¿Y partir por producto en vez de por query?** Inquietud razonable: si el mismo producto cae
en train y en test, el ejemplo de test sería "regalado". Medido en este dataset: 9.910 títulos
únicos en 10.000 filas; con el split por query solo el **1.0% de las filas de test** tiene un
título ya visto en train (y 18.8% comparte el "producto base", título sin el sufijo de estado).
Además el modelo **no recibe identidad del producto** (ni id ni título): solo features, así que
no tiene canal para memorizar un producto puntual. Verificación empírica (seed 42): métricas de
test idénticas restringiendo a títulos nunca vistos (PR-AUC 0.766) o a productos base nunca
vistos (PR-AUC 0.770) vs test completo (0.766). Conclusión: el split por query ya es
~producto-disjunto en la práctica; dejamos este chequeo como control de robustez del informe.
La línea entre leakage y señal: leakage es información no disponible al predecir (`cart`, el
outcome de la misma página); que los *patrones de features* se repitan entre train y test no es
fuga — es exactamente la regularidad que el modelo debe aprender.

### 7.2 Métricas y control de ajuste

- **PR-AUC (average precision)**: métrica principal (desbalance 13%; azar ≈ 0.13).
- **ROC-AUC**: complementaria (azar = 0.5).
- Curvas de loss train/val por época → diagnóstico over/underfitting; early stopping por PR-AUC val.
- Extra barato si hay tiempo: curva de calibración (el BTR es una probabilidad; si el modelo está
  bien calibrado, el "promedio de p" por producto es directamente su BTR estimado).

### 7.3 Baselines (escalera de complejidad)

Medidos con **el mismo split por query (seed 42, test 15%) que usan los modelos** —
reproducibles con `.venv/bin/python eda/verificaciones.py`:

| Modelo | ROC-AUC | PR-AUC | Lectura |
|---|---|---|---|
| Azar / prevalencia | 0.50 | 0.13 | piso |
| Regresión logística SOLO `listing_status` | 0.953 | 0.644 | la señal de texto sola |
| Regresión logística todo (one-hot + z-score) | 0.959 | 0.660 | techo **lineal** |
| Regresión logística SIN `listing_status` | 0.535 | 0.142 | sin texto no hay problema que resolver |
| Gradient boosting (referencia no lineal) | 0.968 | **0.762** | techo aproximado con interacciones |
| GBM **sin estado** (mundo "producto nuevo") | 0.556 | 0.162 | techo intrínseco (§2.3.1) |
| MLP baseline (mismos embeddings, sin atención) | 0.964 | 0.715 | primera corrida, seed 42 |

Lecturas importantes: (1) el gap 0.66 → 0.76 de PR-AUC es **el margen que justifica un modelo con
interacciones**: ahí tiene que vivir el transformer (primera corrida seed 42: 0.766 ✓);
(2) la comparación honesta transformer vs MLP con los mismos inputs es parte del informe;
(3) si alguien del equipo obtiene 0.99+, oler leakage antes que festejar.

### 7.4 Plan de ablaciones (en orden de valor)

Toda la suite está codificada en `experimentos.py` (24 configuraciones × N seeds, con
`--resumen` para la tabla comparativa); los nombres de abajo referencian esos experimentos.

1. **Formulaciones: `features` vs `text` vs `hybrid` sin `listing_status`** — ¿el transformer
   redescubre desde los caracteres la señal que parseamos a mano? El experimento estrella (§4C).
   Corre en GPU.
2. **Transformer vs MLP vs logística** (mismos features): ¿la atención aporta sobre mezclar todo?
3. **Codificación numérica**: proyección lineal vs binning (§6.2) — predicción: bins gana por la U.
4. **Sin `listing_status`** (en formulación A): cuánta señal aporta el resto de los features
   (feature importance por ablación; conecta con el hallazgo del EDA).
5. **Capacidad**: `d_model ∈ {8, 16, 32, 64}`, `n_layers ∈ {1, 2, 4}`, `n_heads ∈ {1, 2, 4}` —
   grilla chica estilo tabla del paper, con promedios de corridas (en A y en C).
6. **Positional encoding on/off**: en A la teoría predice indistinto (§3); en C debería ser
   imprescindible — el contraste entre ambos es teoría hecha experimento. Y **[CLS] vs mean-pooling**.
7. **Máscara causal on/off**: ¿cuánto cuesta ponerle al problema una restricción que no corresponde?
   (bonito para conectar encoder vs decoder en la presentación).
8. **Formulación B (listwise)** vs A: ¿el contexto de la página aporta? (predicción: poco, 2.5).
9. Menores: `pos_weight` en la loss, dropout ∈ {0, 0.1, 0.2}, ReLU vs GELU, doble escalado de la
   demo vs escalado simple.
+ **Visualización de mapas de atención** del modelo final (no es ablación pero es el gráfico
  estrella de la presentación: ¿el CLS mira al `listing_status`? ¿`price_rel` interactúa con el tier?).

### 7.5 Presupuesto de cómputo (medido)

| Corrida | Secuencia | CPU (medido) | Dónde correr |
|---|---|---|---|
| A (`features`), 1 corrida completa | 14 | ~30 s | CPU, donde sea |
| A, grilla de capacidad × 3 seeds (~60 corridas) | 14 | ~30 min | CPU ok |
| C (`text`), 1 corrida completa | 257 | ~74 s/época → 40–90 min | **GPU (RTX 3070)** |
| C/híbrido, grilla × seeds | 257–270 | días en CPU | **GPU**; plan B: Colab |

La lectura pedagógica: el cómputo del TP no está en el modelo tabular (10k filas, secuencia 14,
`d_model<100` como sugiere el enunciado) sino en (i) meter el **texto** por el transformer —
atención cuadrática en la longitud — y (ii) el **volumen de experimentación** estilo SIA
(grillas × seeds). Ahí es donde entran la advertencia de la cátedra sobre costo computacional y
la sugerencia de Colab.

## 8. Estructura de código propuesta

```
llm-tp1/
├── propuesta.md              ← este documento (diseño vivo)
├── bitacora.md               ← registro cronológico de discusiones y decisiones
├── eda/verificaciones.py     ← reproduce todos los números del EDA y los baselines
├── supermarket_products.csv
├── btr/
│   ├── __init__.py
│   ├── data.py               ← carga, parseo (listing_status, price_rel), vocabularios,
│   │                            split por query, tensores / batches (A y B)
│   ├── model.py              ← Head / MultiHeadAttention / FeedForward / Block (de la demo,
│   │                            máscara opcional) + FeatureTokenizer + BTRTransformer
│   └── train.py              ← loop de entrenamiento, early stopping, métricas, seeds,
│                                guardado de corridas y soporte GPU (--device)
├── experimentos.py           ← suite curada (24 configs × seeds) + tabla resumen
├── resultados/               ← un JSON por corrida: config, historial por época
│                                (loss/ROC/PR en train y val), métricas finales de val y test
├── pesos/                    ← checkpoints .pt (con --save-pesos): state_dict + config +
│                                preprocesador; se recargan con btr.model.load_checkpoint
├── notebooks/                ← EDA (Ej. 1) y experimentos/gráficos (Ej. 2)
└── material/                 ← NO se versiona (.gitignore)
```

## 9. Riesgos y limitaciones a declarar

- **El problema se vuelve "fácil" una vez parseado el texto**: el valor del TP no está en el número
  final sino en la justificación de cada módulo + ablaciones. Reportar la escalera de baselines
  evita vender humo.
- **10k filas vs apetito del transformer**: mantener el modelo chico, dropout, early stopping,
  promediar seeds. Si hay ruido entre corridas, subir a 5 seeds.
- **No hay posición/rank en la página de resultados**: el BTR real sufre position bias (los
  primeros resultados se compran más) y acá no podemos modelarlo ni corregirlo. Limitación honesta
  para mencionar.
- **Timestamps comprometidos** → sin validación temporal; en producción sería obligatoria.
- 88 títulos repetidos entre queries → cuantificado: 1% de test con título visto en train y métricas idénticas en el subconjunto no visto (§7.1); riesgo descartado empíricamente.
- El dataset es sintético con estructura generadora bastante determinística (tier=0 exacto):
  cuidado con leer métricas altas como "resolvimos retail".
- La variante de texto entrena desde cero con solo ~7k ejemplos: puede no alcanzar el techo de A.
  No es un fracaso — es la comparación honesta entre aprender la señal desde el crudo y
  dársela parseada (documentar cualquiera de los dos desenlaces).

## 10. Ejercicio 3 — personalización (borrador de la diapositiva)

Idea a desarrollar (teórico, sin implementar): hoy el modelo estima `p(bought | producto, query)`;
personalizar es condicionar también en **quién busca**: `p(bought | producto, query, usuario)`.
Cambios mínimos sobre nuestra arquitectura A:

1. **Datos**: loggear `user_id` por evento + historial de interacciones (impresiones, carts,
   compras con timestamp confiable).
2. **Modelo**: agregar tokens de usuario a la secuencia del transformer:
   - un **embedding de usuario** aprendido (para regulares), y/o
   - los **últimos K productos comprados/carteados** como tokens (cada uno tokenizado igual que el
     candidato) — la atención cruza candidato × historial (estilo Behavior Sequence Transformer:
     "compra comida de perro todos los meses" ⇒ el candidato "dog food" atiende a esas compras,
     con features de recencia/periodicidad).
3. **Cold-start**: usuario sin historial → cae al modelo actual (BTR global); alternativa de
   arquitectura: two-tower (torre usuario / torre producto) si hiciera falta servir a gran escala.
   Los embeddings de usuario/producto del two-tower se entrenan con el mismo esquema de pares
   positivos (usuario, producto comprado) y negativos muestreados que el skipgram de la clase de
   embeddings (§4D) — así conectamos el Ej. 3 directamente con la teoría vista.
4. **Riesgos**: privacidad, feedback loops (recomendar lo ya comprado refuerza), y evaluación
   (haría falta split por usuario y por tiempo).

## 11. Temas abiertos / para preguntar a la cátedra

- [ ] ¿Está bien leer "predecir BTR" como clasificación binaria por impresión + agregación por
  producto (§1.2)? (Las métricas sugeridas apuntan a que sí.)
- [ ] ¿Extraer `listing_status` del título vía regex cuenta como preprocesamiento válido del texto,
  o esperan que el texto entre "crudo" por un encoder? (Llevar el argumento §2.3/§4C.)
- [ ] ¿Excluir `cart` por leakage es la lectura esperada, o quieren verlo usado (p. ej. multi-task)?
- [ ] El estado del listing (Best Seller, etc.) deriva de popularidad pasada: ¿esperan que se use
  como feature (está disponible al predecir), que se excluya (circularidad para promocionar), o
  exactamente la doble lectura de §2.3.1? (Llevar los números: 0.76 vs 0.16 de PR-AUC.)
- [ ] Confirmar que el promedio de corridas con distintas seeds (sin CV) alcanza (dijeron que sí).
- [ ] ¿El timestamp intra-query de 2 años es intencional (trampa de EDA) o artefacto del generador?

## 12. Referencias

**De la materia:** [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) ·
[BERT (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805) · demo de la cátedra (decoder-only
a nivel caracteres, base de nuestros bloques).

**Codificación de variables:**
[Entity Embeddings of Categorical Variables (Guo & Berkhahn, 2016)](https://arxiv.org/abs/1604.06737) ·
[Survey on categorical data for neural networks (Hancock & Khoshgoftaar, 2020)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00305-w) ·
[On Embeddings for Numerical Features in Tabular DL (Gorishniy et al., 2022)](https://arxiv.org/abs/2203.05556)

**Transformers para tabular / CTR / ranking:**
[Revisiting Deep Learning Models for Tabular Data — FT-Transformer (Gorishniy et al., 2021)](https://arxiv.org/abs/2106.11959) ·
[TabTransformer (Huang et al., 2020)](https://arxiv.org/abs/2012.06678) ·
[AutoInt (Song et al., 2019)](https://arxiv.org/abs/1810.11921) ·
[Personalized Re-ranking for Recommendation (Pei et al., 2019)](https://arxiv.org/abs/1904.06813) ·
[SetRank (Pang et al., 2020)](https://arxiv.org/abs/1912.05891) ·
[Behavior Sequence Transformer (Chen et al., 2019)](https://arxiv.org/abs/1905.06874) — para Ej. 3.

**Métricas con desbalance:**
[The Precision-Recall Plot Is More Informative than ROC (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) ·
[contrapunto reciente (McDermott et al., 2024)](https://www.cell.com/patterns/fulltext/S2666-3899(24)00109-0)
