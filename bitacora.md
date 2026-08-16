# Bitácora del TP1

> Registro cronológico de lo que fuimos pensando, discutiendo y decidiendo, con su porqué.
> El documento de diseño *vivo* es [`propuesta.md`](propuesta.md) — acá queda la historia de cómo
> llegamos a cada decisión, para el informe ("desafíos encontrados") y para defender en la
> presentación. Cada entrada: **discusión → decisión → dónde quedó escrito/implementado**.

## 15/08 — Lectura del material y formulación del problema

**Discusión.** Lectura completa del enunciado, los transcripts de las clases (0, 1, 2a, 2b), las
ppts y la demo (decoder-only a nivel caracteres estilo Karpathy). Primera pregunta grande: ¿qué es
exactamente lo que se predice y qué sería un "token" en este problema?

**Decisiones.**
- Target: `bought` por impresión (cada fila = un producto impreso en una búsqueda); el BTR de
  negocio sale de agregar las probabilidades por producto. → `propuesta.md §1`
- Tres formulaciones candidatas de "token": (A) cada feature es un token (estilo FT-Transformer),
  (B) cada producto de la página es un token (listwise), (C) el texto crudo como tokens (la demo
  literal). A como principal. → `propuesta.md §4`
- Encoder-only sin máscara causal (clasificación de un conjunto, no generación), token `[CLS]` +
  1 logit (como BERT en la clase 2), sin positional encoding para A (los features no tienen
  orden — a verificar como ablación). → `propuesta.md §3, §5`

## 15/08 — EDA inicial: el dataset tiene trampa

**Hallazgos** (números reproducibles con `eda/verificaciones.py`):
- `bought=true ⟹ cart=true` en el 100% de las filas → `cart` es el funnel, no un feature (leakage).
- **La señal dominante está escondida en el texto**: el sufijo del título define tiers de BTR
  brutales (4 sufijos con ~0.65, 4 con ~0.03, 11 + sin-sufijo con 0.000 exacto). No es sentimiento:
  "Highly Rated" suena igual de bien que "Top Rated" y compra 50 veces menos.
- Precio: **U invertida** dentro del rango del filtro (0.39 → 0.86 → 0.43) → motiva encodings no
  lineales de numéricas.
- Competencia dentro de la página: débil (se compran varios productos por query).
- Timestamps rotos (hasta 2 años dentro de una misma query) → sin split temporal, feature descartado.
- Redundancias: filtros ≡ atributos del producto, `package_size` ≈ `net_weight_oz`, descripción ≡
  sufijo del título.

**Decisiones.** Parsear el sufijo como `listing_status`; derivar `price_rel`; split por `query_id`;
métricas PR-AUC/ROC-AUC con promedio de seeds. Baselines medidos para saber dónde estamos parados
(logística, GBM). → `propuesta.md §2, §6, §7`

## 15/08 — Implementación v1 + infraestructura

**Hecho.** Paquete `btr/` con los bloques de la demo (corrigiendo el doble escalado por √d_k que
trae la demo — el paper escala una sola vez), `FeatureTokenizer`, entrenamiento con early stopping.
Carpetas `resultados/` (JSON por corrida) y `pesos/` (checkpoints recargables), soporte GPU.

**Duda del equipo: ¿no habría que partir por producto en vez de por query?** Instinto correcto
(no queremos en validation cosas demasiado relacionadas con train). Verificado: 9.910 títulos
únicos en 10.000 filas, 1% de test con título visto en train, y las métricas son idénticas
restringiendo test a productos jamás vistos. El modelo no recibe identidad del producto, así que
no tiene canal para memorizarlo. La distinción clave: leakage = información no disponible al
predecir; que los *patrones de features* se repitan es la señal aprendible, no fuga.
→ `propuesta.md §7.1`

## 16/08 — "Esto corre demasiado rápido en CPU"

**Discusión.** Objeción del equipo: la cátedra dio a entender que haría falta GPU; si entrena en
30 segundos en CPU, algo estamos haciendo mal. Análisis honesto: el modelo tabular chico NO está
mal (10k filas, secuencia 14, `d_model<100` como sugiere el enunciado). Pero releyendo la clase 2:
toda la materia es el pipeline texto → tokens → embeddings → PE → transformer, y "no se enrosquen
con el tokenizador" presupone que hay un tokenizador. Nuestro regex optimizó justo la parte donde
el transformer necesita cómputo. Además, las entregas de SIA muestran el volumen de experimentación
esperado (grillas, 5 seeds, corridas larguísimas).

**Decisiones.**
- Implementar la formulación **`text`** (caracteres de title+description, como la demo; secuencia
  ~257, positional encoding ahora sí necesario, máscara de padding) y **`hybrid`**.
- Experimento central del TP: ¿el transformer redescubre desde los caracteres la señal que
  parseamos a mano? Costos medidos: tabular ~30 s/corrida CPU; texto ~74 s/época CPU → GPU.
- → `propuesta.md §4C, §7.5`; código en `btr/` (`--formulation`).

## 16/08 — ¿Es trampa usar "Best Seller"? (la discusión conceptual del TP)

**Objeción del equipo.** Los badges se asignan *después* de vender mucho; un producto nuevo nunca
va a decir "Best Seller" → ¿circular? ¿no habría que sacar esa información?

**Análisis.** Tres clases de features: (1) outcome del mismo evento (`cart`) = leakage estricto,
nunca; (2) estado del producto al momento de la impresión (badges) = disponible al predecir y
plausiblemente *causa* de la compra (prueba social) → válido para la tarea de predicción, PERO
circular para decidir promociones y ciego al cold-start; (3) atributos intrínsecos = válidos
siempre. Medido: sin la señal de estado, hasta el GBM queda casi en azar (PR 0.162 vs 0.134) y el
61% de las filas está en tiers de BTR = 0.000 — la cátedra puso la señal para que el EDA la
encuentre; sacarla del todo deja un problema impredecible.

**Decisión: dos familias de modelos**, porque responden preguntas distintas: *catálogo* (con
estado — qué promocionar hoy) e *intrínseco* (`--drop-features listing_status --strip-status` —
qué esperar de un producto nuevo; `--strip-status` también limpia la oración de estado de la
descripción, si no la variante de texto la re-aprende). El contraste 0.76 vs 0.16 es un hallazgo
presentable. → `propuesta.md §2.3.1`

## 16/08 — Zoo de arquitecturas y suite de experimentos

**Discusión.** "Que use un transformer en algún lugar" no obliga a que todo sea un transformer
(y nunca pide decoder-only). Idea del equipo: transformer solo como encoder del texto → embedding
→ MLP clasificador. Enfoque de trabajo: probar MUCHAS alternativas y elegir con resultados.

**Hecho.** `--arch transformer/mlp/tower/listwise`: `mlp` = baseline sin atención (mismos
embeddings, aísla "¿la atención aporta?"), `tower` = la idea del equipo (atención que NO cruza
texto↔features — comparada con `hybrid` mide cuánto vale la atención cruzada), `listwise` =
formulación B. Primeros números (seed 42): transformer 0.766 > MLP 0.715 > listwise 0.664.
Suite `experimentos.py`: 24 configuraciones × 3 seeds, `--resumen` para la tabla.
→ `propuesta.md §4, §7.4`

## 16/08 — Idea: embeddings query↔producto estilo skipgram

**Discusión.** ¿Aprender embeddings de queries y productos con pares positivos/negativos como el
skipgram de la clase de embeddings? Es una técnica real (item2vec / two-tower, Airbnb/YouTube).
Análisis: nuestra BCE ya optimiza ese objetivo **con mejores negativos** (los productos realmente
impresos y no comprados, en vez de negativos random); el producto interno u·v es una restricción
de escalabilidad para *retrieval* sobre millones de ítems, y nuestro problema es *ranking* de ≤8
candidatos; un embedding por `query_id` haría overfitting y cold-start. La información de query
que importa ya está en el modelo (`price_rel`) y la composición de página la mide listwise.

**Decisión.** Descartada acá con justificación; **reciclada como respuesta del Ejercicio 3**
(cambiar "query" por "usuario": embeddings de usuario/producto con negative sampling = la
personalización clásica, conectada con la teoría de la clase 2). → `propuesta.md §4D, §10`

## 16/08 — Robustez de la suite para la máquina con GPU

**Hecho.** La suite quedó a prueba de accidentes: usa la GPU automáticamente y **aborta con
instrucciones** si la familia texto fuera a correr en CPU (torch mal instalado); es **resumible**
(saltea toda corrida cuyo JSON ya existe — si se corta, se relanza la misma línea y sigue);
guarda checkpoints por defecto; si una corrida falla, sigue y reporta al final. Los números del
EDA y baselines quedaron reproducibles en `eda/verificaciones.py` (y la tabla de baselines de la
propuesta se re-midió con el mismo split por query que usan los modelos, para que todo sea
comparable). Este archivo (`bitacora.md`) queda como registro de las discusiones.

## 16/08 — Diagramas de las arquitecturas

**Pedido.** Un gráfico/esquema por arquitectura, completo (no solo el transformer), para poder
imaginarse cada una.

**Decisión.** Dos versiones con el mismo contenido: `diagramas.md` en el repo (mermaid, GitHub lo
renderiza solo, queda versionado) y una versión visual más rica como página con SVG (código de
color compartido entre las seis figuras: teal=tabular, ámbar=texto, violeta=CLS, azul=MLP/lineal,
frambuesa=salida; flechas anotadas con los shapes reales de los tensores). Las dimensiones salen
del código real (`btr/model.py` instanciado): 28.289 parámetros el tabular, 126.209 el MLP,
35.713 el de texto, 39.073 el híbrido, 96.225 la torre, 41.601 el listwise. Dato lindo que salió
de armarlos: el MLP baseline tiene 4,5× los parámetros del transformer tabular y aun así rinde
menos — la ventaja de la atención no es cuestión de tamaño.

## Pendientes

- [ ] Correr la suite completa en la RTX 3070 (`experimentos.py`) y analizar `--resumen`.
- [ ] Notebook prolijo del EDA (Ej. 1) a partir de `eda/verificaciones.py`, con gráficos.
- [ ] Mapas de atención del modelo final (¿el CLS mira al listing_status? ¿price_rel × tier?).
- [ ] Consultar a la cátedra las preguntas de `propuesta.md §11`.
- [ ] Armar la presentación (25-30 min) con la estructura de SIA: experimento → tabla de config →
      resultado → decisión.
