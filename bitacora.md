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

## 16/08 — Todas las métricas + el laboratorio interactivo (panel)

**Pedido.** Una página interactiva y modular: elegir cada decisión (features, split, estrategia
de entrenamiento, encodings — incluso por feature —, cantidad de bloques, etc.), que dé la
combinación armada para probar, que lo ya armado quede guardado en el código, y que si hay
resultados los muestre; y que se calculen SIEMPRE todas las métricas con sentido para poder
graficar cualquiera después.

**Hecho (dos partes).**
1. `compute_metrics` en `btr/train.py`: cada época (train y val) y cada corrida final (val y
   test) guardan **16 métricas** (PR-AUC, ROC-AUC, loss, log-loss sin pesar, Brier, F1 máximo +
   su umbral óptimo, precision/recall/F1/accuracy/balanced/especificidad/MCC @ 0.5, tasas de
   positivos). Se hizo ANTES de correr la suite en la GPU a propósito: así las 72 corridas ya
   nacen con el set completo. Dato: el F1 máximo del transformer tabular es 0.784 con umbral
   0.312 — no 0.5, consecuencia directa del 13% de positivos.
2. `panel.py` genera `panel.html` (el "Laboratorio BTR"): configurador por decisiones que
   muestra el comando exacto, detecta si la combinación ya está en la suite (`--only nombre`),
   si está soportada por el código, o si requiere implementación (en ese caso arma un spec JSON
   para pegar en el chat y pedirla — así quedaron representados sin implementar: GroupKFold,
   split por producto, target encoding, one-hot directo al MLP, encodings numéricos por feature,
   métricas por página). Si la config ya tiene corridas: todas las métricas (media ± desvío),
   curvas por época de cualquier métrica y ranking global clickeable. La identidad de una config
   es su **clave canónica** (misma función en Python y JS, con auto-test al cargar la página que
   avisa si divergen; verificada también con node: 27/27 claves y 6/6 comandos).

**Por qué así.** La página es estática (no puede entrenar ni leer archivos): el estado vive en el
repo (suite en `experimentos.py`, corridas en `resultados/`) y `panel.py` lo embebe al generar.
Ciclo: elegir en el panel → correr el comando → `python panel.py` → republicar el artifact.

## 16/08 — Primera tanda GPU: análisis, reparación de métricas y segunda tanda

**Qué pasó.** Matias corrió en la 3070 las 24 configs con el protocolo original Y una variante
`pac20_` (paciencia 20, tope 300), seeds 42–47: 288 corridas. Las corrió con el código previo a
`compute_metrics`, así que sus JSONs traían 3 métricas; se recuperaron las 16 finales SIN
reentrenar recargando cada checkpoint (`eda/recalcula_metricas.py`, ~10 min — validado: el
PR-AUC recalculado coincide exactamente con el original).

**Análisis completo en `analisis.md`.** Lo grande: campeón `pac20_feat_h1` **0.816 ± 0.026**
(supera al GBM 0.762 con claridad); atención +0.048 sobre MLP apareado (5/6 seeds); paciencia 20
ayuda en tabular (21/24) y daña texto puro; 1 cabeza grande > 4 chicas; CLS > mean (6/6);
pos_weight y bins dañan; **hybrid recupera desde los chars la señal del regex (sin_regex ≈
full), tower no** (−0.04: cuello de botella del embedding único); familia intrínseca clava el
techo (~0.16); y `feat_causal` estaba **degenerado por diseño** — con máscara causal el CLS en
posición 0 solo se ve a sí mismo (verificado: p constante 0.2214, ROC 0.500 exacto). Calibración
(propuesta #3 de Junior): ECE ~0.01, T≈1 → no hace falta corregir; el BTR agregado sobreestima
~+0.9 puntos.

**Implementado a partir de esto** (todo smoke-testeado, checkpoints viejos siguen cargando):
`--cls-position last` (causal bien hecho); `--cat-encoding target/freq/hashing/onehot` (el
"modular/columnar" del compañero = hashing con módulo; one-hot solo para MLP porque en el
transformer ≡ embedding); `--extra-features` (volumen desde dimensions, package parseado,
nº de ingredientes — pedido de Fer); `--cart-aux λ` (multi-task de Junior #2, cart como label
auxiliar, nunca input); `--listwise-texto` (Junior #1, la torre de chars dentro del token de
producto); `eda/calibracion.py` (Junior #3) y `eda/graficos_resumen.py` (+matplotlib en
requirements). **15 configs nuevas en la suite** (grilla campeón, causal_last, 4 encodings,
extras, 3 λ de cart, listwise_texto, text_len96); protocolo nuevo: 6 seeds, tabulares con
paciencia 20. El panel y el zoo quedaron actualizados con los resultados y los ejes nuevos.

## 16/08 — El estado como campo separado y encodings con orden (idea de Fer)

**Pedido.** Separar el "Best Seller" del título al analizarlo y ponerlo como un campo más, con
el encoding apropiado — "incluso asignando una especie de orden, pero que hay que pensarlo bien".

**Análisis.** El campo ya existía (`listing_status`, parseado por regex desde el día 1). Lo
genuinamente nuevo del pedido son dos cosas: (a) que al analizar el TEXTO el sufijo no viaje
duplicado adentro del título — con las corridas de la 1ª tanda esto completa un **2×2 limpio**
{token parseado sí/no} × {sufijo en el texto sí/no} del que ya teníamos 3 celdas (full,
sin_regex, intrinseco); y (b) el orden: uno **semántico a mano es indefendible** en este dataset
(EDA §2.3: "Highly Rated" suena igual que "Top Rated" y compra 50× menos — el wording no predice
el tier), pero sí hay dos órdenes defendibles derivados del BTR de train: **ordinal** (solo el
rango, normalizado) y **target** (rango + magnitud). Hipótesis: target ≥ ordinal (los tiers
tienen saltos de magnitud enormes) y embedding ≥ target (el embedding puede aprender cualquiera
de los dos); el experimento mide cuánto cuesta comprimir el campo a UN escalar.

**Hecho.** `--cat-encoding ordinal` (nuevo modo) y `--cat-feature-encoding feature=modo` para
aplicar un encoding SOLO a un campo (p. ej. `listing_status=ordinal`, el resto embedding) —
esto además volvió real el eje "encoding por feature categórica" del panel. 5 configs nuevas:
`hybrid_status_campo` y `tower_status_campo` (la celda faltante del 2×2: texto limpio + campo),
`feat_ordinal`, `feat_status_ordinal`, `feat_status_target`. Round-trip de checkpoints con
encoding mixto verificado (ROC idéntico al recargar). → `analisis.md §4.1`

## Pendientes

- [x] ~~Correr la suite completa en la RTX 3070 y analizar~~ → hecho (1ª tanda), ver `analisis.md`.
- [ ] Correr la SEGUNDA tanda en la 3070 (mismas dos líneas; corre solo lo nuevo) y analizarla.
- [ ] Tras cada tanda: `python panel.py` para re-embeber resultados y pedir republicar el artifact.
- [ ] Notebook prolijo del EDA (Ej. 1) a partir de `eda/verificaciones.py`, con gráficos.
- [ ] Mapas de atención del modelo final (¿el CLS mira al listing_status? ¿price_rel × tier?).
- [ ] Consultar a la cátedra las preguntas de `propuesta.md §11`.
- [ ] Armar la presentación (25-30 min) con la estructura de SIA: experimento → tabla de config →
      resultado → decisión.
