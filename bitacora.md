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

## 17/08 — Segunda tanda: campeón ordinal, y la revisión externa

**Resultados 2ª tanda** (116 corridas; lectura completa en `analisis.md §5`). Lo grande:
**campeón nuevo `feat_ordinal` 0.824 ± 0.018** — la idea del ORDEN de Fer aplicada globalmente
(nuestra hipótesis "embedding ≥ target ≥ ordinal" quedó invertida: el rango como prior regulariza).
Sorpresas honestas: one-hot > embeddings en el MLP (6/6; parte del déficit del MLP era la entrada,
no la falta de atención — el enunciado fino queda +0.048 misma representación / +0.027 contra el
mejor MLP); el 2×2 del estado confirmó que el canal doble diluía al híbrido (status_campo 0.733 >
full 0.705); listwise_texto +0.016 (2/2) — Junior #1 validada; cart-aux dentro del ruido — Junior
#2 respondida en negativo; bidireccionalidad da igual (causal_last ≈ base, cierre del arco del
bug del CLS); freq/hashing destruyen la señal (contraejemplos pedagógicos); extras y text_len96
confirman al EDA.

**Revisión externa** (conversación de Fer con otro agente; veredicto punto por punto en
`analisis.md §6`). Casi todo ya estaba implementado y medido; lo valioso que faltaba quedó hecho:
(1) **tokenización word-level** (`--text-tokens words`, la que la clase recomendaba — éramos
char-level por la demo); (2) **fusión** (`--formulation fusion`): el CLS de la torre de texto como
token 15 de la secuencia tabular — el punto medio entre hybrid (diluido) y tower (sin cruce);
(3) **word2vec pre-entrenado vs end-to-end** (`--w2v-init`, skipgram sobre train — la conexión
clase 1 ↔ clase 2); (4) `feat_tiempo` (hora/día, para verificar el descarte del EDA);
(5) **mapas de atención** (`eda/atencion.py`): la capa 1 pone el 51% de la atención del CLS en
`status` y la familia de precio se consulta entre sí — el modelo mira donde el EDA dijo.
Adoptamos también su corrección conceptual (la atención no pide tokens "del mismo tipo" sino el
mismo espacio ℝ^d) y su conexión column-embedding ≈ nuestro `feat_pos` (Δ≈0 la confirma). Su
split temporal: rechazado con evidencia (timestamps rotos). Su MLM-sobre-features: anotado como
opcional (él mismo advierte el scope). 3ª tanda en la suite: 8 configs (`camp_ordinal_*`,
`feat_tiempo`, `text_words`, `fusion_*`). El generador del Zoo ahora vive en el repo (`zoo.py`) —
antes estaba en un scratchpad de sesión y se perdió en una limpieza; reconstruido con la figura
de fusión.

## 18/08 — Tercera tanda: convergencia. Modelo final: `feat_ordinal`

**Resultados** (50 corridas; lectura completa en `analisis.md §8`). La búsqueda **convergió**:
sumarle capacidad al campeón ordinal empeora todo (−0.02 a −0.05) — ordinal gana por ser un prior
simple, y la capacidad extra reintroduce el overfit que ordinal eliminaba. **Elección del modelo
final con disciplina**: en validación hay empate técnico entre 4 configs (Δ<0.002) → desempate
por parsimonia (26.177 parámetros, menor desvío, menor gap val→test), NO por test; test lo
confirma después. **Modelo final: `feat_ordinal` — PR-AUC test 0.824 ± 0.018, ROC 0.975, F1 máx
0.784 @ umbral ~0.40, Brier 0.042.** Su mapa de atención: el CLS pone **0.75** de su atención en
`status` en capa 1 (el escalar ordinal ES la propensión y el modelo lo sabe).

**El veredicto del 5B** (lo más elegante de la tanda): fusión **+0.069 sobre el híbrido (6/6)** —
comprimir el texto a un token cura la dilución por completo — pero **empata exacto con la torre
(Δ −0.0002)**: una vez comprimido, cruzar por atención o por concat da igual. Y sigue sin superar
al tabular puro: el texto solo importa en el mundo sin regex. Words ≈ chars en texto puro; en
fusión words sobreajusta su tabla de embeddings y **w2v-init la regulariza (+0.010, 4/6)** — la
conexión clase 1→2 funciona en la dirección esperada, sin alcanzar a chars: el tokenizador chico
de la demo era el correcto a esta escala. Cierres: `feat_tiempo` −0.022 (cuarta vindicación del
EDA), `listwise_texto` a 4 seeds +0.005 (más modesto que con 2). No hay 4ª tanda: no queda
dirección abierta que prometa; lo que sigue es análisis y presentación.

## 18/08 — Ideas post-convergencia: tres gratis, ejecutadas en el momento

**Pedido.** "¿Alguna otra idea para probar?" Respuesta: tres que no requieren GPU (corren sobre
los 454 checkpoints) se implementaron y corrieron en el acto; el resto quedó propuesto con costo.

1. **Ensemble de configs** (`eda/ensemble.py`): promediar p(bought) de varias configs del mismo
   seed (comparten split → apareado válido), composición elegida por VAL. Resultado:
   **0.834 ± 0.021, +0.0099 sobre feat_ordinal, gana 6/6 seeds** — la mejora más consistente del
   proyecto. Val eligió h1+target+d64h1l4 (sin ordinal: prefiere diversidad de representación).
   El TP queda con dos números finales: modelo único 0.824 / ensemble 0.834.
2. **Importancia por permutación** (`eda/importancia.py`): status +0.68, price_rel +0.14,
   allergens +0.05, resto ≈0 — converge con los mapas de atención y con el EDA. Respuesta
   preparada a "attention is not explanation": dos diagnósticos independientes, misma historia.
3. **Métricas por página** (`eda/metricas_pagina.py`): top-1 0.912 (azar 0.267), MRR 0.954,
   NDCG 0.964 — la traducción del modelo al uso de negocio (elegir qué promocionar por página).

Propuestas restantes (si se quiere seguir): curva de aprendizaje (25/50/75/100% de train, barata
en GPU, slide clásica); --init-seed para separar varianza de split vs inicialización (habilita
deep-ensembles puros); logística + cross manual price_rel×tier (¿cuánto del aporte de la
atención es esa única interacción?); MLM sobre features (el "guiño" de la revisión externa,
scope medio); GroupKFold para intervalos más finos.

## 18/08 — Cuarta tanda preparada (las 5 ideas, sin tocar nada de lo existente)

**Pedido de Fer**: preparar las cinco ideas para mandarlas a correr a la noche. La quinta
(logística + cross manual) no necesita GPU y se corrió en el acto: **+0.015 (6/6) pero explica
solo ~12% del gap** logística→transformer — la atención aprende más que "el" cruce del EDA
(`eda/cross_manual.py`, analisis §9.4). Las otras cuatro quedaron implementadas y en la suite
(15 configs nuevas, 67 totales; analisis §10): `--train-frac` (curva de aprendizaje),
`--init-seed` (varianza split vs init + deep-ensemble puro), `--pretrain-mlm` (MLM sobre
features, con [MASK] aprendido y cabezas temporarias por feature), `--cv-k/--cv-fold`
(GroupKFold por query con val recortada del resto). Todo smoke-testeado; claves canónicas
únicas (67) y espejo JS verificado en node (143/143). El eje "GroupKFold" del panel dejó de
ser "a pedir".

## 18/08 — Cuarta tanda corrida: el capítulo experimental queda sellado

**Todo completo** (90/90; `listwise_texto` queda en 4 seeds por decisión de costo — no cambia
conclusiones). Resultados (analisis §10): curva de aprendizaje **casi saturada** (el último 25%
de datos aporta +0.007; con 75% ya se supera al GBM-100%); la varianza es **~55% split / ~45%
init** (grilla 5×6 — valida promediar seeds y explica por qué el ensemble ayuda); **MLM
regulariza a los embeddings (+0.011) pero no aporta sobre ordinal** (−0.007): regularizadores
alternativos, gana el más simple; **deep-ensemble puro +0.0095 → 0.8334**, convergiendo con el
ensemble de configs (0.8339) en el mismo techo **~0.834**; GroupKFold 5×6: **0.8207 ± 0.0119**,
el intervalo fino de cabecera. Bug corregido en el camino: los globs de checkpoints de los
scripts de eda matcheaban también `feat_ordinal_mlm20` (no existía cuando se escribieron) —
precisados a `{tag}_features_*`. No queda nada por correr en GPU.

## 20/08 — La carpeta /entrega: lo que efectivamente se entrega

**Pedido de Fer**: una carpeta `entrega/` con el modelo final y su código (solo esa
arquitectura), los experimentos/resultados/análisis, y la presentación completa con guión —
narrada limpia (base FT-Transformer → experimentos de a un aspecto), no la maraña cronológica
real. Verificado al final contra los transcripts (sobre todo clase2b) y el Enunciado.pdf.

**Hecho.** `entrega/modelo/` (data/model/train/predecir + 6 checkpoints en formato plano sin
pickles de clases; equivalencia verificada: tensores idénticos al pipeline del repo y
predicciones allclose 1e-6), `entrega/experimentos/` (analisis.md adaptado + CSV de las 91
configs + 7 figuras), `entrega/presentacion/` (21 diapositivas autocontenidas — 20 + backup —
con guión de ~27 min y apéndice para preguntas), `entrega/README.md` con la tabla de cobertura
del enunciado punto por punto. La auditoría contra clase2b encontró y corrigió dos cosas:
(1) la corrida de sanidad había pisado el checkpoint del seed 42 con un modelo CPU de la cola
mala (0.694) — re-convertido del GPU (0.8042 ✓); (2) faltaba la VISUALIZACIÓN de
overfitting/underfitting que el enunciado pide explícitamente — nueva figura
`curvas_entrenamiento.png` + diapositiva de backup. Nota de reproducibilidad documentada: la
varianza de entrenamiento tiene cola izquierda (grilla: 0.718–0.870) — por eso el protocolo
promedia 6 seeds y los pesos entregados son los de la suite.

## 21/08 — Idea de Fer: pesos por feature dentro del transformer (5ª tanda)

**Pedido.** "Cada feature con sus parámetros específicos: su propio W_q (y W_k y demás) y su
propia MLP por token." Evaluación: la idea es *principled* — el weight-tying entre posiciones es
un sesgo pensado para posiciones intercambiables, y acá la posición ES el feature; ya hacemos
identidad-por-parámetros en la entrada (tokenizador), esto la extiende a los bloques. No es el
estándar de los transformers tabulares → ablación fresca y defendible. Hipótesis honesta
registrada ANTES de correr: multiplica parámetros ×4–12 y este TP viene demostrando que el prior
simple le gana a la capacidad → lo esperable es que no supere a 0.824; correrlo cierra la
pregunta del sesgo inductivo en cualquier caso.

**Hecho.** `HeadPorFeature` (W_q/W_k/W_v de shape (T,d,h) por posición, einsum) y
`FeedForwardPorFeature` (14 MLPs) en `btr/model.py`, flag `--per-feature {qkv,ffn,both}` con
guardias (solo transformer features, sin causal); checkpoints viejos intactos. 4 configs nuevas
(71 totales): pf_qkv / pf_ffn / pf_full sobre el campeón ordinal + pf_full_emb sobre embeddings.
Panel con el eje nuevo (canon 162/162 verificado en node). La entrega NO se toca salvo que gane.

## 21/08 — El contrapeso: reducir complejidad (idea de Fer) — la 5ª tanda queda en 10 configs

**Pedido.** Junto al per-feature (que agrega capacidad), probar lo opuesto: reducir la
complejidad para contrarrestar el overfitting esperado, sobre la mejor base actual
(feat_ordinal, con "(Best Seller)" parseado a categórica ordinal — confirmado).

**Diseño (3 patas + lo ya preparado = un espectro alrededor del campeón).** (a) Minimalismo:
min_d16/d8/l1/d16l1 — achicar SOBRE ordinal nunca se había probado; min_d16 = 6.945 params.
(b) `pf_gate`: especialización barata — compuertas diagonales por posición sobre los W
compartidos, init=1 ⇒ arranca idéntico al campeón (VERIFICADO con predicciones) y aprende solo
la desviación. (c) `pf_qkv_d16`: desatado compensado — 26.913 params ≈ los 26.177 del campeón:
misma escala, ¿especializar o compartir? Espectro completo: 6.9k → 323k parámetros. 6 configs
nuevas (77 totales en la suite); panel con la opción gate (canon 168/168 en node).

## 21/08 — 5ª tanda corrida: la idea per-feature, medida — y el modelo de 3.713 parámetros

**Resultados** (60 corridas; lectura completa en `analisis.md §11.2`). La hipótesis
pre-registrada se corrigió en las dos direcciones. El desatado NO colapsa por overfitting
(early stopping + weight decay lo contienen): pf_qkv 0.8284 y pf_full 0.8262 EMPATAN con el
campeón dentro del ruido (3/6); solo pf_ffn (245k params) queda abajo. **El hallazgo real:
`pf_full_emb` +0.0205, gana 6/6 sobre embeddings compartidos** — la idea de Fer funciona (la
especialización por feature aporta identidad), pero su beneficio es redundante con el prior
ordinal, que ya la da más barata. A presupuesto fijo d16: especializar > compartir (+0.0094,
4/6). **Titular del contrapeso: `min_d16l1` (d16, 1 bloque, 3.713 parámetros) EMPATA al
campeón** (val 0.8350 vs 0.8345; test 0.8254 vs 0.8239) — compresión 7× gratis; "el prior
simple gana" en su forma final. **Lección metodológica en vivo**: pf_ffn tiene la MEJOR val del
proyecto (0.8435) y no es mejor en test (gap 0.026, 2,5× el del campeón) — sobreajuste de
selección con 100+ configs probadas; por eso la selección del modelo final quedó CERRADA en la
4ª tanda y estas tandas son exploratorias. **El modelo final no cambia** (feat_ordinal);
min_d16l1 queda registrado como su versión comprimida equivalente. Entrega: solo se actualizó
el apéndice del guión (respuesta preparada), la copia del análisis y el CSV (101 configs).

## 22/08 — 6ª tanda preparada: regularización + transfer learning (clase 3) + SIA

Disparador: dos preguntas de Fer con la clase 3 recién subida (transcript + PPT de transfer
learning & fine-tuning, leídos completos). (1) ¿Hubo pruebas de regularización? Respuesta
honesta: la regularización efectiva fue early stopping + capacidad + 6 seeds + el prior
ordinal; **weight decay quedó en 1e-2 (default de AdamW, nunca fue una decisión), dropout fijo
en 0.1, y residuales/LayerNorm jamás se ablacionaron** → tanda `reg_*`/`abl_*` (11 configs).
(2) ¿Transfer learning / Kohonen / PCA / autoencoders? La clase 3 da el marco exacto y las
TRES técnicas son implementables con nuestros checkpoints como teachers: `tl_probe` (feature
extraction: tronco congelado + probe lineal — el smoke de 1 época ya da 0.8055), `tl_mlm_*`
(¿cuánto del 0.82 se alcanza SIN labels? + fine-tuning anclado L2-SP, la "KL penalty" de la
clase), `tl_distill_*` (**la apuesta: destilar el deep-ensemble 0.833 del mismo split en UN
d32 — y en el min de 3.713 params**; soft labels > labels duras, dixit clase 3), `tl_emb_mlp`
("el embedding más un montón de cosas"). SIA: `sia_som*` (celda BMU de Kohonen como categórica
extra), `sia_ae_cls*` (AE con cuello en el CLS, hermano del MLM), `sia_pca_mlp`/`sia_ae_mlp`
(representación no supervisada como única entrada del MLP — PCA ≙ Oja/Sanger). Hipótesis
pre-registradas en `analisis.md §13`; la selección del modelo final SIGUE cerrada (todo
exploratorio). Infra nueva en `btr/train.py` (16 flags), `experimentos.py` (27 configs, total
104), panel con sección 5b + clave canónica extendida (test de paridad Python↔JS ahora
versionado: `eda/test_canon_panel.js`, 205 claves OK). Smokes CPU de TODOS los caminos nuevos:
OK. Teachers verificados en `pesos/` (git): campeón + robu_init43..47 por seed.

## 22/08 — 6ª tanda corrida y analizada (162 corridas; 766 totales, 104/104 configs de suite)

Análisis apareado en `analisis.md §13.1`. **Regularización**: `reg_nada` (wd 0 + dropout 0)
−0.0004 (4/6) — la hipótesis "decorativa" CONFIRMADA: a 26k params, early stopping + prior
ordinal bastan; sobre-regularizar daña (do03 −0.011, fdrop02 −0.029, ls01 −0.015). El matiz:
**sin residuales el modelo casi no entrena (−0.59, PR 0.23, inestable entre seeds)** — 10× el
daño predicho, la pieza más crítica de la demo; sin LayerNorm −0.037 (0/6). **Transfer**:
`tl_probe` +0.0025 y **gana 6/6, nunca pierde** — la representación del campeón es linealmente
separable (explica retroactivamente el empate de min_d16l1); `tl_mlm_probe` 0.157 — el
self-supervised puro NO captura la tarea (peor que la logística cruda 0.660): su valor era como
init, espejo medido de la transferability de la clase 3; `tl_mlm_l2sp` el ancla no ayuda
(−0.011 vs mlm sin ancla — no hay preentrenado fuerte que retener); distillation: same/mix
neutro-positivo (4/6), `tl_distill_ens` +0.0033 pero 2/6 (la media la hacen 2 seeds; la grande
es la MEJOR seed del campeón → varianza, no rescate), soft targets convergen más rápido (47-57
vs 64 épocas); **titular: `tl_distill_ens_min` test 0.8274 con 3.713 params** — 4º mejor
single-model del proyecto, nunca peor que −0.004 por seed; `tl_emb_mlp` el MLP con el embedding
congelado EMPATA al transformer (−0.006, 3/6) — la atención ya trabajó dentro del extractor — y
su val 0.8458 (la más alta de la tanda) con gap 0.028 es el TERCER ejemplo vivo de sobreajuste
de selección. **SIA**: SOM resta (−0.017/−0.033; figura nueva `graficos/som_btr.png`: organiza
las numéricas pero el BTR por celda queda 0.09–0.20 ≈ base — la señal vive en el status);
AE-CLS replica el patrón MLM; PCA/AE→16d como entrada del MLP DESTRUYEN la señal (0.20/0.23 vs
0.75): reconstruir ≠ predecir, medido. **Modelo final NO cambia**; selección cerrada. Sync:
panel re-embebido (232 claves Python==JS OK) y republicado; CSV 128 grupos; copia de análisis y
guión-apéndice (3 respuestas nuevas: regularización / transfer / SIA) actualizados en entrega/.

## 23/08 — 7ª mini-tanda preparada: el piso de la compresión (con y sin teacher)

Fer ofreció más GPU; auditoría honesta de qué queda: el programa está cerrado (104/104 con
respuesta) y lo único genuinamente abierto es la curva "PR vs parámetros" en dos ramas — sin
teacher degrada a partir de 1.937 params (min_d8 0.8142), pero NO hay puntos destilados por
debajo de 3.713. 5 configs (30 corridas, ~30-45 min): min_d8l1 (1.089) / min_d4l1 (**353
params**) + destiladas d8/d8l1/d4l1 desde el deep-ensemble. Hipótesis pre-registrada
(analisis.md §13.2): las soft labels corren el piso ~un nivel; si no, el dark knowledge no
compra compresión acá — ambas salidas cierran la figura. Rechazado a propósito: más seeds del
campeón (fishing de ±0.004), combos do0+distill (fishing), refit train+val sin val (rompe la
higiene del protocolo por +0.007 estimado), listwise seeds 46-47 (decisión de costo ya
tomada). Selección sigue cerrada. Suite 109 configs; panel 237 claves OK republicado.

## 23/08 — 7ª mini-tanda corrida: la curva de compresión, cerrada (796 corridas, 109/109)

Hipótesis confirmada en su forma exacta (analisis.md §13.3, figura
graficos/curva_compresion.png): **las soft labels corren el piso de la compresión un nivel**
— el nivel campeón, que en plain requiere 3.713 params, destilando del deep-ensemble aguanta
hasta **1.937 params** (0.8282, +0.0140 sobre plain, gana 5/6 — el efecto de destilación más
grande y consistente del proyecto, exactamente donde la capacidad empieza a faltar). A 353
params las dos ramas caen juntas (−0.0006): no hay dónde guardar el conocimiento — el umbral
está entre 353 y ~1.1k. La rama destilada es PLANA en nivel campeón de 1.937 a 26.177
(0.8282/0.8274/0.8272): "el conocimiento del problema cabe en ~2.000 parámetros si te lo
enseña un ensemble". Capítulo experimental CERRADO de verdad: sin preguntas abiertas que
ameriten GPU. Sync: CSV 133 grupos, panel 242 claves OK republicado, zoo regenerado
byte-idéntico (ya estaba al día), guión-apéndice actualizado con la curva, figura copiada a
entrega/. Modelo final sin cambios: feat_ordinal.

## Pendientes

- [x] ~~Correr la suite completa en la RTX 3070 y analizar~~ → hecho (1ª tanda), ver `analisis.md`.
- [ ] Correr la SEGUNDA tanda en la 3070 (mismas dos líneas; corre solo lo nuevo) y analizarla.
- [ ] Tras cada tanda: `python panel.py` para re-embeber resultados y pedir republicar el artifact.
- [ ] Notebook prolijo del EDA (Ej. 1) a partir de `eda/verificaciones.py`, con gráficos.
- [ ] Mapas de atención del modelo final (¿el CLS mira al listing_status? ¿price_rel × tier?).
- [ ] Consultar a la cátedra las preguntas de `propuesta.md §11`.
- [ ] Armar la presentación (25-30 min) con la estructura de SIA: experimento → tabla de config →
      resultado → decisión.

## 23/08 — Figura EDA retro-agregada (pregunta de Fer: ¿matriz de correlación / PCA?)

Fer preguntó si en algún momento hicimos matriz de correlaciones, relación con el target y/o
un PCA del dataset (candidato a abrir la presentación), y si eso podría haber cambiado
decisiones. Auditoría honesta: la relación con el target se hizo **variable por variable**
(propuesta §2) y de ahí salieron todas las decisiones (cart afuera, extracción del status,
price_rel, descartes §2.6, timestamps); la **figura** matriz no existía, y PCA solo se había
usado como compresión de entrada (sia_pca_mlp, 6ª tanda), no como visualización.

Hecho (CPU, sin GPU): `eda/grafico_correlacion.py` → `graficos/matriz_asociacion.png` —
matriz de asociación mixta (|Spearman| / V de Cramér / η, 19 features + target, incluidas las
columnas descartadas marcadas con †) + PCA 2D de la matriz cruda coloreado por bought.
Lecturas (analisis.md §14): fila del target = desierto con un pico (status 0.75, resto <0.09);
bloques de redundancia = los descartes de §2.6 en imagen (unit↔pkg_unit 1.00,
category↔n_ingredients 1.00, category↔filtros 0.95–0.99, price↔price_rel 0.80); PCA no separa
clases (27% var en 2 PCs) — contracara visual de sia_pca_mlp. Salvedad: U invertida del precio
invisible para medidas monótonas (price 0.003) → la matriz complementa, no reemplaza, el EDA
por variable. **No habría cambiado ninguna decisión** — todo lo fuerte de la matriz ya estaba
decidido por otra vía. Lo de "sacar Best Seller del título y encodearlo": hecho desde el día 1
(`listing_status`). Sync: figura y análisis copiados a entrega/, guión §3 (alternativa de
apertura) + apéndice (bullet nuevo) + fix 766→796.

## 23/08 — 8ª tanda preparada: transfer desde un preentrenado externo (idea de Fer)

Fer propuso lo que faltaba del arco de transfer: usar el embedding de un modelo conocido para
title+description como entrada de nuestro transformer, en dos regímenes — congelado (feature
extraction) y actualizando el encoder (fine-tuning). Decisiones: MiniLM-L6-v2 (22M params, el
estándar chico de sentence-transformers; entrenable en la 3070); el embedding entra como UN
token extra (proyección aprendida 384→32, mismo mecanismo que fusion) para no repetir la
dilución del hybrid; los embeddings congelados se precomputaron acá en CPU
(eda/embed_texto.py, ~3 min) y van commiteados en embeddings/*.npy (2×15MB) — la 3070 solo
necesita `transformers` para los dos configs de fine-tuning. Salvedad de reglas: el enunciado
pide transformer propio; acá el preentrenado es entrada, no el modelo — igual se reporta como
exploratorio y la selección sigue cerrada.

7 configs bert_* (116 totales, 42 corridas nuevas): solo/solo_intr (¿cuánto ve MiniLM con y
sin status?), mlp/token (redundancia con la categórica), token_sin/ft_sin (¿reemplaza al
regex, congelado vs fine-tuneado?), ft. Hipótesis pre-registradas en analisis.md §15 — la
apuesta central: la partición de tiers es ANTI-semántica ("Highly Rated" suena igual que "Top
Rated" y compra 50× menos), así que el congelado recupera parcial y el fine-tuning debería
comprar la diferencia justo en *_sin.

Infra: --text-emb/--text-emb-finetune/--text-emb-lr con guards, temb_proj en BTRTransformer
(hf_encoder colgado DESPUÉS de _init_weights para no pisar los pesos preentrenados), lr por
grupo en AdamW, --drop-features all, pad_token_id==0 verificado, ckpt de ft no se guarda
(~90MB, exploratorio). Bug corregido de paso: drop_feature_columns con lista vacía
(torch.tensor([]) es float → index_select explotaba). Panel: clave canónica 51 campos
(espejo verificado, 249 claves OK), sección 5b con el bloque "preentrenado externo".
transformers agregado a requirements.txt. Smokes CPU: los 5 caminos corren.

## 23/08 — 8ª tanda corrida y analizada: congelado resta, fine-tuneado repara

Matias corrió las 42 (838 totales, 116/116). Análisis apareado en analisis.md §15.1, figura
graficos/bert_transfer.png. Contra las hipótesis: (1) anti-semántica confirmada con MÁS fuerza
— bert_solo 0.567 pierde hasta con la logística cruda (0.660), porque MiniLM acerca lo que acá
hay que separar; (2) bert_solo_intr 0.128 ≈ azar — quinta vindicación del EDA, con un lector
de 22M params; (3) REFUTADA la neutralidad: el token congelado RESTA (−0.073 0/6; bert_mlp
−0.051 0/6) — contraste con tl_emb_mlp (−0.006): features congeladas del mismo dominio eran
inofensivas, las ajenas son ruido activo; los 4 grupos congelados repiten gap val-test ~0.04;
(4) recuperación parcial sin regex (0.667 desde piso 0.162); (5) el fine-tuning repara — el
efecto más consistente de la tanda: +0.060 (6/6) con status, +0.132 (6/6) sin; el sobreajuste
temido no apareció (lr 1e-5, gap 0.011); (6) nada supera 0.824 — selección intacta. Hallazgo
positivo: bert_ft_sin 0.798 es la mejor lectura del status desde texto crudo del proyecto
(> tower 0.775 > hybrid 0.735) — preentrenamiento + fine-tuning le gana a entrenar de cero,
medido; pero extraer la señal limpia (regex) sigue valiendo más que 22M de parámetros.

Sync: panel 9.3MB con 838 corridas (paridad 256 claves OK) republicado en el artifact
canónico (ojo: un republish sin url creó un duplicado accidental — ignorar/borrar de la
galería; el link bueno sigue siendo el de siempre); zoo byte-idéntico (sin cambios); CSV 140
grupos; entrega actualizada (análisis 838, figuras bert_transfer + resumen, guión: bullet
nuevo "¿y con un preentrenado de verdad?" + corridas 838). Capítulo experimental CERRADO
(ahora sí, con el arco de transfer completo).

## 24/08 — Plan de Juan: flujo de experimentos por eje → mapa de decisiones + figura

Juan propuso ordenar los experimentos como flujo lógico de decisiones sobre la base ganadora
(cabezas, bloques, encodings, d_model, algoritmos de embedding) y "fijarse si ya no hay algún
experimento de este estilo ya hecho". Lo hay: TODOS — es la historia de las tandas 1–3 (más
5–8 como exploración). Lo que faltaba era el soporte visual por decisión para la presentación
(los gráficos eran globales). Hecho:

- eda/grafico_decisiones.py → graficos/decisiones_por_eje.png: 6 paneles "una decisión por
  eje" (formulación / encoding / cabezas / bloques / d_model / pre-entrenamiento), opción
  elegida en teal, alternativas violeta, contexto de otra base en gris. Números del CSV.
- entrega/presentacion/mapa_decisiones.md: el índice eje-por-eje (por qué se probó, configs,
  números, decisión, § de analisis.md) que implementa el plan de Juan sobre lo ya corrido.
  Incluye el matiz metodológico lindo: cabezas INTERACTÚA con encoding (emb: h1 gana; ordinal:
  h4 gana) → por eso se decide por val sobre la base final.
- guion.md diapo 12: referencia a la figura como paraguas de las diapos 11–14.

Cero GPU. El guión ya era experimento-por-experimento (diapos 11–14); esto lo hace defendible
pregunta por pregunta.

## 25/08 — La ppt ahora va experimento por experimento (fix al reclamo de Fer)

El 24/08 armé la figura y el mapa pero NO toqué la ppt — Fer lo marcó (con razón). Hecho:
generar.py reescrito → 24 diapositivas. La vieja diapo única "capacidad y entrenamiento" se
partió en TRES, cada una con su gráfico propio embebido: 12 Cabezas (decision_cabezas.png, con
la interacción encoding×cabezas como punto metodológico), 13 Bloques (decision_bloques.png),
14 d_model (decision_dmodel.png, con min_d16l1 y la destilación como epílogo). La 15 (encoding,
el decisivo) cambió la tabla por decision_encoding.png. Diapo NUEVA 16: Experimento 6 · Inits
(decision_init.png: MLM/w2v/AE — init aleatoria gana; MiniLM al backup). El texto pasa a ser
Experimento 7 (diapo 17). eda/grafico_decisiones.py ahora emite también los 6 PNG individuales
tamaño-diapo (singles()). Portada actualizada: 838 corridas y CUARTO nombre Juan Bautista
Albertoni Salini (los 4 siguen a confirmar por Fer). Guión renumerado completo (24 secciones,
~30 min — al límite del rango: ensayar y recortar 16/17B/20 si aprieta), referencias cruzadas
corregidas. Verificado con screenshots headless (diapos 12/14/15/16, claro).

## 26/08 — ¿query_id es un template re-ejecutado? No: nada se repite jamás (chequeo de Fer)

Pregunta de Fer: si para una misma query apareciera el mismo producto más de una vez, se
confirmaría que query_id es un *template* (una combinación exacta de filtros ejecutada muchas
veces a lo largo del tiempo) — y eso les daría una lectura física a los timestamps intra-query
de 2 años. Si no, gana la sospecha de que los timestamps están directamente mal.

Chequeo agregado a eda/verificaciones.py (sección 2.6b). No hay product_id → identidad =
las 14 columnas catalográficas. Resultado: **10.000 productos únicos en 10.000 filas** — cero
repeticiones dentro de una query y cero en todo el dataset. El único título repetido dentro de
una query (q_000239) es una colisión del generador de nombres: dos productos con precio, peso y
dimensiones distintos. Y el mejor escenario para el template también falla: 26 combos exactos de
filtros se repiten (55 queries, el precio del filtro viene de una grilla — 393 mínimos
distintos), y esas queries comparten **0 productos y 0 títulos** entre sí.

Veredicto: no existe catálogo persistente → query_id NO es un template re-ejecutado → los spans
intra-query de ~2 años no tienen ninguna lectura física posible → timestamp = ruido del
generador. Cierra la dicotomía que propuesta §2.6 dejaba abierta (bullet actualizado ahí, más
la pregunta a la cátedra de §11 afilada, más una oración en guion.md §4). Sexta vindicación del
EDA: el descarte de timestamp ya no es "sospecha fuerte" sino conclusión demostrada.
