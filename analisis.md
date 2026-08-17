# Análisis de las tandas GPU (16/08)

Primera tanda corrida por Matias en la RTX 3070: las 24 configs de la suite con el protocolo
original (paciencia 8, tope 60 épocas) **y** una variante `pac20_` (paciencia 20, tope 300) para
cada una, seeds 42–47. Total: **288 corridas, 48 grupos × 6 seeds**. Las métricas finales de las
285 corridas con schema viejo se recalcularon desde los checkpoints sin reentrenar
(`eda/recalcula_metricas.py`), así que todos los grupos tienen las 16 métricas en val y test.

Exploración interactiva de todo: [el Laboratorio (panel.html)](panel.html) — curvas por época,
las 16 métricas y ranking clickeable. Vista global:

![Resumen PR-AUC](graficos/resumen_prauc.png)

Convención de este documento: **PR-AUC de test, media ± desvío sobre 6 seeds**; las
comparaciones clave se reportan **apareadas por seed** (misma seed = mismo split, así que el
delta por seed elimina la varianza del split; "gana x/6" = en cuántas seeds el delta es
positivo). Azar = 0.131 (tasa base). Referencia sin red: GBM 0.762, logística 0.660.

## 1. Los resultados principales

| grupo | PR test | qué es |
|---|---|---|
| **pac20_feat_h1** | **0.816 ± 0.026** | el campeón: tabular, 1 cabeza, paciencia 20 |
| pac20_feat_d64 | 0.815 ± 0.027 | d_model 64, paciencia 20 |
| pac20_feat_l4 | 0.803 ± 0.018 | 4 bloques, paciencia 20 |
| pac20_feat_base | 0.798 ± 0.036 | la config base con paciencia 20 |
| feat_base | 0.794 ± 0.033 | la config base (paciencia 8) |
| tower_base | 0.775 ± 0.022 | la mejor de las que ven texto |
| pac20_mlp_base | 0.750 ± 0.033 | el control sin atención |
| pac20_listwise_base | 0.740 ± 0.036 | productos como tokens |
| pac20_hybrid_full | 0.735 ± 0.031 | features + chars en una secuencia |
| text_base | 0.652 ± 0.039 | solo caracteres |
| feat_intrinseco | 0.159 ± 0.014 | sin estado (producto nuevo) ≈ techo GBM 0.162 |
| feat_causal | 0.124 ± 0.006 | degenerado (ver §2.7) |

El transformer tabular **supera al GBM con claridad** (0.816 vs 0.762) — en la medición
preliminar de CPU la ventaja era marginal (0.766); con 6 seeds y el protocolo con más paciencia
quedó establecida.

## 2. Lecturas, con su evidencia

### 2.1 La atención aporta, y no es cuestión de tamaño
transformer vs MLP con la misma entrada, apareado: **Δ = +0.048 ± 0.038, gana 5/6 seeds** —
con el MLP teniendo 4,5× más parámetros (126k vs 28k). Es la respuesta a la pregunta central
del TP.

### 2.2 Paciencia 20 gana en tabular, pierde en texto puro
`pac20_*` mejora o empata el test en **21/24** configs (mayores saltos: listwise +0.041,
feat_h1 +0.037). Pero `text_base` **empeora** (−0.019, gana 1/6) y `text_d64` también (−0.021):
con más paciencia, la selección del mejor checkpoint por val PR-AUC sobreajusta a la validación
en la familia con más varianza. **Decisión**: protocolo paciencia 20 / tope 300 para las
tabulares nuevas; texto queda en paciencia 8. Ninguna corrida tocó el tope de 300 épocas.

### 2.3 Menos cabezas, mejor (a este tamaño)
h1 vs h4 (con pac20): **+0.018 ± 0.028, gana 4/6**. Con d=32, una cabeza de dimensión 32 le
gana a 4 de dimensión 8: la señal dominante es una sola (el tier de estado), y una consulta
"rica" parece valer más que 4 consultas chicas. d64 vs d32: +0.017 (4/6), mismo signo. La
grilla campeón de la 2ª tanda combina estos ganadores (¿h1 y d64 son aditivos o redundantes?).

### 2.4 Los que pierden: mean pooling, pos_weight, bins, PE
- **CLS > mean**: −0.034, mean gana **0/6** — el token de agregación dedicado es mejor que
  promediar (consistente con BERT).
- **pos_weight daña**: −0.059 (gana 1/6). PR-AUC es una métrica de ranking: re-pesar la clase
  positiva no ayuda a ordenar y distorsiona las probabilidades. Se descarta.
- **bins < lineal**: −0.023 (gana 1/6). El FFN ya captura la U invertida del precio; discretizar
  pierde resolución. La hipótesis de §6.2 de la propuesta quedó refutada empíricamente — para
  la presentación es un lindo "lo pensamos, lo medimos, no era".
- **PE en features**: Δ ≈ 0 (−0.004, 3/6) — como predice la teoría: un conjunto de features no
  tiene orden que codificar.

### 2.5 ¿El transformer redescubre la señal del texto? Sí — pero depende de dónde esté la atención
La pregunta de diseño más linda del TP, y la primera tanda la respondió con un contraste limpio:

- **hybrid**: sacarle el token parseado `listing_status` **no le cuesta nada**
  (sin_regex vs full: +0.006 / +0.0005 con pac20) → la atención cruzada recupera desde los
  caracteres crudos lo que nosotros extrajimos con una regex.
- **tower**: sacárselo **le cuesta −0.04/−0.05** (gana 1/6) → cuando el texto tiene que pasar
  por UN embedding de 32 dims antes de encontrarse con lo tabular, la señal no sobrevive
  entera. El cuello de botella es real.

Al mismo tiempo, **ninguna variante con texto le gana al tabular puro** (tower 0.775 < feat
0.794; hybrid 0.735; hybrid vs feat apareado: −0.063, gana 1/6). Con el estado ya parseado como
feature, el texto no agrega — solo diluye (256 tokens de chars compitiendo con 13 tabulares por
la atención). El texto importa en el mundo donde NO parseamos: `text_base` 0.652 está muy por
encima del techo intrínseco (0.16) — el modelo de chars encontró el sufijo solo — pero muy por
debajo del tabular: leer texto crudo con un modelo de 36k parámetros cuesta.

### 2.6 La familia intrínseca confirma el techo
`feat_intrinseco` 0.159 ≈ GBM sin estado 0.162: en el mundo "producto nuevo" nadie puede pasar
de ~0.16 (el 61% de las filas cae en tiers de BTR exactamente cero — no hay señal que aprender).
`text_intrinseco` 0.142 es el MÁS bajo de la familia: el `--strip-status` funcionó — no quedó
ninguna puerta trasera de popularidad en el texto. Las dos familias de la propuesta §2.3.1
quedan medidas de punta a punta.

### 2.7 `feat_causal` no es "peor": está roto por diseño (y es una gran historia)
ROC exactamente 0.500, PR 0.124 = tasa base. Verificado sobre el checkpoint: **predice una
constante** (p = 0.2214 para todo test, desvío 3·10⁻⁸). La causa: con máscara causal, el CLS en
la **posición 0** solo puede atenderse a sí mismo — clasifica sin ver ningún feature. La
moraleja de arquitectura: los decoders (GPT) leen la secuencia desde el **último** token; los
encoders (BERT) pueden poner el CLS adelante porque la atención es bidireccional. La ablación
"¿importa la bidireccionalidad?" recién se puede responder con el CLS al final:
`feat_causal_last` (2ª tanda, ya implementado con `--cls-position last`).

### 2.8 Selección y gap val→test
El gap val−test es ≈ +0.02/+0.03 en casi todos los grupos (sesgo normal de selección por val).
En texto con pac20 sube a ≈ +0.05–0.06 — otra cara del sobreajuste de selección de §2.2.

### 2.9 Calibración (propuesta #3 de Junior): estamos bien
Sobre checkpoints ya entrenados (`eda/calibracion.py`, reliability + ECE + temperature scaling
ajustado en val):

| checkpoint | T | ECE antes → después | BTR real vs predicho |
|---|---|---|---|
| pac20_feat_h1 (mejor val) | 0.91 | 0.012 → 0.014 | 0.123 vs 0.132 |
| feat_base seed42 | 1.10 | 0.016 → 0.014 | 0.134 vs 0.142 |

![Calibración](graficos/calibracion.png)

Lectura: **no hay descalibración grave** (ECE ~0.01, T ≈ 1) — el promedio de p por producto es
un estimador razonable del BTR, con una sobreestimación global de ~+0.9 puntos concentrada en
los deciles altos (el modelo es levemente sobreconfiado con los "ganadores"). Temperature
scaling no mueve la aguja: no hace falta. Queda respondido el "requisito silencioso" que Junior
señaló, con evidencia.

## 3. Decisiones que quedan fijadas

1. **Protocolo**: 6 seeds; tabular con paciencia 20 / tope 300; texto con paciencia 8.
2. **Se descartan**: pos_weight, mean pooling, bins global, PE en features, causal con CLS
   adelante (degenerado).
3. **Campeón provisorio**: `pac20_feat_h1` (0.816). La grilla campeón de la 2ª tanda decide si
   d64/l4 suman sobre h1.
4. Con estado parseado, **el texto no agrega** — la historia del TP es el contraste
   hybrid-recupera / tower-no (2.5), no "texto mejora el número".

## 4. La segunda tanda (ya en la suite; 15 configs nuevas)

Diseñada desde este análisis + los pedidos de Fer (encodings, features descartados) + las
propuestas de Junior (1 y 2; la 3 ya está respondida en §2.9). Todo implementado y smoke-testeado;
`experimentos.py` las corre con las mismas dos líneas de siempre (resume automático: solo corre
lo nuevo).

| config | pregunta que responde |
|---|---|
| `camp_d64h1`, `camp_d64l4`, `camp_h1l4`, `camp_d64h1l4` | ¿los ganadores individuales (d64, h1, l4) suman o se pisan? |
| `feat_causal_last` | causal bien hecho (CLS al final): ¿importa la bidireccionalidad? |
| `feat_target` / `feat_freq` / `feat_hash8` | encodings de categóricas: target suavizado / frecuencia / hashing-módulo (el "modular") |
| `mlp_onehot` | one-hot crudo al MLP: ¿cuánto aporta embeber? (en transformer one-hot ≡ embedding, §6.1) |
| `feat_extras` | volver a meter volumen/package/ingredientes: ¿el EDA tenía razón en descartarlos? |
| `feat_cartaux01/03/05` | multi-task con `cart` como label auxiliar (Junior #2), barrido de λ |
| `listwise_texto` | Junior #1: ¿listwise pierde por la idea o por no ver el texto? (torre de chars dentro del token de producto) |
| `text_len96` | truncar a 96 chars (el título entero, donde vive la señal): atención 7× más barata, ¿mismo resultado? |
| `hybrid_status_campo`, `tower_status_campo` | idea de Fer: el estado SOLO como campo, texto limpio — la celda que faltaba del 2×2 (§4.1) |
| `feat_ordinal`, `feat_status_ordinal`, `feat_status_target` | idea de Fer: encoding del estado CON ORDEN — rango (ordinal) vs magnitud (target), global y solo para `listing_status` (§4.1) |

### 4.1 El estado como campo separado (idea de Fer)

Dos experimentos con una misma motivación: "separar el Best Seller del título y tratarlo como un
campo, con un encoding que tenga sentido, incluso con orden".

**(a) El 2×2 completo.** Con las corridas de la 1ª tanda, hybrid y tower tienen medidas 3 de las
4 celdas de {token parseado sí/no} × {sufijo en el texto sí/no}: *full* (ambos canales,
redundante), *sin_regex* (solo texto) e *intrinseco* (ninguno). Faltaba exactamente la
separación prolija que propone Fer: **texto limpio + estado solo como campo**
(`--strip-status` sin drop). Si `status_campo` > `full`, el sufijo dentro del texto era ruido
puro que diluía la atención; si < `full`, los chars aportaban algo más que el token parseado.

**(b) El orden del campo.** Un orden semántico a mano es indefendible acá — el EDA (§2.3) mostró
que el wording NO predice el tier ("Highly Rated" suena igual que "Top Rated" y compra 50×
menos). Los órdenes defendibles se derivan del BTR de train: **ordinal** = solo el rango del
nivel (normalizado a [0,1]), **target** = rango + magnitud (media suavizada, m=50). Como los
tiers tienen saltos enormes de magnitud (0.65 / 0.03 / 0.000), la hipótesis es
target ≥ ordinal; y el embedding aprendido puede representar cualquiera de los dos, así que
embedding ≥ target es lo esperable — el valor del experimento es medir cuánto se pierde al
comprimir 21 niveles × 32 dims en UN escalar, y la eficiencia de parámetros. Implementado como
encoding **por feature** (`--cat-feature-encoding listing_status=ordinal`): el resto de las
categóricas queda en embedding, así el efecto se aísla.

**En la máquina de la 3070** (tras `git pull`): las dos líneas de siempre —

```bash
.venv/bin/python experimentos.py            # corre SOLO lo nuevo (~20 configs × 6 seeds)
.venv/bin/python experimentos.py --resumen
```

y al terminar: `python panel.py` (re-embebe resultados), commit de `resultados/ pesos/
panel.html` y push. Estimado: las 13 tabulares son segundos por corrida; `listwise_texto` y
`text_len96` son la parte con texto (la más pesada es listwise_texto; `text_len96` es ~7× más
liviana que `text_base`).

## 5. Segunda tanda: resultados (116 corridas)

18 configs × 6 seeds + `listwise_texto` × 2 (la pesada; los 4 seeds restantes corren en la 3ª
tanda por resume automático). Todas con las 16 métricas. Lo importante, apareado por seed:

### 5.1 Campeón nuevo: `feat_ordinal` 0.824 ± 0.018 — la idea del orden, aplicada globalmente
La hipótesis de §4.1 (embedding ≥ target ≥ ordinal) quedó **invertida**: ordinal global
(TODAS las categóricas como su rango de BTR de train) da **0.8239 ± 0.018**, arriba de target
global (0.8134), del mejor embedding (`camp_d64h1l4` 0.8178) y de `pac20_feat_h1` (0.8157) —
además con el desvío más chico. Lectura: el rango inyecta como *prior* el orden que el embedding
tendría que aprender, con muchísimos menos parámetros → menos overfit; y los rangos
equiespaciados en [0,1] están mejor condicionados que las magnitudes de target (0.65/0.03/0.000,
apelmazadas en los extremos). Las versiones solo-status (`feat_status_ordinal/target`) son ≈
neutras: la ganancia viene de simplificar TODAS las categóricas, no solo el estado.
La 3ª tanda combina ordinal con los ganadores de capacidad (`camp_ordinal_*`).

### 5.2 Los contraejemplos de encoding, según lo esperado
`feat_freq` **0.218** (la frecuencia no correlaciona con el BTR: destruye la señal) y
`feat_hash8` **0.498** (las colisiones del módulo mezclan tiers). Moraleja presentable: el
encoding debe *preservar la relación nivel→propensión*; freq y hashing la rompen a propósito.

### 5.3 one-hot le gana al embedding en el MLP (+0.047, 6/6) — matiz honesto al "atención aporta"
`mlp_onehot` 0.797 ≈ `pac20_feat_base` 0.798. El déficit del MLP era en buena parte su
**entrada** (los 416 dims de embeddings entrelazados sobreajustan; el one-hot ralo deja que la
primera capa aprenda pesos por nivel, casi logístico). El enunciado fino pasa a ser: con la misma
representación, la atención aporta +0.048 (5/6); contra el MEJOR MLP posible, el mejor
transformer aporta ~+0.027 (0.824 vs 0.797). La atención sigue ganando, con la vara más alta.

### 5.4 El 2×2 del estado (idea de Fer): el canal doble era ruido para el híbrido
`hybrid_status_campo` (texto limpio + estado como campo) **0.733 > hybrid_full 0.705**
(+0.027, 4/6): el sufijo duplicado adentro del texto diluía. En tower no cambia nada (−0.003)
— su cuello de botella ya aislaba el texto. Y aun limpio, el híbrido queda −0.065 (0/6) bajo el
tabular puro: el texto no suma cuando el estado ya está parseado. El 2×2 queda completo:
full 0.705 · sin_regex 0.711 · **status_campo 0.733** · intrinseco 0.150.

### 5.5 Las propuestas de Junior, medidas
- **listwise_texto (#1)**: +0.016 en 2/2 seeds (0.753 ± 0.005 vs 0.740). El texto ayuda a
  listwise de forma consistente → estaba parcialmente ciego, no mal concebido. Pero sigue ~0.05
  abajo del tabular: la competencia de página es débil (EDA §2.5, re-confirmado).
- **cart-aux (#2)**: U invertida sobre λ (0.1: −0.003 · **0.3: +0.011** · 0.5: −0.007), todo
  dentro del ruido (3/6). Veredicto: no ayuda significativamente — `cart` es casi la misma señal
  que `bought` (bought ⟹ cart), no agrega información nueva al encoder.

### 5.6 Otros
- **Bidireccionalidad: no importa** — `feat_causal_last` 0.795 ≈ base (−0.003, 3/6). El arco
  completo del causal: catástrofe (1ª tanda) → diagnóstico (CLS en pos 0) → fix (CLS al final)
  → respuesta: *da igual*, con 13 tokens de features el orden de lectura no aporta ni quita.
- **Grilla campeón**: los ganadores individuales NO son aditivos (d64+h1 0.800 < h1 solo 0.816);
  solo el combo completo `camp_d64h1l4` empata arriba (0.8178, +0.002, 4/6). Meseta en ~0.82.
- **feat_extras: Δ=−0.006 (3/6)** — volumen/package/ingredientes no aportan: el descarte del
  EDA queda verificado empíricamente, como pidió Fer.
- **text_len96: −0.027 (2/6)** — truncar al título EMPEORA pese a que la señal vive ahí: la
  última oración de la descripción (la copia redundante del estado) ayudaba al modelo chico.
  La redundancia es amiga de los modelos de 36k parámetros.

## 6. Revisión externa (16/08): qué confirma, qué corrige, qué faltaba

Fer trajo la lectura de un agente externo que analizó el enunciado y las clases sin ver nuestro
código. Veredicto punto por punto:

- **Ya lo teníamos** (y ahora con números): FT-Transformer (= formulación A), el menú completo
  de encodings (§5.1–5.3), CLS + su ablación, las tres trampas (cart, split por query, PR-AUC),
  su "5a" (= feat) y su "5c" (la de dos niveles = nuestro listwise; su versión con encoder de
  ítem que incluye texto = exactamente `listwise_texto`, la propuesta #1 de Junior).
- **Corrección conceptual que adoptamos** (va a la presentación): la atención no requiere tokens
  "del mismo tipo", requiere que vivan en el **mismo espacio ℝ^d** — el FeatureTokenizer es lo
  que los proyecta ahí, igual que el embedding en un LLM o los parches en un multimodal.
- **Conexión que no habíamos escrito**: el "column embedding" de TabTransformer (vector
  identificador por columna) es exactamente nuestro `feat_pos` — y su Δ ≈ 0 **confirma
  empíricamente** que en FT-Transformer la identidad de columna ya viene implícita en los
  parámetros propios de cada feature.
- **Su sugerencia de split temporal**: rechazada con evidencia — los timestamps están rotos
  (spans de 2 años dentro de una misma query, EDA §2.6). Sí adoptamos verificar hora/día como
  extras (`feat_tiempo`, 3ª tanda): el EDA dice ruido; que lo diga también un modelo.
- **Lo que sí nos faltaba — su "5b", implementado en la 3ª tanda**:
  1. **Tokenización word-level** (`--text-tokens words`): la clase recomendaba palabras; nuestra
     torre era char-level (la demo). Con palabras, "(Best Seller)" son 2 tokens en vez de 13
     caracteres, y la secuencia baja de 257 a 64 (`text_words`).
  2. **El resumen del texto como UN token del transformer tabular** (`--formulation fusion`):
     la torre comprime el texto a su CLS y ese vector entra como token 15 de la secuencia de
     features — la atención cruza texto↔features al nivel del resumen, **sin que 256 chars
     diluyan** (el mal medido del híbrido, §5.4). Es el punto medio exacto entre hybrid (cruce
     total, diluido) y tower (sin cruce, cuello de botella).
  3. **word2vec pre-entrenado vs end-to-end** (`--w2v-init`): skipgram con negative sampling
     sobre el corpus de train como inicialización del encoder de palabras — la comparación
     clase 1 (embeddings no supervisados) vs clase 2 (end-to-end) que pedía la materia.
- **Su "MLM sobre features"** (enmascarar una columna y predecirla como pre-training): anotado
  como opcional en el panel; el propio agente advierte el scope, y la conexión "pre-entrenar →
  ajustar" ya queda cubierta por w2v-init. Si sobra tiempo, se implementa.
- **Mapas de atención** (insistió, y tenía razón): hechos — `eda/atencion.py`:

![Atención del campeón](graficos/atencion_pac20_feat_h1_features_d32_h1_l2_linear_seed45.png)

**La capa 1 rutea las dos señales del EDA**: el CLS pone el 51% de su atención en `status`
(columna encendida: casi todos los tokens lo consultan) y la familia de precio se consulta entre
sí (la columna `p_rel` brilla para `price` y `f_min` — la señal relacional "¿dónde caigo en el
rango pedido?"). La capa 2 mezcla en forma pareja. Es el gráfico de interpretabilidad del TP:
el modelo mira exactamente donde el EDA dijo que había que mirar.

## 7. Tercera tanda (ya en la suite)

| config | pregunta |
|---|---|
| `camp_ordinal_h1` / `_l4` / `_d64h1l4` | ¿el campeón ordinal se combina con los ganadores de capacidad? |
| `feat_tiempo` | hora/día del timestamp como features: ¿el EDA tenía razón en que es ruido? |
| `text_words` | tokenización word-level (64 tokens) vs chars (257): la que "recomendaron" |
| `fusion_base` / `fusion_words` | el resumen del texto como token 15: ¿arregla la dilución del híbrido? |
| `fusion_words_w2v` | embeddings skipgram pre-entrenados vs end-to-end (clase 1 vs clase 2) |
| (`listwise_texto` seeds 44–47) | completa los 6 seeds por resume automático |

## 8. Pendientes analíticos (con los checkpoints ya disponibles)

- ~~Mapas de atención del campeón~~ → hechos (§6).
- Métricas por página (top-1 de la query, NDCG) — eje "pedir" en el panel.
- GroupKFold si queremos intervalos más finos para la presentación.
