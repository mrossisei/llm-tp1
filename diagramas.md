# Diagramas de las arquitecturas

Referencia visual de las **seis arquitecturas** implementadas en `btr/model.py`, completas: de la
fila del CSV a `p(bought)`. Las dimensiones son las reales del código con la config base
(`d_model=32`, 4 cabezas de dim 8, 2 bloques). GitHub renderiza los diagramas solo; las
justificaciones de cada decisión están en `propuesta.md` (§4 formulaciones, §5 arquitectura) y la
suite que las corre a todas es `experimentos.py`.

Convención: `(B, 14, 32)` = (batch, cantidad de tokens, d_model). Todas comparten el
preprocesamiento (panel 0) y el protocolo (split por query 70/15/15, AdamW 1e-3, early stopping
por PR-AUC de validación, 3 seeds).

## 0. Preprocesamiento compartido (`btr/data.py`)

```mermaid
flowchart TD
    CSV["supermarket_products.csv<br/>10.000 impresiones (fila = producto mostrado en una búsqueda) × 22 columnas<br/>2.012 búsquedas de 1 a 8 productos · target: bought (13% positivos)"]
    DER["Derivar features (sin mirar el target)<br/>listing_status ← regex al sufijo '( … )' del título · 21 niveles<br/>price_rel ← (price − filter_min) / (filter_max − filter_min)<br/>allergens: NaN → 'None'"]
    SPL["Split por query_id — 70 / 15 / 15<br/>una misma búsqueda nunca cruza train / val / test"]
    FIT["Preprocessor — ajustado SOLO con train<br/>7 vocabularios categóricos (índice 0 = UNK)<br/>z-score en las 6 numéricas (log1p antes en price y net_weight_oz)<br/>texto: title + salto + description → vocabulario de 67 chars (PAD=0, UNK=1)"]
    OUT["x_cat (B, 7) · x_num (B, 6) · x_text (B, 256) · y (B,)"]
    CSV --> DER --> SPL --> FIT --> OUT
```

Quedan **afuera de la entrada**: `cart` (leakage: bought ⟹ cart al 100%), `query_id` (solo
particiona), `timestamp` (ruido), `package_size` y `dimensions_in` (redundantes con el peso),
`filter_category` y `filter_storage_type` (los productos siempre los cumplen; el rango de precio
sí entra vía `price_rel`), `ingredients` (v1).

## A. Transformer tabular — cada feature es un token

`--arch transformer --formulation features` · 14 tokens · 28.289 parámetros · estilo FT-Transformer

```mermaid
flowchart TD
    IN["1 impresión: x_cat (7 índices) + x_num (6 valores z-score)"]
    FT["FeatureTokenizer<br/>cada categórica → su PROPIA tabla nn.Embedding(cardᵢ, 32)<br/>cada numérica → xⱼ·wⱼ + bⱼ aprendidos — o bins por cuantiles (--numeric-mode bins)"]
    SEQ["secuencia: CLS + 13 tokens → (B, 14, 32)<br/>sin positional (un set de features no tiene orden) · sin máscara (todos reales)"]
    BLK["Bloque Transformer ×2<br/>pre-LN · atención multi-cabeza (4 cabezas de dim 8) · FFN 32→128→32 · residuales"]
    CLS["LayerNorm final → quedarse con el token CLS → (B, 32)"]
    HEAD["Linear 32 → 1"]
    OUT(["σ → p(bought) · loss: BCE (pos_weight opcional)"])
    IN --> FT --> SEQ --> BLK --> CLS --> HEAD --> OUT
```

Responde: **¿la atención entre features aporta?** (interacciones tipo price_rel × categoría).
Primer resultado (seed 42): PR-AUC test **0.766**.

## Baseline MLP — mismos embeddings, sin atención

`--arch mlp` · 126.209 parámetros (4,5× el transformer tabular)

```mermaid
flowchart TD
    IN["1 impresión: x_cat (7) + x_num (6)"]
    FT["FeatureTokenizer — idéntico al de A<br/>lo único que cambia es qué mezcla los tokens"]
    FLAT["flatten: 13 × 32 → (B, 416)"]
    MLP["MLP denso (como en SIA)<br/>Linear 416→256 → ReLU → Dropout<br/>Linear 256→64 → ReLU → Dropout<br/>Linear 64→1"]
    OUT(["σ → p(bought)"])
    IN --> FT --> FLAT --> MLP --> OUT
```

Es el **control** del experimento: si el transformer no le gana, la atención no se justifica.
Primer resultado: 0.715 < 0.766 pese a tener 4,5× más parámetros.

## C. Transformer de texto — cada carácter es un token (la demo, adaptada)

`--arch transformer --formulation text` · 257 tokens · 35.713 parámetros

```mermaid
flowchart TD
    TXT["title + salto + description — texto crudo, sin parsear<br/>'Riverbend Cultured Half And Half - 24 oz (Best Seller)'<br/>p95 = 243 chars → truncado a 256"]
    ENC["cada carácter → su índice en el vocabulario de 67<br/>UNK=1 para chars no vistos · PAD=0 al final"]
    EMB["CLS + char embedding (67×32) → (B, 257, 32)<br/>+ embedding posicional aprendido (257×32): en texto el ORDEN importa<br/>máscara: los PAD no reciben atención (score −1e9)"]
    BLK["Bloque Transformer ×2<br/>atención 257×257 por cabeza — acá el cómputo es real (GPU)"]
    CLS["LayerNorm → token CLS → (B, 32)"]
    HEAD["Linear 32 → 1"]
    OUT(["σ → p(bought)"])
    TXT --> ENC --> EMB --> BLK --> CLS --> HEAD --> OUT
```

Es la adaptación directa de la demo de la cátedra: de *decoder que genera el próximo carácter*
(máscara causal, 65 logits por posición) a *encoder que clasifica* (bidireccional, CLS, 1 logit).
Nadie le parsea nada: tiene que descubrir sola que la señal vive en el sufijo del título y en la
última oración de la descripción.

## A+C. Transformer híbrido — features y caracteres en una misma secuencia

`--arch transformer --formulation hybrid` · 270 tokens · 39.073 parámetros

```mermaid
flowchart TD
    TAB["x_cat (7) + x_num (6)"]
    TXT["256 índices de caracteres"]
    TAB -->|"FeatureTokenizer → 13 tokens"| SEQ
    TXT -->|"char embedding → 256 tokens"| SEQ
    SEQ["UNA sola secuencia: CLS + 13 + 256 = 270 tokens<br/>+ PE (270×32) sobre toda la secuencia · máscara solo en los PAD del texto"]
    BLK["Bloque Transformer ×2<br/>la atención puede CRUZAR texto ↔ features en la misma capa<br/>p. ej. el token price_rel puede atender a los chars de '(Best Seller)'"]
    CLS["LayerNorm → token CLS → (B, 32)"]
    HEAD["Linear 32 → 1"]
    OUT(["σ → p(bought)"])
    SEQ --> BLK --> CLS --> HEAD --> OUT
```

La variante `hybrid_sin_regex` saca el token `listing_status` parseado y pregunta: ¿la atención
recupera sola desde los caracteres lo que extrajimos con la regex?

## C2. Torre de texto + MLP — el transformer solo hace embeddings

`--arch tower` · 257 + 13 tokens · 96.225 parámetros

```mermaid
flowchart TD
    TXT["256 índices de caracteres"]
    TORRE["Torre de texto<br/>CLS + chars + PE → Bloques ×2 (atención SOLO entre caracteres)<br/>LayerNorm → CLS = embedding del texto (B, 32)"]
    TAB["x_cat (7) + x_num (6)"]
    FT["FeatureTokenizer → 13 tokens → flatten (B, 416)"]
    CAT["⊕ concatenar → (B, 448)"]
    MLP["el CLASIFICADOR es un MLP<br/>Linear 448→128 → ReLU → Dropout · Linear 128→1"]
    OUT(["σ → p(bought)"])
    TXT --> TORRE --> CAT
    TAB --> FT --> CAT
    CAT --> MLP --> OUT
```

La diferencia con el híbrido es exactamente una: acá texto y tabular **se encuentran recién
después de la atención** (en el concat), así que la atención nunca puede mirar un feature. Si el
híbrido gana, el cruce dentro de la atención vale; si empatan, alcanza con el transformer como
módulo de embedding (estilo BERT, clase 2).

## B. Transformer de página (listwise) — cada producto de la query es un token

`--arch listwise` · 8 tokens (el batch es en búsquedas) · 41.601 parámetros

```mermaid
flowchart TD
    Q["UNA búsqueda completa: los 1–8 productos que compiten en la misma página"]
    COL["Colapsar cada producto a UN token<br/>FeatureTokenizer → 13 tokens → flatten (416) → Linear 416→32"]
    SEQ["secuencia: 8 tokens producto + prod_mask sobre los slots vacíos → (Q, 8, 32)<br/>sin positional: el CSV no trae orden de página"]
    BLK["Bloque Transformer ×2<br/>la atención corre ENTRE los productos de la página:<br/>'¿me compran a MÍ, dado lo que aparece al lado?'"]
    HEAD["LayerNorm → Linear 32→1 aplicado a CADA producto → 8 logits"]
    OUT(["σ por producto → BCE SOLO sobre los slots reales"])
    Q --> COL --> SEQ --> BLK --> HEAD --> OUT
```

La única formulación que ve la página completa y puede capturar competencia. Primer resultado:
0.664 — consistente con el EDA (§2.5: la competencia acá es débil).

## Eje transversal: dos familias (no son arquitecturas nuevas)

Sobre casi todas las de arriba corren **dos variantes de entrada** que resuelven la discusión del
"(Best Seller)" (`propuesta.md §2.3.1`):

| familia | qué usa | flags | simula |
|---|---|---|---|
| **catálogo** | todo, incluido el estado del listing | (ninguno) | rankear el catálogo actual |
| **intrínseco** | recorta el estado: el token parseado Y su copia en el texto | `--drop-features listing_status --strip-status` | el producto nuevo, sin historial |

El techo medido con GBM anticipa la diferencia: PR-AUC 0.762 con estado vs 0.162 sin estado.
En la suite: `feat_intrinseco`, `text_intrinseco`, `hybrid_intrinseco`.

## Las seis, lado a lado

| arquitectura | un token es… | secuencia | la atención cruza | parámetros | PR-AUC test (seed 42) | comando |
|---|---|---|---|---|---|---|
| A · tabular | un feature | 14 | features ↔ features | 28.289 | 0.766 | `--formulation features` |
| MLP (control) | — (sin atención) | 416 flat | — | 126.209 | 0.715 | `--arch mlp` |
| C · texto | un carácter | 257 | chars ↔ chars | 35.713 | esperando la 3070 | `--formulation text` |
| A+C · híbrido | feature o carácter | 270 | texto ↔ tabular | 39.073 | esperando la 3070 | `--formulation hybrid` |
| C2 · torre | un carácter (en la torre) | 257 + 13 | solo chars ↔ chars | 96.225 | esperando la 3070 | `--arch tower` |
| B · listwise | un producto entero | 8 | producto ↔ producto | 41.601 | 0.664 | `--arch listwise` |

Referencias sin red: GBM 0.762, regresión logística 0.660 (mismo split, `eda/verificaciones.py`).
El transformer tabular ya supera al GBM y al MLP con 4,5× sus parámetros: primera evidencia de
que la atención aporta. Las tres de texto son la parte cara y corren en la suite de la 3070.
