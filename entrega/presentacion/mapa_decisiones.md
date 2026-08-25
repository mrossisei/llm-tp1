# Mapa de decisiones: un eje por vez sobre la arquitectura base

> Respuesta al plan de Juan (24/08): ordenar los experimentos como un flujo lógico de
> decisiones sobre la arquitectura base, cada uno con su justificación y su resultado.
> **Ese flujo es exactamente lo que se corrió** — 838 corridas, 116 configuraciones, todas
> con 6 seeds y comparaciones apareadas — y este archivo es el índice para presentarlo.
> Figura paraguas: `../experimentos/graficos/decisiones_por_eje.png`. El detalle completo de
> cada eje vive en `../experimentos/analisis.md` (secciones citadas); todo es explorable en
> `panel.html` (raíz del repo).

Convención: **test PR-AUC, media ± desvío sobre 6 seeds**; azar 0.131; baselines logística
0.660 / GBM 0.762. "Base" = d32 · 4 cabezas · 2 bloques.

## 0 · Protocolo antes que experimentos (§1–2)

**Por qué**: sin protocolo, los deltas entre configs son ruido de split/init. Decisiones:
split **por query** 70/15/15 (una búsqueda no cruza splits), 6 seeds con **análisis apareado**
("gana x/6"), selección por PR-AUC de **validación**, early stopping. La paciencia también se
experimentó: 8→20 mejora 21/24 configs tabulares (+0.037 en la mejor) pero **empeora** texto
puro (sobreajuste de selección) → paciencia 20 tabular / 8 texto.

## 1 · ¿Qué es un token? — la formulación (§1, §5, §8)

**Por qué**: es LA pregunta conceptual del TP; cada respuesta es una arquitectura distinta.
Corridas las 6: `features` (cada feature es un token, FT-Transformer) **0.798** ✓ ·
`fusion` (features + resumen del texto como token 15) 0.775 · `tower` (torre de texto aparte)
0.772 · `listwise` (los productos de la query son los tokens) 0.740 · `hybrid` (features + 256
chars en una secuencia) 0.735 · `text` (solo caracteres, la demo) 0.634.
**Decisión**: features — el texto solo diluye o empata lo que el status ya trae limpio.

## 2 · ¿La atención aporta? — transformer vs MLP (§2.1)

**Por qué**: justificar el transformer contra el control sin atención, misma entrada.
**Δ = +0.048 apareado, gana 5/6**, con el MLP teniendo 4,5× más parámetros (126k vs 28k).
Matiz honesto (§5): con la mejor representación para cada uno, la ventaja es +0.027.

## 3 · Cabezas de atención (§2, §8, camp_*)

**Por qué**: ¿la señal necesita varias "consultas" en paralelo o una sola rica? Resultado con
sorpresa metodológica: sobre embeddings, **1 cabeza gana** (0.816 vs 0.798, +0.018 — coherente
con una señal dominante); sobre **ordinal**, 4 cabezas ganan (0.824 vs 0.800). **El eje
interactúa con el encoding** — por eso las decisiones se toman por val sobre la base final, no
se heredan de otra base. Decisión: 4 cabezas.

## 4 · Profundidad — bloques (§8, camp_ordinal_l4, min_l1)

**Por qué**: ¿el problema necesita composición profunda? No: sobre ordinal, 1 bloque 0.801 ·
**2 bloques 0.824** ✓ · 4 bloques 0.811 (51k params). Más capacidad no compra nada; menos,
pierde poco. Decisión: 2.

## 5 · d_model — capacidad (§8, min_*, §13.2–13.3)

**Por qué**: dimensionar el embedding al problema, no al hábito. Meseta amplia: d8 (1.937
params) 0.814 · d16 (6.945) 0.815 · **d32 (26.177) 0.824** ✓ · d64 (105k, embeddings) 0.815.
Epílogo fuerte (7ª tanda): **min_d16l1 (3.713 params) empata al campeón** (0.825), y
destilando del ensemble el nivel campeón aguanta hasta **1.937 params** — la curva completa en
`curva_compresion.png`. Decisión: d32 por val; la versión comprimida queda documentada.

## 6 · Micro-ejes de la entrada (§1, §3)

Numéricas **linear vs bins**: bins −0.023 (los cortes rompen la U del precio) → linear.
**Pooling CLS vs mean**: mean −0.034 (6/6) → CLS. **Positional encoding**: −0.004 (los
features no tienen orden; la identidad vive en los pesos por-feature) → sin positional.
**Máscara causal**: degenerada con CLS al inicio (ROC 0.500, §2.7) → bidireccional.

## 7 · Encoding de las categóricas — la decisión que nombra al modelo (§5, §8)

**Por qué**: sugerencia explícita del enunciado; y one-hot+proyección ≡ embedding (se prueba
solo en MLP). Menú completo, todo lo demás fijo: **ordinal (rango por BTR de train) 0.824** ✓ ·
target 0.813 · embedding 0.798 · one-hot/MLP 0.797 · hashing 0.498 · freq 0.218.
Los contraejemplos calibran: la codificación debe **preservar nivel→propensión**. Hipótesis
previa invertida por los datos (esperábamos embedding ≥ target ≥ ordinal). Encoding **por
feature** también medido (§8): aislar solo `listing_status` en ordinal no alcanza al global.

## 8 · "Algoritmos de embedding" — inits y pre-entrenamiento propio (§6, §10)

**Por qué**: ¿arrancar de pesos informados ayuda? **w2v-init** (skipgram propio, words) +0.010
sobre words pero < chars · **MLM 20 épocas** (BERT-style sobre features): +0.011 sobre
embeddings, −0.007 sobre ordinal (el prior ya hace ese trabajo) · **AE cuello de botella**:
mismo patrón que MLM. Decisión: init aleatoria.

## 9 · Regularización (6ª tanda, §13.1)

**Por qué**: pregunta de Fer post-clase 3. Barrido de weight decay {0, 1e-3, 1e-2, 1e-1},
dropout {0, 0.1, 0.3}, feature-dropout, label smoothing: **todo decorativo** (reg_nada
−0.0004) — la regularización efectiva es early stopping + capacidad chica + prior ordinal.
Ablaciones estructurales: **sin residuales −0.59** (casi no entrena) · sin LayerNorm −0.037.

## 10 · Transfer learning (6ª–8ª tandas, §13, §15)

Con checkpoints propios: probe congelado **gana 6/6** (representación linealmente separable) ·
distilación del deep-ensemble: neutra en d32, **+0.014 (5/6) justo donde falta capacidad**
(1.937 params). Con preentrenado externo (MiniLM): **congelado RESTA** en todas las variantes
(la partición de tiers es anti-semántica), **fine-tuning repara 6/6** y sin el regex lee el
status desde texto mejor que nuestros encoders (0.798 > tower 0.775) — pero nada supera 0.824.
Figuras: `curva_compresion.png`, `bert_transfer.png`.

## El final del flujo

Cada eje convergió: `feat_ordinal` (0.824 ± 0.018 · ensemble 0.834 · CV 5×6 0.821 · top-1
página 0.912). La selección se cerró en la 4ª tanda; las tandas 5–8 son exploración post-cierre
que **no encontró nada mejor** — y eso también es un resultado del flujo.
