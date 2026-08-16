# Análisis de la primera tanda GPU (16/08) y diseño de la segunda

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

## 5. Pendientes analíticos (con los checkpoints ya disponibles)

- Mapas de atención del campeón (¿el CLS mira al token de estado? ¿price_rel × tier?).
- Métricas por página (top-1 de la query, NDCG) — eje "pedir" en el panel.
- GroupKFold si queremos intervalos más finos para la presentación.
