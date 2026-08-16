# Propuestas de Junior — 3 experimentos nuevos para sumar a la suite

> Documento de propuesta. Salió de cruzar ideas propias (mirando el código ya implementado en
> `btr/`) con una consulta a un modelo externo **sin mostrarle el código** — a propósito, para
> tener una segunda opinión sin el sesgo de haber visto ya `btr/model.py`. De ahí surgieron varias
> ideas; estas tres son las que se decidió llevar al repo por relación costo/impacto. Otras cuatro
> quedaron descartadas de esta ronda (ver el final del documento, con el motivo de cada una).

Ninguna de las tres reemplaza la arquitectura ya decidida (`propuesta.md`, 6 arquitecturas ya
implementadas en `btr/model.py`, 24 configuraciones en `experimentos.py`). Son extensiones puntuales
para sumar a esa suite, no un cambio de rumbo.

| # | Propuesta | Tipo | Familia (costo) |
|---|---|---|---|
| 1 | Listwise + texto enriquecido | Variante de algo ya construido | texto (GPU) |
| 2 | Multi-task: `cart` como señal auxiliar | Genuinamente nueva | tabular (barata) |
| 3 | Calibración del modelo | Genuinamente nueva (análisis, no arquitectura) | gratis (sin re-entrenar) |

---

## 1. Listwise + texto enriquecido

### Qué es

`ListwiseTransformer` (`btr/model.py`) ya implementa la formulación donde cada **producto** de una
misma búsqueda es un token, y la self-attention corre entre los productos de la página (¿me compran
a mí dado lo que aparece al lado?). Hoy ese token de producto se arma **solo con features
tabulares** (categoría, marca, precio, etc. vía `FeatureTokenizer`) — el modelo nunca ve el texto
(`title`/`description`) de los productos que está comparando.

La propuesta: enriquecer el token de cada producto agregándole también un resumen de su texto,
antes de que los productos se comparen entre sí.

### Mecánica

Hoy, en `ListwiseTransformer.forward`:

```
tokens = FeatureTokenizer(x_cat, x_num)      # (B*P, 13, d_model)
products = Linear(13*d_model → d_model)(tokens.flatten(1))   # (B*P, d_model), SOLO tabular
```

La variante agrega, antes de esa proyección, el vector de texto que ya sabe producir
`TextTowerModel.encode_text()` (la torre de texto del modelo `tower`: `[CLS] + chars → bloques →
embedding del CLS`, un vector de `d_model` por producto):

```
texto_emb = TextTower.encode_text(x_text)     # (B*P, d_model)
tabular   = FeatureTokenizer(...).flatten(1)  # (B*P, 13*d_model)
products  = Linear(d_model + 13*d_model → d_model)(cat[texto_emb, tabular])
```

El resto de `ListwiseTransformer` (self-attention entre productos de la página, un logit por
producto, loss solo sobre los slots reales vía `prod_mask`) queda igual.

### Qué pregunta responde

Hoy `listwise` (PR-AUC 0.664) pierde contra `features` (0.766). No se sabe si pierde porque:
(a) comparar contra los productos vecinos de página es, en sí, una señal débil en este dataset
(consistente con el EDA §2.5: la competencia entre productos apenas afecta el BTR), o
(b) porque cada producto entra "ciego" a la señal más fuerte que existe en los datos (el sufijo del
título, `listing_status`, con tiers de BTR 0.65/0.03/0.000).

Con esta variante:
- Si sube cerca de 0.766 → la pérdida de `listwise` era por falta de información (texto), no por la
  idea de comparar productos entre sí.
- Si se mantiene bajo → refuerza con más evidencia el hallazgo del EDA: la competencia dentro de la
  página realmente no importa mucho acá (se compran varios productos por búsqueda, "changuito de
  supermercado", no elección exclusiva).

Ambos desenlaces son presentables — no es una apuesta a que mejore, es cerrar una pregunta abierta.

### Costo y riesgo

Familia **texto** (cara, requiere GPU): mete la torre de caracteres dentro de cada producto de
cada query, así que el costo escala con `productos_por_query × longitud_del_texto`. Es la más cara
de las tres propuestas de este documento, pero de bajo riesgo de implementación porque reutiliza
dos piezas ya construidas y ya probadas por separado (`listwise` y la torre de texto de `tower`).

### Cómo correrla

Agregar una entrada a `EXPERIMENTOS` en `experimentos.py`, análoga a las existentes:

```python
'listwise_texto': (['--arch listwise', '--listwise-texto'], 'texto'),  # flag nuevo a implementar
```

(el flag exacto depende de cómo se decida exponerlo en `btr/train.py`; la arquitectura nueva iría
en `btr/model.py` como una variante de `ListwiseTransformer`, no como clase separada).

---

## 2. Multi-task: `cart` como señal auxiliar de entrenamiento

### Qué es

El dataset tiene la columna `cart` (agregado al carrito), que hoy está **excluida como feature de
entrada** porque es leakage estricto: `bought ⟹ cart` en el 100% de las filas (`propuesta.md §2.2`).
Esa decisión no cambia. La propuesta es distinta: usar `cart` **como una segunda tarea de
entrenamiento** (no como input), para que el modelo practique con una señal más abundante mientras
aprende a predecir `bought`.

### Por qué NO es leakage

Importante distinguir de lo ya descartado en `propuesta.md §2.2`: ahí se descartó `cart` como
**feature** (dato que el modelo mira para hacer la predicción). Acá `cart` nunca entra como input,
ni en entrenamiento ni en inferencia — solo se usa como **etiqueta extra** que el modelo también
intenta predecir durante el entrenamiento, con una cabeza de salida que se descarta después. En
producción, el modelo solo expone `p(bought)`.

### Mecánica

Todos los modelos de `btr/model.py` terminan hoy en una sola cabeza:

```python
self.cls_head = nn.Linear(d_model, 1)   # -> p(bought)
```

La variante agrega una segunda cabeza sobre el mismo `[CLS]` (mismo tronco/encoder compartido):

```python
self.cart_head = nn.Linear(d_model, 1)  # -> p(cart), se descarta en inferencia
```

Y la loss combina ambas tareas:

```python
loss = bce(logits_bought, y_bought) + lambda_cart * bce(logits_cart, y_cart)
```

`lambda_cart` es un hiperparámetro a barrer (probar algo como 0.0 → baseline actual, 0.1, 0.3, 0.5).
El dataset ya trae la columna `cart`; solo hace falta que `btr/data.py` la exponga también como
target (hoy solo expone `bought`).

### Por qué puede ayudar

`bought` es una señal escasa y ruidosa (13% positivos). `cart` es más densa (30% positivos) y
ocurre en el mismo proceso de decisión del usuario, un paso antes. El gradiente de la tarea
auxiliar regulariza el encoder compartido — la representación aprendida se ve influida por una señal
más abundante del mismo fenómeno, aunque el forward de producción nunca calcule esa cabeza.

### Costo y riesgo

Familia **tabular** (barata) — se puede aplicar sobre `features`, `mlp`, o cualquier arquitectura
existente sin tocar el encoder, solo la cabeza de salida y la loss. El riesgo principal es elegir mal
`lambda_cart`: si es muy alto, el modelo puede optimizar más por `cart` que por `bought` — hay que
barrer valores y mirar PR-AUC de validación **de `bought`**, no de `cart`, para elegir el mejor.

### Cómo correrla

```python
'feat_multitask_l01': (['--formulation', 'features', '--cart-aux', '0.1'], 'tabular'),
'feat_multitask_l03': (['--formulation', 'features', '--cart-aux', '0.3'], 'tabular'),
```

(flag `--cart-aux <lambda>` nuevo a implementar en `btr/train.py`).

---

## 3. Calibración del modelo

### Qué es y por qué falta

El repo hoy mide **discriminación**: ROC-AUC y PR-AUC responden "¿el modelo ordena bien los
positivos por encima de los negativos?". Nunca mide **calibración**: "cuando el modelo dice
`p=0.7`, ¿de cada 100 productos así, compran realmente 70?". Son propiedades matemáticamente
independientes — un modelo puede tener AUC excelente (orden perfecto) y estar mal calibrado (los
números sistemáticamente corridos hacia arriba o abajo), porque AUC es invariante a cualquier
transformación monótona de la probabilidad de salida.

### Por qué importa para este problema puntual

El BTR de negocio se define en `propuesta.md §1.2` como el **promedio de las probabilidades
predichas** por producto (`BTR(producto) = E[p(bought)]`). Si las probabilidades individuales están
mal calibradas (aunque el orden esté perfecto), ese promedio agregado — lo que efectivamente se le
entrega al negocio para decidir qué promocionar — queda sesgado, y hoy nada en el repo lo detectaría.
Es un requisito silencioso de la consigna que no se está verificando.

### Método

Se aplica sobre modelos **ya entrenados** (los checkpoints en `pesos/`, vía `load_checkpoint` +
`predict_proba`), sobre el set de validación/test. No requiere re-entrenar nada.

1. **Reliability diagram** ("diagrama de confiabilidad"): agrupar las predicciones en ~10 franjas
   por probabilidad predicha (deciles), y graficar el promedio predicho de cada franja contra la
   tasa real de `bought` observada en esa franja. Si el modelo calibra bien, los puntos caen sobre
   la diagonal.
2. **Brier score**: `mean((p_pred - y)^2)` sobre el set de test — el equivalente del error
   cuadrático medio para probabilidades vs. etiquetas binarias. Complementa AUC porque sí es
   sensible a la magnitud del error, no solo al orden.
3. **ECE** (Expected Calibration Error): resumen numérico del reliability diagram — promedio
   ponderado por franja de `|predicho − observado|`.
4. Si aparece descalibración clara: **temperature scaling** (Guo et al., 2017) — un único parámetro
   escalar `T` que divide los logits antes del sigmoide, ajustado sobre el set de validación con el
   modelo ya congelado (no se re-entrena la red, solo se calibra la salida final).

### Costo

El más barato de los tres: cero cómputo de entrenamiento, corre como script de evaluación sobre
checkpoints que ya existen. No depende de la corrida en GPU — se puede hacer en paralelo con
cualquier otra cosa, incluso sobre los modelos tabulares que ya corrieron en CPU.

### Cómo correrla

Script nuevo, p. ej. `eda/calibracion.py`, que reutilice `load_checkpoint` de `btr/model.py`:

```python
from btr.model import load_checkpoint
model, prep = load_checkpoint('pesos/features_d32_h4_l2_linear_seed42.pt')
# predict_proba sobre val/test, agrupar en deciles, graficar reliability diagram,
# calcular Brier score y ECE
```

---

## Descartado en esta ronda (no va al `.md` de experimentos, queda como nota)

Cuatro ideas adicionales surgieron en la discusión pero quedaron afuera por bajo payoff esperado o
por depender demasiado de haber visto ya el código (sesgo, sin validación externa):

- **TabTransformer puro** (atención solo entre categóricas, numéricas concatenadas después de los
  bloques): variante intermedia entre `mlp` y `features` ya existentes. Sin validar externamente.
- **SwiGLU** en el `FeedForward` (reemplazo moderno de ReLU): mismo lugar que la ablación
  ReLU/GELU ya planeada en `propuesta.md §7.4`, impacto esperado chico con secuencias de 14 tokens.
- **Attention pooling** (tercera opción de pooling adaptativo, además de CLS/mean): completa una
  ablación ya prevista pero de impacto incierto.
- **Token de contexto de filtros** (agregar `filter_category`/`filter_price_min/max` como token
  extra en `listwise`): el EDA (`propuesta.md §2.1`) ya midió que esos filtros son 100% redundantes
  con los atributos del producto — resultado esperado nulo.

## Referencias

- Guo, C. et al., ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599)
  (2017) — temperature scaling, base del punto 3.
- Caruana, R., ["Multitask Learning"](https://link.springer.com/article/10.1023/A:1007379606734)
  (1997) — fundamento del punto 2.
- Resto de referencias de arquitectura en `propuesta.md §12`.
