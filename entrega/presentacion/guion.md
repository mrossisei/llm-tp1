# Guión de la presentación — TP1: Predicción de BTR con Transformers

**Duración medida: ~32 min de habla pura**, ~36 min con pausas y transiciones. El rango pedido
es 25–30, así que **hay que recortar** — abajo está la ruta corta. Está escrito **para decir en
voz alta**, no para leer: frases cortas, sin subordinadas largas, con los tecnicismos justos. Lo
que no está acá va en el apéndice de preguntas, al final.

El mazo tiene **29 pantallas**, todas se presentan — incluidas 4 divisorias de sección y el
cierre. El apéndice de preguntas de este archivo no tiene diapositiva: es solo para responder.

La presentación es `presentacion.html` (navegador, flechas ← →; imprimir a PDF si Campus pide
archivo). `presentacion.pptx` es la misma, con este guión como notas del orador.

Cada sección indica **[tiempo]**, el texto hablado, y `→ en pantalla` qué señalar.

## Ruta corta — para entrar en 28 minutos

Estás en **~33 min de habla · ~37 con pausas**. Hay que sacar unos 7 min. Recortá en este orden:

| # | Sección | Qué recortar | Ahorro |
|---|---|---|---|
| 19 | Experimento 6 · inits | la sección entera — se menciona en una frase en la 21 | 0.9 min |
| 21 | Desafíos | contar los cuatro paneles en 1.2 min | 0.6 min |
| 20 | Experimento 7 · texto | quedarse con «recuperable pero redundante» y el gráfico | 0.6 min |
| 8 | Métricas | saltear el párrafo de por qué no split temporal ni por producto | 0.6 min |
| 10 | Arquitectura | narrar el diagrama en 2 min en vez de 2.5 | 0.5 min |
| 7 | ¿Best Seller? | la tríada se explica en un minuto | 0.5 min |
| 11 | Alternativas | nombrar las tres sin desarrollar el porqué de cada una | 0.5 min |
| 5 | EDA (2) | las tres tarjetas de la derecha, más rápido | 0.4 min |

**Lo que no se toca nunca** — por rubro o por peso propio: la 5 (el hallazgo del sufijo), la 10
(la arquitectura), la 18 (el encoding, el experimento decisivo), la 21 (desafíos: el enunciado la
pide como punto propio de la presentación), la 25 (overfitting/underfitting: la pide el ejercicio
2.3) y la 27 (el ejercicio teórico de personalización).

La **19 (inits)** es el primer recorte porque no responde a ningún punto del enunciado: es un eje
secundario con resultado negativo. Si la salteás, decilo en una frase dentro de la 21 —
«pre-entrenar tampoco aportó» — y listo.

---

## 1 · Portada — [30 s]

> "Buenas. Vamos a presentar el TP1: predicción de Buy Through Rate en un e-commerce de
> supermercado, usando Transformers.
>
> Lo vamos a contar en cuatro partes. Primero el problema: qué predecimos y qué encontramos en
> los datos. Después el modelo, dónde metimos el transformer y por qué ahí. Tercero los
> experimentos, que es donde decidimos cada pieza midiendo. Y al final los resultados, con el
> ejercicio teórico de personalización."

---

## 2 · Divisoria — El problema — [5 s]

> "El problema."

`→ en pantalla`: pantalla oscura, número grande, «qué predecimos, con qué datos, y cómo lo medimos».

---

## 3 · El problema y la formulación — [1.3 min]

> "Primero: qué es el BTR.
>
> Es compras sobre impresiones. De todas las veces que mostramos un producto, cuántas terminaron
> en compra. Es una métrica de negocio — sirve para decidir qué promocionar.
>
> Lo primero que tuvimos que decidir fue **qué predice el modelo exactamente**. Y no es obvio,
> porque el dataset no es una lista de productos. Es un registro de eventos: cada fila es un
> producto que apareció en una búsqueda. El mismo producto puede estar en varias búsquedas, y
> cada vez es una fila distinta.
>
> Así que lo planteamos así: el modelo mira **una impresión** y dice qué probabilidad hay de que
> la compren.
>
> *(señalando el diagrama)* Fila del CSV, entra al modelo, sale una probabilidad. Y si querés el
> BTR de un producto, promediás las probabilidades de todas sus impresiones. Eso te da el ranking.
>
> Esto trae dos consecuencias que van a volver más adelante. Una: las métricas tienen que ser de
> clasificación, fila por fila. Nada de accuracy — ya vamos a ver por qué. Y dos, esta es
> importante: **no elegimos ningún umbral**. Nunca decimos 'si pasa de 0.5, lo compran'. Solo
> ordenamos."

`→ en pantalla`: el flujo fila → p → agregación.

---

## 4 · EDA (1): la estructura y la primera trampa — [1.4 min]

> "Vamos a los datos.
>
> Diez mil impresiones, repartidas en dos mil búsquedas de entre uno y ocho productos. Trece por
> ciento de compras. Y los filtros de la búsqueda son constantes dentro de cada query — o sea,
> son contexto de la búsqueda, no del producto. Verificamos que el cien por ciento de los
> productos mostrados cumple su filtro, así que `filter_category` y `filter_storage_type` no
> aportan nada a nivel fila.
>
> Ahora, la primera trampa. Y esta es importante.
>
> *(señalando el panel rojo)* La columna `cart`. Si un producto se compró, **siempre** estuvo en
> el carrito. Cien por ciento de las filas, sin excepción.
>
> Eso parece una feature buenísima, y de hecho lo es — pero es trampa. `cart` es parte del mismo
> embudo: impresión, carrito, compra. Es un resultado del evento, no algo que sepamos cuando
> tenemos que decidir qué promocionar. En producción esa columna llega vacía.
>
> Y es peor que eso: si el carrito está en falso, la compra es cero. Siempre. Así que el setenta
> por ciento del dataset queda resuelto de una. El modelo aprendería a mirar solo eso.
>
> Entonces `cart` queda afuera. Más adelante la usamos de otra forma — como etiqueta auxiliar,
> que no es leakage porque el modelo la predice en vez de recibirla — pero tampoco aportó."

`→ en pantalla`: la tabla de estructura y el panel rojo del leakage.

---

## 5 · EDA (2): la señal está escondida en el texto — [2.0 min]

> "Este es el hallazgo central del trabajo.
>
> Miramos los títulos de los productos y notamos que terminan con algo entre paréntesis. 'Best
> Seller'. 'Top Rated'. 'New Listing'. Pasa en el noventa y cinco por ciento de las filas. No era
> una columna: había que sacarlo del texto con una expresión regular.
>
> Y cuando lo separamos, aparece esto.
>
> *(señalando la tabla)* Hay veinte sufijos distintos, y se parten en tres grupos limpios. Cuatro
> compran entre el sesenta y tres y el sesenta y ocho por ciento. Otros cuatro compran entre el
> dos y el cuatro. Y los once restantes, más los que no tienen sufijo: **cero exacto**. Ninguna
> compra.
>
> Esa sola columna parte el dataset al medio.
>
> Pero acá está lo que más nos sorprendió: **no es cuestión de sentimiento**. 'Highly Rated'
> suena tan bien como 'Top Rated', y compra cincuenta veces menos. Si hubiéramos ordenado los
> niveles a ojo, por lo que parecen significar, nos equivocábamos feo. Hay que aprender la
> partición exacta de los datos, no la valencia de las palabras. Esto vuelve en la diapositiva
> de codificación, que es el experimento decisivo.
>
> *(señalando las tarjetas de la derecha)* Tres cosas más del EDA.
>
> El precio tiene efecto de **U invertida** dentro del rango que filtró el usuario: se compra más
> en el medio que en los extremos. Y está condicionado al tier. Por eso derivamos `price_rel`, la
> posición del precio dentro del rango — que es una señal relacional entre el producto y la
> búsqueda.
>
> Los **timestamps están rotos**: dentro de una misma búsqueda hay eventos separados por hasta
> dos años. Es imposible. Así que ni split temporal ni features de tiempo. Después lo verificamos:
> agregar hora y día empeora.
>
> Y hay **redundancias**: el tamaño del envase es el peso, las dimensiones son el envase, la
> descripción repite el sufijo."

`→ en pantalla`: la tabla de tiers, y las tres tarjetas.

---

## 6 · Features y preprocesamiento — [1.5 min]

> "Con ese EDA, esto es lo que entra al modelo.
>
> Siete columnas categóricas y seis numéricas. Trece en total.
>
> Dos de ellas las derivamos nosotros. `listing_status`, que es el sufijo del título que acabamos
> de ver. Y `price_rel`, la posición del precio en el rango filtrado. Ninguna de las dos venía en
> el CSV.
>
> El preprocesamiento va por tipo. *(señalando el encabezado de cada panel)* Las numéricas van
> a media cero — z-score, con logaritmo antes en las dos que están sesgadas, precio y peso. Y las
> categóricas, cada nivel a un número. Ahí nos detuvimos, porque el enunciado sugería investigar
> one-hot y alternativas. Le dedicamos un experimento entero, es la diapositiva dieciocho.
>
> Un punto que nos importa: **todo se ajusta solo con train**. Los vocabularios, los promedios,
> las tablas de codificación. Si los calculáramos con todo el dataset estaríamos filtrando
> información de test. Y los niveles que aparecen en test y no estaban en train van a un valor
> especial de 'desconocido'.
>
> *(señalando el panel de abajo)* Y lo que queda afuera. Cada etiqueta roja dice por qué:
> `cart` es leakage, `query_id` solo sirve para partir, el timestamp está roto, los dos filtros
> son constantes dentro de la búsqueda, y las últimas cuatro repiten algo que ya tenemos.
>
> Acá hay algo metodológico que queremos remarcar: **no descartamos en papel**. Cada descarte lo
> volvimos a meter y medimos. Todos dieron diferencia cero."

`→ en pantalla`: los dos paneles de arriba con las trece features en pastillas — las dos verdes
son las que derivamos — y abajo el panel rojo de descartes.

---

## 7 · ¿Es válido usar "Best Seller"? — [1.5 min]

> "Acá tuvimos una discusión en el equipo que vale la pena contar.
>
> Esos badges se asignan **después** de que un producto vendió mucho. Entonces, ¿no es circular
> usarlos para predecir ventas?
>
> Lo resolvimos separando tres tipos de información.
>
> Uno: resultado del mismo evento. Eso es `cart`. Leakage estricto, nunca se usa.
>
> Dos: **estado del producto al momento de mostrarlo**. Los badges están acá. Cuando el buscador
> arma la página, el badge ya existe — está disponible. Y además es plausiblemente causal: la
> gente compra más lo que tiene prueba social. Así que es válido para **predecir**. Pero sí es
> circular para **decidir promociones**, y es ciego al producto nuevo que todavía no tiene badge.
>
> Tres: atributos intrínsecos, marca, peso, categoría. Válidos siempre.
>
> Entonces entrenamos **dos familias de modelos**. Una con estado, que responde 'qué promociono
> hoy'. Y otra sin estado, ni parseado ni escondido en el texto, que responde 'qué esperar de un
> producto nuevo'.
>
> *(señalando las tarjetas)* Y el resultado encuadra todo el trabajo. Con estado llegamos a
> ochenta y dos. Sin estado, **nadie** pasa de dieciséis. Ni el GBM, ni nada.
>
> Y eso no es un problema del modelo. Es un hallazgo sobre el dataset: el sesenta y un por ciento
> de las filas vive en tiers donde el BTR es cero exacto. Sin esa columna, no hay nada que
> aprender."

---

## 8 · Métricas y protocolo — [2.0 min]

> "Cómo medimos.
>
> Con trece por ciento de positivos, el accuracy no sirve. Un modelo que diga 'nadie compra nada'
> acierta el ochenta y siete por ciento y no encuentra un solo comprador.
>
> Así que la métrica principal es **PR-AUC** — precisión y recall — que es la que sugiere el
> enunciado. Mide qué tan bien el modelo **ordena**, sin tener que elegir ningún umbral. Y el
> piso no es cero coma cinco: es la tasa de compra, cero coma ciento treinta y uno. Contra eso
> comparamos siempre.
>
> Aparte reportamos ROC-AUC, y guardamos log-loss y Brier para ver la calidad de las
> probabilidades.
>
> Sin umbral, como dijimos. Aunque igual lo medimos: el F1 máximo cae en cero cuarenta, no en cero
> cinco. O sea que cualquier umbral fijo hubiera sido arbitrario.
>
> Guardamos **dieciséis métricas por época** en cada corrida. Eso nos deja ver overfitting y
> underfitting en las curvas de train y validación, para todas las corridas.
>
> *(pasando al panel de la derecha)* Y cómo particionamos.
>
> Por `query_id`, setenta quince quince. Una búsqueda entera cae del mismo lado. Si partiéramos
> fila por fila, productos de la misma página quedarían en train y en test, y estaríamos filtrando
> el contexto.
>
> ¿Por qué no temporal? Timestamps rotos. ¿Por producto? Lo verificamos: el noventa y nueve por
> ciento de los títulos son únicos, y si restringimos test a productos nunca vistos las métricas
> son idénticas. El modelo no recibe identidad del producto, así que no tiene cómo memorizar.
>
> Sobre la varianza: seguimos lo que priorizó la cátedra, que es promediar corridas. **Seis
> semillas por configuración.** Y como cierre igual corrimos validación cruzada, cinco por seis:
> da cero ochocientos veintiuno con desvío cero cero doce. Consistente.
>
> Y una disciplina que mantuvimos todo el trabajo: los hiperparámetros se eligen mirando
> **validación**. Test se reporta al final, y nada más."

---

## 9 · Divisoria — El modelo — [5 s]

> "El modelo."

`→ en pantalla`: pantalla oscura, número grande, «dónde va el transformer, y por qué ahí».

---

## 10 · La arquitectura: dónde va el transformer y por qué — [2.5 min]

> "La pregunta del enunciado es dónde, cómo y por qué un transformer. Nuestra respuesta sale del
> EDA.
>
> La señal de este problema es **relacional**. El precio no importa por sí solo, importa en
> relación al rango que filtró el usuario. Y la U del precio está condicionada al tier del
> producto. O sea: lo que importa son los cruces entre features.
>
> Un modelo lineal necesita que le des esos cruces hechos a mano. La self-attention los computa
> sola, de a pares, y aprendidos. Es la generalización de las feature crosses.
>
> Por eso elegimos un **FT-Transformer**: cada feature es un token.
>
> Ahora, la pregunta natural: ¿cómo puede la atención comparar un precio con una marca? Y la
> respuesta es que no hace falta que los tokens sean del mismo tipo. Hace falta que vivan en el
> **mismo espacio**. Y eso es exactamente lo que hace el tokenizador: cada numérica entra como x
> por w más b, cada categórica con su codificación, y las dos salen como vectores del mismo
> tamaño. Es el mismo principio por el que en un modelo de lenguaje conviven la palabra 'perro',
> una coma y un número.
>
> *(recorriendo el diagrama)* De ahí para arriba es un encoder estándar. Usamos los mismos
> bloques de la demo de la cátedra: atención multi-cabeza, el MLP interno, bloques pre-LN con
> conexiones residuales.
>
> Con dos adaptaciones, y las dos justificadas.
>
> La primera: **sacamos la máscara causal**. Esto es clasificar un conjunto, no generar texto. No
> hay 'siguiente token' que adivinar. Atención bidireccional, como BERT.
>
> La segunda: el escalado por raíz de d_k se hace **una** vez, como en el paper. La demo lo
> aplicaba dos veces, y eso aplana la atención de más.
>
> Agregamos un token **CLS** de lectura. No aporta información: la recolecta. Como atiende a todo
> en cada capa, su estado final es el resumen que va al clasificador. Es lo que hace BERT.
>
> Y **sin positional encoding**. Un conjunto de features no tiene orden — el precio no está
> 'antes' que la marca. La identidad de cada columna ya vive en sus propios parámetros. Eso no lo
> asumimos: lo medimos, y agregarlo da diferencia cero.
>
> El tamaño arranca donde sugiere el enunciado: d_model treinta y dos, cuatro cabezas, dos
> bloques."

`→ en pantalla`: el diagrama completo, del CSV a p(bought).

---

## 11 · Alternativas consideradas — y medidas — [1.2 min]

> "Antes de comprometernos con eso, evaluamos dónde **más** podía ir el transformer. Porque 'usá
> un transformer' no te dice dónde ponerlo.
>
> Tres alternativas serias.
>
> Una: **el texto crudo como tokens**. Los caracteres del título y la descripción, que es la demo
> literal. La señal está ahí, sin duda. Pero el EDA ya nos mostró que se extrae con una expresión
> regular — gastar atención cuadrática sobre doscientos cincuenta y siete tokens para eso es caro.
>
> Dos: **los productos de la página como tokens**. Eso modela la competencia dentro de la
> búsqueda. Pero el EDA midió competencia débil: en muchas páginas se compra más de un producto.
>
> Tres: **el transformer solo como encoder de texto**, y un MLP arriba que clasifica. El problema
> es que comprime toda la señal a un cuello de botella antes de decidir.
>
> Elegimos features-como-tokens de base. Pero no las descartamos en papel: **implementamos y
> corrimos las tres**, más dos variantes. Los resultados les dan la razón a los diagnósticos del
> EDA, y los vemos en la diapositiva veinte."

---

## 12 · Baselines: la vara — [35 s]

> "Antes del transformer, armamos una escalera de complejidad. Mismo split, mismas métricas,
> mismas seis semillas.
>
> Regresión **logística**: cero seiscientos noventa y ocho. Esa es la vara lineal.
>
> Un **MLP** denso, con exactamente los mismos embeddings de entrada que el transformer: cero
> setecientos cuarenta y seis.
>
> Y un **GBM**, que es la vara no lineal fuerte — árboles con interacciones: cero setecientos
> sesenta y dos.
>
> Y nos pusimos una regla de honestidad: **si el transformer no supera esto, la capa de atención
> no se justifica.**"

---

## 13 · Divisoria — Los experimentos — [35 s]

> "Los experimentos. Siete preguntas, y ninguna la contestamos de memoria.
>
> Antes de arrancar, la base. *(señalando la lista)* Todo lo que viene mueve **un solo eje** y
> deja el resto quieto: el mismo tamaño de vector por feature, el mismo optimizador, el mismo
> batch, la misma paciencia. Y **seis semillas** en cada configuración.
>
> Así, cuando algo cambia, sabemos qué lo cambió.
>
> Esa lista de abajo la van a ver repetida al pie de cada experimento, con lo que varía marcado
> a la derecha."

`→ en pantalla`: pantalla oscura, el 3 grande, y los diez hiperparámetros con su explicación en
una línea cada uno.

---

## 14 · Experimento 1: ¿la atención aporta? — [1.3 min]

> "Esta es la comparación central.
>
> Transformer contra MLP, **con la misma entrada**. Los mismos feature-tokens. Lo único que
> cambia es qué los mezcla: atención o capas densas. Y apareado por semilla, así que comparamos
> corrida contra corrida.
>
> Resultado: **más cero cero cuarenta y ocho**, ganando en cinco de seis semillas. Y el MLP tenía
> cuatro veces y media más parámetros. Así que no es cuestión de tamaño.
>
> Pero le hicimos dos refinamientos, porque nos parecía una vara fácil.
>
> *(señalando la primera tarjeta)* Primero: al MLP le probamos one-hot crudo en vez de
> embeddings, y mejoró a cero setecientos noventa y siete. O sea, parte de su déficit era la
> entrada, no la arquitectura. Contra el **mejor** MLP posible, la ventaja del transformer baja a
> más cero cero veintisiete. Sigue ganando, pero con la vara más alta.
>
> *(segunda tarjeta)* Y segundo: nos preguntamos si la atención no estaría simplemente
> descubriendo el cruce precio por tier que ya sabíamos del EDA. Así que se lo dimos a mano a la
> logística. Mejora, pero eso explica **solo el doce por ciento** del gap. La atención aprende
> bastante más que esa única interacción."

---

## 15 · Experimento 2: ¿cuántas cabezas? — [0.9 min]

> "Multi-head significa varias consultas en paralelo, cada una en su propio subespacio. La
> pregunta era si este problema las necesita, o si hay una sola señal dominante y con una alcanza.
>
> Y acá tuvimos una sorpresa metodológica.
>
> Con **embeddings**, una cabeza grande le gana a cuatro chicas: cero ochocientos dieciséis contra
> cero setecientos noventa y ocho. Una consulta rica vale más que cuatro pobres — que es
> coherente con el EDA, donde hay una señal que manda.
>
> Pero sobre la base **ordinal**, que es la que terminó ganando, **cuatro cabezas ganan**: cero
> ochocientos veinticuatro contra cero ochocientos.
>
> O sea: el eje **interactúa con la codificación**. Y de ahí sacamos una regla para todo el
> trabajo: ninguna decisión se hereda de otra base. Cada una se vuelve a decidir por validación
> sobre la base final."

`→ en pantalla`: el **4** en verde grande arriba a la derecha, el gráfico a la izquierda, y al
pie la base con «varía · cabezas 1 / 2 / 4» en la pastilla lila.

---

## 16 · Experimento 3: ¿cuánta profundidad? — [35 s]

> "Más bloques es componer atención sobre atención. Pero con trece features que ya se ven todas
> de un salto, la hipótesis era que no hace falta mucha profundidad.
>
> Confirmado. Un bloque pierde poco. **Dos ganan.** Cuatro no suman nada, con el doble de
> parámetros.
>
> Y esto se conecta con la interpretabilidad, que vemos al final: el CLS ya concentra el setenta y
> cinco por ciento de su atención en el estado **en la primera capa**. No necesita más capas para
> encontrar lo que importa."

`→ en pantalla`: el **2** en verde, y al pie «varía · bloques 1 / 2 / 4».

---

## 17 · Experimento 4: ¿qué d_model? — [0.8 min]

> "Acá la idea era dimensionar el embedding al problema — trece features, diez mil filas — y no
> al hábito de los papers, que usan quinientos doce.
>
> Lo que encontramos es una **meseta amplia**. De ocho a sesenta y cuatro, los resultados son
> parecidos. La señal cabe en poquísimos parámetros. Treinta y dos se elige por validación, pero
> sin mucho margen.
>
> Y hay un epílogo fuerte. Un modelo de dieciséis dimensiones con un solo bloque —**tres mil
> setecientos parámetros**— empata al campeón. Y destilando de un ensamble, el nivel campeón
> aguanta hasta **mil novecientos**.
>
> Cerrando los tres ejes de capacidad: cabezas, profundidad y dimensión. Ninguno movió mucho la
> aguja. El experimento decisivo no era la capacidad. Era **la codificación de la entrada**."

`→ en pantalla`: el **32** en verde, y al pie «varía · d_model 8 / 16 / 32 / 64».

---

## 18 · Experimento 5: la codificación de las categóricas — [2.2 min]

> "Y este es el experimento decisivo del trabajo.
>
> El enunciado sugería investigar codificaciones, one-hot y alternativas. Implementamos y corrimos
> cinco, con todo lo demás fijo.
>
> Antes que nada, un resultado teórico que nos ahorró un experimento: **one-hot seguido de una
> capa lineal aprende exactamente la misma matriz que un embedding**. Multiplicar un vector
> one-hot por una matriz te devuelve una fila de esa matriz, que es lo que hace el lookup del
> embedding. Son el mismo modelo. Así que one-hot solo lo probamos como entrada cruda al MLP.
>
> Las que sí probamos: **embedding aprendido** por columna, que es el estándar. **Target
> encoding**, cada nivel a su BTR promedio suavizado de train. **Ordinal**, cada nivel a su
> **rango** al ordenar por ese BTR. **Frecuencia** y **hashing**.
>
> *(señalando el gráfico)* Y acá está la sorpresa: gana el **ordinal**, cero ochocientos
> veinticuatro. Por encima del target, cero ochocientos trece, y bastante por encima del embedding,
> cero setecientos noventa y ocho.
>
> ¿Por qué gana, si el embedding puede aprender cualquier cosa? Justamente por eso. Con diez mil
> filas, 'poder aprender cualquier cosa' es sobreajustar. Una marca que aparece doce veces no
> alcanza para aprenderle treinta y dos números. El rango le inyecta el orden como **prior**, con
> un solo escalar por nivel.
>
> Y le gana al target porque los rangos quedan equiespaciados, mientras que las magnitudes están
> apelmazadas: cero sesenta y cinco, cero cero tres, cero exacto. Los rangos están mejor
> condicionados.
>
> Los dos que fallan calibran la regla. **Frecuencia** da cero veintidós — codifica qué tan común
> es un nivel, y eso no tiene nada que ver con comprarlo. **Hashing** da cero cincuenta — las
> colisiones mezclan tiers distintos. La regla que sacamos es: la codificación tiene que
> **preservar la relación nivel-propensión**.
>
> Y una reflexión honesta: nuestra hipótesis previa era exactamente la inversa. Pensábamos que el
> embedding iba a ganar. **Los datos nos corrigieron**, y esa corrección terminó siendo el mejor
> modelo del trabajo."

`→ en pantalla`: el gráfico ocupa toda la pantalla y la única frase escrita es «nuestra hipótesis
era la inversa». **Esta diapositiva no tiene texto de apoyo**: todo el argumento lo lleva la voz.
Es la más importante del mazo — no la apures.

---

## 19 · Experimento 6: ¿arrancar de pesos informados? — [0.8 min]

> "Este es el pre-entrenamiento de la clase tres, en miniatura. La pregunta: ¿un arranque
> informado le gana a uno aleatorio?
>
> Probamos tres formas, todas con modelos propios.
>
> **MLM estilo BERT** sobre los feature-tokens: veinte épocas enmascarando features, sin usar
> etiquetas. Sobre embeddings suma un poco. Sobre ordinal, **nada** — el prior ordinal ya hace ese
> trabajo de regularización.
>
> **w2v-init**, un skipgram propio, regulariza la variante de palabras pero no alcanza a la de
> caracteres. Y el autoencoder repite el mismo patrón.
>
> Decisión: inicialización aleatoria.
>
> Y si preguntan por transfer desde un preentrenado **de verdad**: lo probamos con MiniLM, está
> en el apéndice. Congelado **resta**. Fine-tuneado repara. Y aun así nada supera nuestro cero
> ochocientos veinticuatro, con veintidós millones de parámetros contra veintiséis mil."

---

## 20 · Experimento 7: ¿y el texto? — [1.4 min]

> "La señal nace en el texto. Entonces la pregunta obvia: ¿el transformer puede leerla solo, sin
> nuestra expresión regular?
>
> Corrimos el arco completo.
>
> **Texto puro**, caracteres como tokens — la demo adaptada de decoder a encoder: cero seiscientos
> cincuenta y dos. Muy por encima del techo sin señal, que era dieciséis. O sea que **encontró el
> sufijo solo**. Pero muy por debajo del tabular: leer caracteres con treinta y seis mil
> parámetros cuesta caro.
>
> **Híbrido**, features y doscientos cincuenta y seis caracteres en una sola secuencia: cero
> setecientos cinco. Peor que el tabular solo. Los tokens de texto **diluyen** la atención sobre
> los trece que importan.
>
> Pero acá hay un dato fino, y es lindo. Al híbrido, sacarle el token parseado no le cuesta nada.
> O sea que **reconstruye desde los caracteres crudos lo que nosotros extraíamos con la regex**.
>
> Y la **fusión** —comprimir el texto a un solo token que entra a la secuencia tabular— cura la
> dilución por completo. Pero empata exacto con la torre. Conclusión: una vez que comprimís,
> cruzar por atención o por concatenación da igual. Lo que importa es comprimir.
>
> La conclusión de diseño: el texto crudo es **recuperable**, pero **redundante** cuando el EDA ya
> parseó la señal."

---

## 21 · Desafíos encontrados — [1.8 min]

> "Cuatro cosas que nos pasaron y vale la pena contar.
>
> *(panel uno)* **El causal degenerado.** Hicimos la ablación de 'qué pasa si dejamos la máscara
> causal', y dio ROC exactamente cero coma cinco. Exactamente. Eso no es 'causal anda peor', eso
> es que algo está roto.
>
> Y el diagnóstico fue este: con máscara causal, nuestro CLS está en la posición cero, así que
> solo puede verse a sí mismo. No ve ninguna feature. El modelo predecía una constante — la misma
> probabilidad para todo el test.
>
> El arreglo es ponerlo al final, que es desde donde lee GPT. Y con eso la respuesta real aparece:
> la bidireccionalidad acá **da igual**. La lección que nos llevamos: los decoders leen desde el
> último token, los encoders pueden poner el CLS adelante — pero las dos cosas tienen que ser
> coherentes.
>
> *(panel dos)* **Las trampas del dataset**: `cart` y los timestamps rotos. El EDA los cazó antes
> de que mordieran.
>
> *(panel tres)* **El cómputo.** La familia de texto es inviable en CPU, por la atención
> cuadrática sobre doscientos cincuenta y siete tokens. Armamos una suite de experimentos
> resumible que corre en una GPU de consumo, y deja cada corrida registrada con sus dieciséis
> métricas. Ochocientas treinta y ocho corridas en total.
>
> *(panel cuatro)* Y **hipótesis que refutamos con datos**. Las contamos porque el método importa:
> bins por cuantiles para la U del precio — el MLP interno ya la captura. Pesos por clase para el
> desbalance — daña, porque PR-AUC es de ranking. Mean pooling en vez de CLS — el CLS gana seis de
> seis. Y positional encoding en features — diferencia cero, como predice la teoría."

---

## 22 · Divisoria — Resultados — [5 s]

> "Resultados."

`→ en pantalla`: pantalla oscura, número grande, «qué quedó en pie y qué aprendimos».

---

## 23 · El modelo final — [1.3 min]

> "El modelo final: un **FT-Transformer con codificación ordinal**.
>
> Trece feature-tokens más el CLS. Dimensión treinta y dos, cuatro cabezas, dos bloques pre-LN,
> sin positional. Una capa lineal de treinta y dos a uno, y sigmoide. **Veintiséis mil ciento
> setenta y siete parámetros** — el más chico de los cinco mejores.
>
> Entrenamiento: AdamW, batch doscientos cincuenta y seis, early stopping por PR-AUC de validación
> con paciencia veinte. Corta cerca de la época sesenta y cuatro.
>
> Cómo lo elegimos, y esto nos importa. En **validación** había un empate técnico entre cuatro
> configuraciones, con diferencias menores a dos milésimas. Validación no puede distinguirlas. Así
> que desempatamos por **parsimonia**: menos parámetros, menor desvío entre semillas, y menor gap
> entre validación y test. Recién **después** miramos test, que confirma la elección.
>
> *(señalando los números)* PR-AUC en test: cero ochocientos veinticuatro, con desvío cero cero
> dieciocho, sobre seis semillas. Validación cruzada cinco por seis: cero ochocientos veintiuno.
> Con ensamble, cero ochocientos treinta y cuatro — y llegamos ahí por dos rutas independientes,
> ensamblando configuraciones o ensamblando inicializaciones.
>
> Contra las varas: GBM cero setecientos sesenta y dos, mejor MLP cero setecientos noventa y
> siete."

---

## 24 · Robustez — [1.0 min]

> "Tres verificaciones sobre el modelo final.
>
> *(señalando la curva)* **Curva de aprendizaje.** Entrenamos con el veinticinco, cincuenta,
> setenta y cinco y cien por ciento de los datos. La curva está casi saturada: el último cuarto
> aporta siete milésimas. Y con el setenta y cinco por ciento de los datos ya le ganamos al GBM
> entrenado con todo.
>
> **Varianza.** Hicimos una grilla de cinco inicializaciones por seis splits, para separar de
> dónde viene el desvío. Y es mitad lotería del split, mitad inicialización. Eso valida el
> protocolo de promediar semillas. Y esa mitad de inicialización es justo lo que el ensamble
> elimina.
>
> **Calibración.** ECE de cero coma cero uno, temperatura cerca de uno. Y esto importa por lo que
> dijimos al principio: el BTR de negocio es el **promedio** de las probabilidades. Si el modelo
> no estuviera calibrado, ese promedio no significaría nada. Está calibrado, así que se puede leer
> directo, sin corrección."

---

## 25 · Overfitting y underfitting — [40 s]

> "El enunciado pide mirar overfitting y underfitting, así que acá están las curvas del modelo
> final.
>
> *(señalando el gráfico)* Train sube sostenido. Validación sube, se aplana, y ahí el early
> stopping corta — la línea punteada — y restaura ese checkpoint. Nunca entrenamos de más.
>
> El gap entre train y validación es moderado y estable. No se abre. Eso descarta overfitting
> descontrolado. Y como las dos suben bien al principio, tampoco hay underfitting.
>
> El dato que lo cierra: del valor de validación al de test hay **once milésimas** de diferencia.
> Es el gap más chico de las cuatro configuraciones finalistas. O sea que la elección por
> validación no nos engañó.
>
> Estas curvas las tenemos para las ochocientas treinta y ocho corridas, no solo para esta."

`→ en pantalla`: la curva de train y validación con la vertical punteada del early stopping.

---

## 26 · ¿El modelo mira donde debe? — [1.3 min]

> "Con catorce tokens, la matriz de atención se puede **mirar** directamente. No hace falta
> ninguna técnica sofisticada.
>
> *(señalando el mapa)* En la primera capa, el CLS pone el **setenta y cinco por ciento** de su
> atención en el token de estado. Y la familia del precio se consulta entre sí: el precio y el
> mínimo del filtro atienden a `price_rel`. Esa es la señal relacional, literal en el mapa. La
> segunda capa ya mezcla todo.
>
> Ahora, hay una crítica conocida a esto — 'attention is not explanation'. La atención te muestra
> dónde mira el modelo, no necesariamente qué usa. Así que lo contrastamos con un diagnóstico
> independiente, basado en resultados.
>
> *(segundo gráfico)* **Importancia por permutación**: rompemos una feature y medimos cuánto se
> cae el modelo. Destruir `listing_status` cuesta sesenta y ocho centésimas de PR-AUC. `price_rel`
> cuesta catorce. Alergenos, cinco. El resto, cero.
>
> Dos métodos independientes, la misma historia. Y es exactamente la del EDA. El círculo cierra.
>
> Y la traducción a negocio: en las páginas de test donde hubo al menos una compra, el producto
> que nuestro modelo pone primero fue efectivamente el comprado el **noventa y uno por ciento** de
> las veces. El azar da veintisiete."

`→ en pantalla`: a la izquierda el mapa de atención a lo alto; a la derecha el gráfico de
permutación y, debajo, los tres recuadros — atención, permutación, negocio. Señalá el mapa,
después el gráfico, y cerrá con el recuadro de negocio.

---

## 27 · Ejercicio 3: personalización — [1.6 min]

> "El ejercicio teórico: cómo haríamos que el BTR dependa de **quién** busca.
>
> Hoy nuestro modelo estima la probabilidad dado el producto y la búsqueda. Personalizar es
> condicionar también al usuario.
>
> Lo primero es que hace falta **dato nuevo**: un identificador de usuario y su historial de
> eventos. El dataset actual no lo trae.
>
> *(panel izquierdo)* Y la extensión natural de **nuestra** arquitectura es directa. Así como
> evaluamos meter el texto como tokens, acá **el historial del usuario entra como tokens**: sus
> últimas compras y búsquedas, cada una codificada con el mismo tokenizador de productos. Y la
> secuencia los atiende.
>
> El ejemplo concreto: alguien que compra comida de perro todos los meses tiene esos productos en
> su historial. Cuando aparecen en la página, la atención cruza historial contra candidato y le
> sube la probabilidad. Esto en producción existe y se llama BST o SASRec — es exactamente esto.
>
> *(panel derecho arriba)* La alternativa clásica, que conecta con la clase dos: **embeddings de
> usuario y producto entrenados con negative sampling**. Item2vec o two-tower — un skipgram donde
> el contexto son los productos con los que el usuario interactuó. Eso sirve como **retrieval** si
> el catálogo fuera enorme, con nuestro modelo de ranker arriba.
>
> *(panel derecho abajo)* Y el detalle que este trabajo nos dejó bien aprendido: el usuario nuevo
> sin historial es exactamente el mismo problema que el producto nuevo sin badge. Cold-start. Y el
> fallback es justo el modelo que ya tenemos, que no depende del usuario."

---

## 28 · Conclusiones — [1.5 min]

> "Cinco conclusiones.
>
> **Una: el EDA mandó.** La señal dominante no era ninguna columna del CSV. Estaba escondida en
> el sufijo del título, entre paréntesis. La sacamos con una expresión regular y partió el dataset
> en tres grupos limpios. Todo lo que hicimos después sale de haberla encontrado.
>
> **Dos: la formulación valió más que el tamaño.** El modelo campeón tiene veintiséis mil
> parámetros. Lo que movió la aguja no fue agregar capacidad — la probamos y no alcanzó — sino
> decidir bien qué entra y cómo se codifica.
>
> **Tres: la atención aporta, y lo medimos con vara honesta.** Cuarenta y ocho milésimas contra
> su gemelo sin atención, con la misma entrada y las mismas semillas. Y contra el mejor MLP
> posible, que tiene cuatro veces y media más parámetros, veintisiete milésimas. Sigue ganando.
>
> **Cuatro: el mejor encoding fue el más simple.** Ordinal — un número por nivel, ordenado por su
> tasa de compra. Le ganó a embeddings, a target y a hashing. Y refutó nuestra propia hipótesis,
> que es exactamente lo que uno quiere de un experimento.
>
> **Y cinco: el modelo es auditable.** La atención y la importancia por permutación coinciden en
> qué mira. Las probabilidades están calibradas. Y en la práctica elige bien el producto a
> promocionar el noventa y uno por ciento de las veces, contra veintisiete del azar."

`→ en pantalla`: las cinco numeradas, cada una con su titular y su explicación. Leé el titular en
voz alta y contá la explicación con tus palabras — no la leas.

---

## 29 · Gracias — [5 s]

> "Eso es todo. Gracias, y quedamos para preguntas."

`→ en pantalla`: pantalla oscura, «Gracias» grande, «Preguntas» y los tres nombres. Queda ahí
mientras contestan.

---

# Apéndice para preguntas

No se presenta. Es para tener las respuestas a mano.

**¿Por qué PR-AUC y no F1?**
F1 necesita umbral; el uso es ranking. Igual lo medimos: F1 máximo 0.784 en umbral 0.40 — o sea
que 0.5 hubiera sido arbitrario.

**¿Por qué no ROC como principal?**
Con 13% de positivos el ROC oculta fallas. Ejemplo propio: el encoding por hashing tiene ROC 0.88
—suena razonable— y PR-AUC 0.498, que es un colapso.

**¿Hay overfitting?**
Curvas de train y validación por época en cada corrida. Early stopping por validación. Gap de
validación a test entre 0.01 y 0.03, que es selección normal. Y el encoding ordinal es en sí una
regularización.

**¿Por qué no validación cruzada desde el principio?**
La cátedra pidió priorizar promedio de corridas. Igual la hicimos al final: GroupKFold 5×6 da
0.821 ± 0.012, consistente con el 0.824.

**¿Y el split por producto?**
Verificado: métricas idénticas restringiendo test a productos nunca vistos. El modelo no recibe
identidad de producto.

**¿Usar el estado no es hacer trampa?**
Es información de estado del catálogo: está disponible al momento de predecir, así que es válida
para predecir. Sí es circular para decidir promociones. Por eso medimos las dos familias — sin
estado, el techo es 0.16 para cualquier modelo, porque el 61% de las filas tiene BTR cero exacto.

**¿Probaron multi-task con `cart`?**
Sí, como etiqueta auxiliar con λ en 0.1, 0.3 y 0.5. No es leakage porque el modelo la predice en
vez de recibirla. No mejoró: el mejor da 0.809 contra 0.824. Probablemente porque `bought` implica
`cart`, así que la tarea auxiliar no agrega información.

**¿Probaron regularización?**
Sí, barrido completo con 6 semillas: dropout, weight decay, feature dropout, label smoothing.
Ninguna aporta — sin dropout da 0.828, con dropout 0.1 da 0.824, sin nada 0.824, todo dentro de un
desvío de 0.02. La interpretación es que el early stopping ya evita el sobreajuste. Lo que sí se
ve es el daño del exceso: dropout 0.3 baja a 0.813.

**¿Probaron achicar más el modelo?**
Sí. `min_d16l1` —dimensión 16, un bloque, 3.713 parámetros— empata al campeón. Y destilando del
ensamble, un student de **1.937 parámetros** llega a 0.828. Compresión de 13×.

**¿Transfer desde un preentrenado de verdad?**
MiniLM. Congelado como token: 0.751. Congelado más MLP: 0.773. Fine-tuneado, con 22,7 millones de
parámetros: 0.811. Ninguno supera nuestros 26 mil parámetros en 0.824. La razón es que el dataset
es sintético y el wording no correlaciona con el comportamiento — "Top Rated" y "Highly Rated"
suenan iguales para BERT y difieren 30× en compra.

**¿Por qué 26k parámetros y no más?**
Porque lo medimos. La meseta de d_model va de 8 a 64 sin diferencias significativas, 4 bloques no
superan a 2, y la config con mejor **validación** de todo el proyecto (245k parámetros) NO es
mejor en test. Eso último es sobreajuste de selección, y es la razón por la que cerramos la
selección con un procedimiento fijado de antemano.
