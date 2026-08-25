# Genera presentacion.html: 20 diapositivas autocontenidas (imagenes embebidas).
# Navegacion: flechas / espacio / Home / End. Imprimir a PDF desde el navegador.
# El guion hablado con tiempos esta en guion.md.
import base64
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent.parent
sys.path.insert(0, str(REPO))

import zoo  # noqa: E402  (reutiliza el diagrama SVG de la arquitectura final)

GRAF = REPO / 'graficos'


def png(nombre):
    datos = base64.b64encode((GRAF / nombre).read_bytes()).decode()
    return f'data:image/png;base64,{datos}'


def barras_texto():
    """Mini grafico del arco textual (slide 14): barras horizontales inline."""
    datos = [('tabular puro (referencia)', 0.794, True), ('torre (texto→1 emb + MLP)', 0.775, False),
             ('fusión (resumen como token)', 0.775, False), ('híbrido sin regex', 0.711, False),
             ('híbrido (256 chars + features)', 0.705, False), ('texto puro (chars)', 0.652, False),
             ('techo SIN señal de estado', 0.162, False)]
    W, H, L = 760, len(datos) * 34 + 10, 300
    out = []
    for i, (lab, v, hi) in enumerate(datos):
        y = 8 + i * 34
        w = (W - L - 70) * v / 0.85
        color = '#0E9B7E' if hi else ('#C08312' if v < 0.3 else '#8FCBBC')
        out.append(f'<text x="{L - 10}" y="{y + 15}" text-anchor="end" class="bl">{lab}</text>')
        out.append(f'<rect x="{L}" y="{y}" width="{w:.0f}" height="22" rx="4" fill="{color}"/>')
        out.append(f'<text x="{L + w + 8:.0f}" y="{y + 15}" class="bv">{v:.3f}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" class="chart">' + ''.join(out) + '</svg>')


S = []  # cada item: (eyebrow, html)

S.append(('TP1 · 73.69 LARGE LANGUAGE MODELS', '''
<h1 class="titulo">Predicción de Buy Through Rate<br>con Transformers</h1>
<p class="sub">Formulación · EDA · un transformer sobre feature-tokens · 838 corridas de
experimentos · modelo final de 26k parámetros</p>
<p class="equipo">Fer Rossi · Junior Rambau · Matias Rossi Seifert · Juan Bautista Albertoni Salini</p>'''))

S.append(('EL PROBLEMA · FORMULACIÓN', '''
<h2>Qué se predice, exactamente</h2>
<div class="cols2">
<div>
<p><b>BTR</b> = compras / impresiones en la página de resultados — la probabilidad de que un
producto <em>mostrado</em> se compre. Objetivo de negocio: <b>qué promocionar</b>.</p>
<p>El dataset es un <b>log de eventos</b>: cada fila = un producto impreso en una búsqueda.</p>
<ul>
<li><b>Formulación</b>: clasificación binaria <b>por impresión</b> — el modelo estima
<code>p(bought | producto, búsqueda)</code></li>
<li><b>BTR por producto</b> = promedio de p sobre sus impresiones → ranking</li>
<li>Consecuencias: métricas de clasificación por fila · <b>sin umbral</b> (el uso es un ranking)</li>
</ul>
</div>
<div class="panel flujo">
<div class="paso">fila del CSV<br><span>producto × búsqueda</span></div>
<div class="flecha">→</div>
<div class="paso acc">modelo<br><span>p(bought) ∈ (0,1)</span></div>
<div class="flecha">→</div>
<div class="paso">agregar por producto<br><span>BTR estimado → ranking</span></div>
</div>
</div>'''))

S.append(('EDA · 1 DE 2', '''
<h2>La estructura — y la primera trampa</h2>
<div class="cols2">
<div>
<table class="t">
<tr><td>Impresiones (filas)</td><td class="n">10.000</td></tr>
<tr><td>Búsquedas (query_id)</td><td class="n">2.012 · de 1 a 8 productos</td></tr>
<tr><td>Positivos (bought)</td><td class="n">13,0%</td></tr>
<tr><td>Filtros de la búsqueda</td><td class="n">constantes por query · 100% cumplidos</td></tr>
</table>
<p class="nota">→ <code>filter_category</code> / <code>filter_storage_type</code> no informan a
nivel fila; el <b>rango de precio</b> sí — vía una feature derivada (slide 4).</p>
</div>
<div class="panel alerta">
<p class="grande"><code>bought ⟹ cart</code> en el <b>100%</b> de las filas</p>
<p><code>cart</code> es el funnel impresión → carrito → compra: <b>resultado del mismo
evento</b>, no disponible al decidir qué promocionar.</p>
<p><b>Leakage ⇒ afuera.</b> (p(bought | cart=False) = 0 exacto: el modelo degeneraría.)<br>
<span class="nota">Probado además como label auxiliar multi-task — nunca input —: no aporta.</span></p>
</div>
</div>'''))

S.append(('EDA · 2 DE 2', '''
<h2>La señal dominante está escondida en el texto</h2>
<div class="cols2">
<div>
<p>El título termina en un sufijo "( … )" en el 95% de las filas → <b>tiers de BTR brutales</b>:</p>
<table class="t">
<tr><th>sufijo del título</th><th>BTR</th></tr>
<tr><td>#1 Pick · Top Rated · Best Seller · Customer Favorite</td><td class="n ok">0.63–0.68</td></tr>
<tr><td>Popular Choice · Highly Rated · Shopper Favorite · Well Reviewed</td><td class="n">0.02–0.04</td></tr>
<tr><td>los 11 restantes · sin sufijo</td><td class="n mal">0.000 exacto</td></tr>
</table>
<p class="nota"><b>No es sentimiento</b>: "Highly Rated" suena igual que "Top Rated" y compra
50× menos — hay que aprender la partición exacta. La última oración de la descripción repite lo
mismo en prosa.</p>
</div>
<div>
<ul>
<li><b>Precio en U invertida</b> dentro del rango filtrado (0.39 → 0.86 → 0.43), condicionada al
tier → derivamos <code>price_rel</code>: posición del precio en el rango del usuario
(señal <em>relacional</em> producto × búsqueda)</li>
<li><b>Timestamps rotos</b>: hasta 2 años dentro de una misma búsqueda → sin split temporal, sin
features de tiempo (verificado: agregarlos empeora)</li>
<li><b>Redundancias</b>: package_size ≈ peso · dimensiones ≈ envase · descripción ≈ sufijo</li>
</ul>
</div>
</div>'''))

S.append(('FEATURES Y PREPROCESAMIENTO', '''
<h2>Qué entra al modelo, y cómo</h2>
<table class="t chica">
<tr><th>entrada</th><th>features</th><th>preprocesamiento</th></tr>
<tr><td><b>7 categóricas</b></td>
<td>listing_status <span class="tag">derivada: regex al sufijo</span> · category · brand ·
storage_type · unit_of_measure · country_of_origin · allergens</td>
<td>codificación = <b>experimento dedicado</b> (slide 13) · índice 0 = UNK para niveles no vistos</td></tr>
<tr><td><b>6 numéricas</b></td>
<td>price · price_rel <span class="tag">derivada: posición en el rango</span> · filter_price_min/max ·
net_weight_oz · nutrition_score</td>
<td>z-score con estadísticos de <b>train</b> · log1p previo en price y peso (sesgadas)</td></tr>
<tr><td><b>afuera</b></td>
<td>cart <span class="tag">leakage</span> · query_id <span class="tag">solo particiona</span> ·
timestamp <span class="tag">roto</span> · filtros redundantes · package / dimensions / ingredients
<span class="tag">redundantes</span></td>
<td><b>cada descarte verificado por ablación</b>: reintroducirlos midió Δ ≈ 0</td></tr>
</table>
<p class="nota">Todo se ajusta SOLO con train (vocabularios, estadísticos, tablas de encoding).</p>'''))

S.append(('LA DISCUSIÓN CONCEPTUAL', '''
<h2>¿Es válido usar "Best Seller"? → dos familias</h2>
<div class="cols2">
<div>
<p>Los badges se asignan <em>después</em> de vender mucho — ¿circular? Tres clases de información:</p>
<ul>
<li><b>Outcome del mismo evento</b> (cart) → leakage estricto, nunca</li>
<li><b>Estado al momento de la impresión</b> (badges) → disponible al predecir, plausiblemente
causal (prueba social) · pero circular para promover y ciego al producto nuevo</li>
<li><b>Atributos intrínsecos</b> → válidos siempre</li>
</ul>
</div>
<div class="panel">
<p class="grande">Familia <b>catálogo</b> (con estado): <span class="ok">PR-AUC ~0.82</span><br>
Familia <b>intrínseca</b> (sin estado, ni parseado ni en el texto): <span class="mal">~0.16</span></p>
<p class="nota">Sin la señal de estado <b>nadie</b> pasa de 0.16 — ni el GBM: el 61% de las filas
vive en tiers de BTR = 0.000. No es un bug del modelo: es un hallazgo sobre el dataset.
Responden preguntas distintas: <em>qué promocionar hoy</em> vs <em>qué esperar de un producto
nuevo</em>.</p>
</div>
</div>'''))

S.append(('MÉTRICAS Y PROTOCOLO', '''
<h2>Cómo medimos y cómo particionamos</h2>
<div class="cols2">
<div>
<p><b>Métricas</b> (sugerencia del enunciado, adoptada con motivo):</p>
<ul>
<li><b>PR-AUC principal</b>: 13% de positivos → accuracy es inútil; azar = 0.131</li>
<li><b>ROC-AUC</b> complementaria · log-loss y Brier (calidad de las probabilidades)</li>
<li><b>Sin umbral</b> — el uso es ranking; el F1 máximo cae en umbral ~0.40, no 0.5:
cualquier umbral fijo hubiera sido arbitrario</li>
<li>Se guardan <b>las 16 métricas</b> por época y corrida → over/underfitting visible en las
curvas train/val (early stopping por PR-AUC de <b>validación</b>, paciencia 20)</li>
</ul>
</div>
<div>
<p><b>Partición</b>: por <code>query_id</code> 70/15/15 — una búsqueda entera del mismo lado.</p>
<ul>
<li>¿por fila? filtraría información de la página · ¿temporal? timestamps rotos ·
¿por producto? verificado: métricas idénticas en productos jamás vistos</li>
<li><b>Promedio de 6 seeds</b> por configuración (lo priorizado por la cátedra) ·
GroupKFold 5×6 al cierre: 0.821 ± 0.012, consistente</li>
<li><b>Disciplina</b>: hiperparámetros con validación; test solo se reporta</li>
</ul>
</div>
</div>'''))

S.append(('ARQUITECTURA · DÓNDE VA EL TRANSFORMER Y POR QUÉ', f'''
<h2>Cada feature es un token (FT-Transformer)</h2>
<div class="cols2 ancha">
<div>
<ul>
<li>La señal es <b>relacional</b> (precio ↔ rango del filtro · precio × tier) → la self-attention
computa interacciones <b>de a pares, aprendidas</b> — la generalización de las feature crosses</li>
<li>Los tokens no necesitan ser "del mismo tipo": necesitan vivir en el mismo espacio
<b>ℝ<sup>d</sup></b> — eso hace el tokenizador (numéricas: x·w+b; categóricas: su codificación)</li>
<li><b>[CLS]</b> = token de lectura (no aporta información: la recolecta — BERT, clase 2)</li>
<li><b>Sin positional</b>: un conjunto de features no tiene orden — la identidad de cada columna
vive en sus parámetros propios (medido: agregar PE da Δ ≈ 0)</li>
<li>Mismos bloques que la demo de la cátedra + 2 adaptaciones: <b>sin máscara causal</b>
(clasificación, no generación) · escalado por √d<sub>k</sub> <b>una</b> sola vez (el paper)</li>
<li>Tamaño inicial según el enunciado: d_model = 32 (&lt; 100), 4 cabezas, 2 bloques</li>
</ul>
</div>
<div class="figzoo">{zoo.FIGS['feat']}</div>
</div>'''))

S.append(('ALTERNATIVAS CONSIDERADAS', '''
<h2>Dónde MÁS podía ir el transformer — y por qué no como base</h2>
<table class="t">
<tr><th>alternativa</th><th>la idea</th><th>por qué no como base</th></tr>
<tr><td><b>Texto crudo como tokens</b><br><span class="nota">la demo literal</span></td>
<td>caracteres de título+descripción; que descubra la señal solo</td>
<td>la señal se extrae con una regex; atención 257² para eso es cara</td></tr>
<tr><td><b>Productos de la página como tokens</b><br><span class="nota">listwise</span></td>
<td>modelar la competencia dentro de la búsqueda</td>
<td>el EDA midió competencia débil (se compran varios por página)</td></tr>
<tr><td><b>Transformer solo como encoder de texto</b> + MLP</td>
<td>el transformer produce un embedding; clasifica un MLP</td>
<td>comprime la señal a un cuello de botella antes de decidir</td></tr>
</table>
<p class="destacado">No las descartamos en papel: <b>las implementamos y corrimos todas</b>
(más 2 variantes). Los resultados — slide 17 — les dan la razón a los diagnósticos del EDA.</p>'''))

S.append(('BASELINES', '''
<h2>La vara, antes del transformer</h2>
<div class="filaNums">
<div class="num"><span>logística</span><b>0.698</b><i>la vara lineal</i></div>
<div class="num"><span>MLP (mismos embeddings)</span><b>0.746</b><i>no-lineal sin atención</i></div>
<div class="num"><span>GBM</span><b>0.762</b><i>árboles con interacciones</i></div>
<div class="num azar"><span>azar</span><b>0.131</b><i>tasa base</i></div>
</div>
<p class="destacado">Regla de honestidad: si el transformer no supera esto,
<b>la capa de atención no se justifica</b>. Mismo split, mismas métricas, mismos 6 seeds.</p>'''))

S.append(('EXPERIMENTO 1', '''
<h2>¿La atención aporta?</h2>
<div class="cols2">
<div>
<p>Transformer vs MLP <b>con la misma entrada</b> (los mismos feature-tokens; lo único que
cambia es qué los mezcla), apareado por seed:</p>
<p class="grande">Δ = <b class="ok">+0.048</b> · gana <b>5/6</b> seeds<br>
<span class="nota">con el MLP teniendo 4,5× más parámetros (126k vs 28k)</span></p>
</div>
<div>
<p><b>Dos refinamientos honestos:</b></p>
<ul>
<li>al MLP le probamos one-hot crudo: mejora a 0.797 (parte de su déficit era la entrada) →
contra el <b>mejor</b> MLP, la ventaja es <b>+0.027</b> — gana igual, con la vara más alta</li>
<li>¿solo "descubrió" el cruce precio×tier del EDA? Se lo dimos a mano a la logística: +0.015,
pero eso explica <b>solo el 12% del gap</b> — la atención aprende bastante más que ese cruce</li>
</ul>
</div>
</div>'''))

S.append(('EXPERIMENTO 2 · CABEZAS', f'''
<h2>¿Cuántas cabezas de atención?</h2>
<div class="cols2 ancha">
<div><img src="{png('decision_cabezas.png')}" alt="cabezas de atencion"></div>
<div>
<ul>
<li><b>Por qué variarlo</b>: multi-head = varias "consultas" en subespacios distintos, en
paralelo. ¿Este problema las necesita, o hay UNA señal dominante?</li>
<li>Con <b>embeddings</b>: 1 cabeza gana (0.816 vs 0.798) — una consulta rica &gt; cuatro
pobres, coherente con el EDA</li>
<li>Con <b>ordinal</b> (la base final): 4 cabezas 0.824 &gt; 1 cabeza 0.800</li>
</ul>
<p class="destacado">El eje <b>interactúa con el encoding</b> → ninguna decisión se hereda de
otra base: se re-decide por validación sobre la base final.</p>
</div>
</div>'''))

S.append(('EXPERIMENTO 3 · BLOQUES', f'''
<h2>¿Cuánta profundidad?</h2>
<div class="cols2 ancha">
<div><img src="{png('decision_bloques.png')}" alt="bloques"></div>
<div>
<ul>
<li><b>Por qué variarlo</b>: más bloques = componer atención sobre atención. Con 13 features
que ya se ven todas a un salto, ¿hace falta?</li>
<li>1 bloque pierde poco (0.801) · <b>2 ganan (0.824)</b> · 4 no suman (0.811, con 2× los
parámetros)</li>
<li>Coherente con la interpretabilidad (slide 21): el CLS ya concentra <b>0.75 de su atención
en el status en la capa 1</b></li>
</ul>
<p class="destacado">La profundidad no era el eje. Paciencia del early stopping sí ayudó:
8→20 mejora <b>21/24</b> configs tabulares.</p>
</div>
</div>'''))

S.append(('EXPERIMENTO 4 · D_MODEL', f'''
<h2>¿Qué dimensión de embedding?</h2>
<div class="cols2 ancha">
<div><img src="{png('decision_dmodel.png')}" alt="d_model"></div>
<div>
<ul>
<li><b>Por qué variarlo</b>: dimensionar al problema (13 features, 10k filas), no al hábito
(los 512 de los papers)</li>
<li><b>Meseta amplia</b> d8→d64: la señal cabe en poquísimos parámetros; d32 se elige por
validación</li>
<li>Epílogo (7ª tanda): <b>d16 con 1 bloque (3.713 params) empata al campeón</b>, y destilando
del ensemble el nivel campeón aguanta hasta <b>1.937</b> — curva completa en backup</li>
</ul>
<p class="destacado">Lección de los tres ejes de capacidad → el experimento decisivo no era
capacidad: era la <b>codificación de la entrada</b>.</p>
</div>
</div>'''))

S.append(('EXPERIMENTO 5 · EL DECISIVO', f'''
<h2>Codificación de las categóricas</h2>
<div class="cols2 ancha">
<div>
<img src="{png('decision_encoding.png')}" alt="encodings de categoricas">
<p class="nota">one-hot + proyección lineal ≡ embedding (misma matriz) → para el transformer no
es una opción distinta; por eso se prueba solo como entrada cruda al MLP.</p>
</div>
<div>
<ul>
<li><b>Por qué gana ordinal</b>: en 10k filas, "poder aprender cualquier cosa" es overfitting —
el rango inyecta como <em>prior</em> el orden nivel→propensión, con un escalar por nivel
(26k parámetros totales)</li>
<li><b>Por qué le gana a target</b>: rangos equiespaciados mejor condicionados que magnitudes
apelmazadas (0.65 / 0.03 / 0.000)</li>
<li><b>Los contraejemplos calibran</b>: freq y hashing destruyen la relación nivel→propensión —
la regla es preservarla</li>
<li><b>Reflexión</b>: nuestra hipótesis era la inversa (embedding ≥ target ≥ ordinal) —
los datos nos corrigieron, y esa corrección es el mejor modelo del TP</li>
</ul>
</div>
</div>'''))

S.append(('EXPERIMENTO 6 · INITS', f'''
<h2>¿Arrancar de pesos informados? (algoritmos de embedding)</h2>
<div class="cols2 ancha">
<div><img src="{png('decision_init.png')}" alt="inits y pre-entrenamiento"></div>
<div>
<ul>
<li><b>Por qué variarlo</b>: la promesa del pre-entrenamiento (clase 3) — ¿un init informado
le gana al aleatorio?</li>
<li><b>MLM estilo BERT</b> sobre los feature-tokens (20 épocas, sin labels): +0.011 sobre
embeddings; sobre <b>ordinal, no</b> — el prior ya hace ese trabajo</li>
<li><b>w2v-init</b> (skipgram propio) regulariza words (+0.010) sin alcanzar a chars</li>
</ul>
<p class="destacado">Init aleatoria + prior ordinal &gt; pre-entrenar, a esta escala.
<span class="nota">(8ª tanda, backup: MiniLM externo congelado RESTA; fine-tuneado repara —
y aun así no supera 0.824.)</span></p>
</div>
</div>'''))

S.append(('EXPERIMENTO 7 · TEXTO', f'''
<h2>¿Y el texto? Recuperable — pero redundante</h2>
<div class="cols2 ancha">
<div>{barras_texto()}</div>
<div>
<ul>
<li><b>Texto puro 0.652</b>: encontró el sufijo solo (≫ 0.16) pero leer chars con 36k parámetros
no alcanza al tabular</li>
<li><b>Híbrido 0.705</b>: 256 tokens de texto <em>diluyen</em> a los 13 que importan — pero
sacarle el token parseado no le cuesta nada: <b>recupera la regex desde los chars</b></li>
<li><b>Torre 0.775</b>: la mejor textual, pero su embedding único no deja pasar la señal entera
(sin regex: −0.04)</li>
<li><b>Fusión</b> (el texto comprimido a UN token): cura la dilución (+0.069, 6/6) y empata con
la torre → comprimir importa, dónde cruzar no</li>
<li>Tokenizador: words ≈ chars; w2v pre-entrenado regulariza (+0.010) — <b>el tokenizador chico
de la demo era el correcto</b> a esta escala (clases 1↔2)</li>
</ul>
</div>
</div>'''))

S.append(('DESAFÍOS ENCONTRADOS', '''
<h2>Cuatro que valen la pena contar</h2>
<div class="cols2">
<div>
<p><b>1 · El causal degenerado.</b> La ablación causal dio ROC = 0.500 <em>exacto</em>.
Diagnóstico: con máscara causal, el [CLS] en la posición 0 solo se ve a sí mismo — predecía una
constante (p = 0.2214 en todo test). Fix: CLS al final, como lee GPT. Respuesta real: la
bidireccionalidad acá <b>da igual</b>. Lección: los decoders leen desde el último token.</p>
<p><b>2 · Las trampas del dataset.</b> cart (leakage del funnel) y timestamps rotos — el EDA los
cazó antes de que muerdan.</p>
</div>
<div>
<p><b>3 · El cómputo.</b> La familia de texto es inviable en CPU (atención 257²) → suite de
experimentos <em>resumible</em> en GPU: 838 corridas, cada una con sus 16 métricas por época.</p>
<p><b>4 · Hipótesis refutadas con datos</b> (el método importa): bins por cuantiles para la U
(el FFN ya la captura) · pos_weight (daña: PR-AUC es de ranking) · mean pooling (CLS gana 6/6) ·
positional en features (Δ ≈ 0, como predice la teoría) · y la hipótesis del encoding (slide 13).</p>
</div>
</div>'''))

S.append(('EL MODELO FINAL', '''
<h2>FT-Transformer con encoding ordinal — 26.177 parámetros</h2>
<div class="cols2">
<div class="panel">
<p><b>Arquitectura</b>: 13 feature-tokens (7 ordinales + 6 numéricas afines) + [CLS] · d_model 32
· 4 cabezas · 2 bloques pre-LN · sin positional · Linear(32→1) → sigmoide · BCE</p>
<p><b>Entrenamiento</b>: AdamW 1e-3 · batch 256 · early stopping por PR-AUC de val, paciencia 20
· ~64 épocas · 6 seeds</p>
<p><b>Elección disciplinada</b>: empate técnico en <b>validación</b> entre 4 configs (Δ&lt;0.002)
→ desempate por <b>parsimonia</b> (menos parámetros, menor desvío, menor gap val→test) →
test lo confirma después</p>
</div>
<div class="filaNums vert">
<div class="num"><span>PR-AUC test (6 seeds)</span><b class="ok">0.824 ± 0.018</b></div>
<div class="num"><span>GroupKFold 5×6</span><b>0.821 ± 0.012</b></div>
<div class="num"><span>ensemble (2 rutas independientes)</span><b>0.834</b></div>
<div class="num"><span>ROC 0.975 · F1 máx 0.784 @ 0.40 · GBM 0.762 · mejor MLP 0.797</span></div>
</div>
</div>'''))

S.append(('ROBUSTEZ', f'''
<h2>Tres verificaciones sobre el modelo final</h2>
<div class="cols2 ancha">
<div><img src="{png('curva_aprendizaje.png')}" alt="curva de aprendizaje"></div>
<div>
<ul class="espaciada">
<li><b>Curva de aprendizaje</b>: casi saturada — el último 25% de datos aporta +0.007; con el
<b>75%</b> de los datos ya supera al GBM entrenado con todo</li>
<li><b>Varianza</b> (grilla 5 inits × 6 splits): el ±0.018 es <b>~55% split / ~45%
inicialización</b> → valida promediar seeds; la mitad de init es lo que el ensemble elimina
(+0.010)</li>
<li><b>Calibración</b>: ECE ~0.01, temperatura ≈ 1 → el promedio de p por producto
<b>es</b> un BTR confiable, sin corrección</li>
</ul>
</div>
</div>'''))

S.append(('INTERPRETABILIDAD', f'''
<h2>¿El modelo mira donde debe? Sí — y por dos vías independientes</h2>
<div class="cols2 ancha">
<div><img src="{png('atencion_feat_ordinal_features_d32_h4_l2_linear_catordinal_seed46.png')}"
alt="mapa de atencion"></div>
<div>
<img src="{png('importancia.png')}" alt="importancia por permutacion" style="max-height:38vh">
<ul>
<li><b>Atención</b> (capa 1): el CLS pone <b>0.75</b> en <code>status</code>; price y f_min
consultan a <code>p_rel</code> — la señal relacional, literal en el mapa</li>
<li><b>Permutación</b> (outcome): status +0.68 · p_rel +0.14 · allergens +0.05 · resto ≈ 0 —
"attention is not explanation" respondido: dos diagnósticos, la misma historia (la del EDA)</li>
<li><b>Negocio</b>: el top-1 de cada página fue comprado el <b>91%</b> de las veces (azar: 27%)</li>
</ul>
</div>
</div>'''))

S.append(('EJERCICIO 3 · TEÓRICO', '''
<h2>Personalización: p(bought | producto, búsqueda, <u>usuario</u>)</h2>
<div class="cols2">
<div>
<p><b>Dato nuevo necesario</b>: <code>user_id</code> + historial de eventos (el dataset no lo trae).</p>
<p><b>Extensión natural de nuestra arquitectura</b>: el historial del usuario entra como
<b>tokens</b> — sus últimas compras/búsquedas, codificadas con el mismo tokenizador de
productos — y la secuencia los atiende (<b>cross-attention</b>, estilo BST/SASRec).
El usuario que compra comida de perro cada mes: sus compras están en el historial → cuando esos
productos aparecen en la página, la atención historial↔candidato sube su p.</p>
</div>
<div>
<p><b>Alternativa clásica (clase 2)</b>: embeddings de usuario y producto con <b>negative
sampling</b> (item2vec / two-tower — skipgram donde el contexto son los productos del usuario) —
útil como <em>retrieval</em> en catálogos enormes, con nuestro modelo como <em>ranker</em>.</p>
<p><b>Cold-start</b>: usuario sin historial = producto sin estado (lo medimos: familia
intrínseca) → fallback exacto: el modelo actual, que no depende del usuario.</p>
</div>
</div>'''))

S.append(('CONCLUSIONES', '''
<h2>Cinco, cortas</h2>
<ul class="espaciada grande2">
<li><b>El EDA mandó</b>: la señal estaba escondida en un sufijo de texto — todo el diseño sale
de haberla encontrado</li>
<li><b>La formulación correcta valió más que el modelo grande</b>: el campeón tiene 26k parámetros</li>
<li><b>La atención aporta, medida con vara honesta</b>: +0.048 vs su gemelo sin atención ·
+0.027 vs el mejor MLP</li>
<li><b>El mejor encoding fue un prior simple</b>: el ordinal derivado de los datos refutó
nuestra propia hipótesis — eso es un experimento funcionando</li>
<li><b>El modelo es auditable</b>: atención e importancia convergen · calibrado (el promedio de
p ES el BTR) · elige bien el producto a promocionar el 91% de las veces</li>
</ul>
<p class="cierreNum">PR-AUC <b>0.824 ± 0.018</b> · ensemble <b>0.834</b> · GBM 0.762 — gracias.</p>'''))

S.append(('APÉNDICE · BACKUP (NO SE PRESENTA)', f'''
<h2>Overfitting / underfitting: las curvas del modelo final</h2>
<div class="cols2 ancha">
<div><img src="{png('curvas_entrenamiento.png')}" alt="curvas train/val"></div>
<div>
<ul class="espaciada">
<li>Train sube sostenido; <b>validación se aplana</b> y el early stopping corta en su máximo
(línea punteada) con paciencia 20 — se restaura ese checkpoint</li>
<li>El gap train/val final es moderado y estable — sin underfitting (supera todas las varas) ni
overfitting descontrolado (dropout 0.1 + el encoding ordinal actúa de regularizador)</li>
<li>Estas curvas existen <b>para cada una de las 838 corridas</b> (16 métricas por época,
explorables en el panel interactivo del repo)</li>
<li>Gap val→test del modelo final: 0.011 — el menor del top-4 (selección sana)</li>
</ul>
</div>
</div>'''))


# ---------------------------------------------------------------- html

ZOO_SVG_CSS = """
.figzoo{--tab-bg:#DFF0EB;--tab-br:#1C8A76;--tab-tx:#0B5B4C;--num-bg:#F0F8F5;--num-br:#66B4A2;
--num-tx:#166553;--txt-bg:#F8ECCB;--txt-br:#B58117;--txt-tx:#7C5709;--hot-bg:#F2D592;
--hot-br:#8A5F06;--hot-tx:#6B4A05;--cls-bg:#E7DEF9;--cls-br:#7052C9;--cls-tx:#4A338F;
--pad-br:#9FACB9;--pad-tx:#7D8C9B;--blk-bg:#EDF1F6;--blk-br:#64778F;--mlp-bg:#DCE9F8;
--mlp-br:#2E68AC;--mlp-tx:#1C4B85;--prod-bg:#1C8A76;--prod-tx:#FFF;--out:#C22B5E;--out-tx:#FFF;
--card:#FFF;--figbg:#F8FAFC;--boxbr:#8A99A9;--ink:#1B2530;--ink2:#3D4B5A;--muted:#5F7183;
--arrow:#8E9CAA;--line:#D8DFE7}
.figzoo svg{width:100%;height:auto;max-height:74vh}
.figzoo text{font-family:var(--sans)}
.figzoo .b-title{font:600 12.5px var(--sans);fill:var(--ink)}
.figzoo .b-line{font:11px var(--sans);fill:var(--ink2)}
.figzoo .shape{font:10.5px var(--mono);fill:var(--muted)}
.figzoo .bx{stroke-width:1.3}
.figzoo .bx-data{fill:var(--card);stroke:var(--boxbr)}
.figzoo .bx-proc{fill:var(--figbg);stroke:var(--boxbr)}
.figzoo .bx-tok{fill:var(--tab-bg);stroke:var(--tab-br)}
.figzoo .bx-txt{fill:var(--txt-bg);stroke:var(--txt-br)}
.figzoo .bx-blk{fill:var(--blk-bg);stroke:var(--blk-br);stroke-width:1.6}
.figzoo .bx-mlp{fill:var(--mlp-bg);stroke:var(--mlp-br)}
.figzoo .bx-cls{fill:var(--cls-bg);stroke:var(--cls-br)}
.figzoo .badge{fill:var(--card);stroke:var(--blk-br);stroke-width:1.2}
.figzoo .badge-t{font:700 11px var(--mono);fill:var(--ink)}
.figzoo .tk{stroke-width:1.3}.figzoo .tkt{font:9.5px var(--mono)}
.figzoo .k-tab{fill:var(--tab-bg);stroke:var(--tab-br)}.figzoo .k-tab-t{fill:var(--tab-tx)}
.figzoo .k-num{fill:var(--num-bg);stroke:var(--num-br)}.figzoo .k-num-t{fill:var(--num-tx)}
.figzoo .k-chr{fill:var(--txt-bg);stroke:var(--txt-br)}.figzoo .k-chr-t{fill:var(--txt-tx)}
.figzoo .k-hot{fill:var(--hot-bg);stroke:var(--hot-br)}.figzoo .k-hot-t{fill:var(--hot-tx)}
.figzoo .k-cls{fill:var(--cls-bg);stroke:var(--cls-br)}.figzoo .k-cls-t{fill:var(--cls-tx);font-weight:700}
.figzoo .k-pad{fill:none;stroke:var(--pad-br);stroke-dasharray:4 3}.figzoo .k-pad-t{fill:var(--pad-tx)}
.figzoo .k-out{fill:var(--out)}.figzoo .k-out-t{fill:var(--out-tx);font-weight:700}
.figzoo .k-ell-t{fill:var(--muted);font-size:13px}
.figzoo .flow{stroke:var(--arrow);stroke-width:1.5;fill:none}
.figzoo .arrhead{fill:var(--arrow)}.figzoo .brk{stroke:var(--arrow);stroke-width:1.1;fill:none}
.figzoo .signode{fill:var(--card);stroke:var(--out);stroke-width:1.6}
.figzoo .sigt{font:italic 700 13px Georgia,serif;fill:var(--out)}
.figzoo .outpill{fill:var(--out)}.figzoo .outt{font:600 12px var(--mono);fill:var(--out-tx)}
"""

CSS = """
:root{--bg:#F4F7F6;--card:#FFF;--ink:#17242C;--ink2:#3D4B5A;--muted:#5F7183;--line:#D8DFE7;
--acc:#0E9B7E;--acc2:#7052C9;--warn:#C08312;--mal:#D42A63;
--sans:"Segoe UI","Noto Sans","Liberation Sans",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"Cascadia Code","Fira Mono","DejaVu Sans Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
html,body{margin:0;background:#20302C;font:16px/1.5 var(--sans);color:var(--ink)}
.slide{display:none;width:100vw;height:100vh;padding:4.5vh 5vw 6vh;background:var(--bg);
flex-direction:column;overflow:hidden}
.slide.activa{display:flex}
.eyebrow{font:700 12px var(--mono);letter-spacing:.18em;color:var(--acc);margin-bottom:1.2vh}
h1.titulo{font-size:5.2vh;line-height:1.15;margin:9vh 0 2vh;letter-spacing:-.01em}
h2{font-size:3.9vh;line-height:1.15;margin:0 0 2.6vh;letter-spacing:-.01em}
.sub{font-size:2.4vh;color:var(--ink2);max-width:70ch}
.equipo{margin-top:auto;font:600 2vh var(--mono);color:var(--muted)}
p,li{font-size:2.15vh;color:var(--ink2)}
li{margin:.9vh 0}
b{color:var(--ink)}
code{font:.92em var(--mono);background:#E9F0EE;border-radius:4px;padding:0 5px}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:3.5vw;align-items:start;min-height:0}
.cols2.ancha{grid-template-columns:1.15fr .85fr}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:2.4vh 1.6vw}
.panel.alerta{border-left:5px solid var(--mal)}
.grande{font-size:2.5vh}
.grande2 li{font-size:2.35vh}
.nota{font-size:1.75vh;color:var(--muted)}
.destacado{margin-top:2.5vh;font-size:2.3vh;background:#E4F2EE;border-left:5px solid var(--acc);
border-radius:8px;padding:1.6vh 1.4vw;color:var(--ink)}
.ok{color:var(--acc)} .mal{color:var(--mal)}
table.t{border-collapse:collapse;width:100%;background:var(--card);border-radius:12px;overflow:hidden}
table.t th{font:700 1.6vh var(--mono);letter-spacing:.08em;text-transform:uppercase;
color:var(--muted);text-align:left;padding:1.2vh 1vw;border-bottom:2px solid var(--line)}
table.t td{padding:1.25vh 1vw;border-bottom:1px solid var(--line);font-size:1.95vh;color:var(--ink2)}
table.t.chica td{font-size:1.8vh;padding:1vh 1vw}
table.t td.n{font-family:var(--mono);white-space:nowrap;color:var(--ink)}
table.t tr.hl td{background:#E4F2EE;color:var(--ink)}
.tag{font:600 1.4vh var(--mono);background:#EDE9F8;color:var(--acc2);border-radius:99px;
padding:.2vh .6vw;white-space:nowrap}
.flujo{display:flex;align-items:center;gap:1vw;justify-content:center;height:100%}
.paso{background:var(--card);border:1.5px solid var(--line);border-radius:12px;padding:2vh 1.2vw;
text-align:center;font-size:2vh;font-weight:600}
.paso span{display:block;font:1.6vh var(--mono);color:var(--muted);font-weight:400;margin-top:.6vh}
.paso.acc{border-color:var(--acc);background:#E4F2EE}
.flecha{font-size:3vh;color:var(--muted)}
.filaNums{display:flex;gap:1.5vw;margin:3vh 0}
.filaNums.vert{flex-direction:column;margin:0}
.num{flex:1;background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:2.2vh 1.4vw;text-align:center}
.filaNums.vert .num{text-align:left;padding:1.6vh 1.4vw}
.num span{display:block;font:700 1.55vh var(--mono);letter-spacing:.1em;text-transform:uppercase;
color:var(--muted)}
.num b{font-size:4vh;color:var(--ink)}
.filaNums.vert .num b{font-size:3.2vh}
.num b.ok{color:var(--acc)}
.num i{display:block;font-style:normal;font-size:1.7vh;color:var(--muted)}
.num.azar{opacity:.65}
.cierreNum{margin-top:auto;font-size:2.8vh;color:var(--ink)}
.espaciada li{margin:1.6vh 0}
img{max-width:100%;max-height:62vh;border-radius:10px;border:1px solid var(--line);background:#fff}
svg.chart{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}
svg.chart .bl{font:1.9vh var(--sans);fill:#3D4B5A}
svg.chart .bv{font:700 1.9vh var(--mono);fill:#17242C}
#hud{position:fixed;bottom:1.6vh;right:1.6vw;font:700 1.8vh var(--mono);color:#fff;opacity:.55;z-index:9}
#barra{position:fixed;bottom:0;left:0;height:.7vh;background:var(--acc);z-index:9;transition:width .2s}
@media print{
 html,body{background:#fff}
 .slide{display:flex !important;page-break-after:always;width:100%;height:100vh}
 #hud,#barra{display:none}
}
""" + ZOO_SVG_CSS

JS = """
const slides=[...document.querySelectorAll('.slide')];let i=0;
function ir(n){i=Math.max(0,Math.min(slides.length-1,n));
slides.forEach((s,j)=>s.classList.toggle('activa',j===i));
document.getElementById('hud').textContent=(i+1)+' / '+slides.length;
document.getElementById('barra').style.width=(100*(i+1)/slides.length)+'vw';
location.hash='s'+(i+1);}
addEventListener('keydown',e=>{
 if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();ir(i+1);}
 if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();ir(i-1);}
 if(e.key==='Home')ir(0); if(e.key==='End')ir(slides.length-1);});
addEventListener('click',e=>{if(e.clientX>innerWidth*.66)ir(i+1);
 else if(e.clientX<innerWidth*.33)ir(i-1);});
ir(parseInt((location.hash.match(/s(\\d+)/)||[0,1])[1],10)-1||0);
"""


def main():
    cuerpo = ''.join(
        f'<section class="slide"><div class="eyebrow">{i + 1:02d} · {eyebrow}</div>{html}</section>'
        for i, (eyebrow, html) in enumerate(S))
    pagina = (f'<title>BTR con Transformers</title>\n<style>{CSS}</style>\n'
              f'{cuerpo}\n<div id="hud"></div><div id="barra"></div>\n<script>{JS}</script>\n')
    out = AQUI / 'presentacion.html'
    out.write_text(pagina)
    print(f'escrito {out.name} ({len(pagina) / 1024:.0f} KB, {len(S)} diapositivas)')


if __name__ == '__main__':
    main()
