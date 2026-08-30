"""Genera zoo.html: los diagramas SVG de las arquitecturas (el "Zoo").

    .venv/bin/python zoo.py    # escribe zoo.html (publicable como artifact)

SVG generado programaticamente para que todas las figuras compartan las mismas
metricas y el mismo codigo de color (teal=tabular, ambar=texto, violeta=CLS,
azul=MLP/lineal, frambuesa=salida). Los numeros de resultados citados en tabla y
epigrafes se actualizan A MANO al analizar cada tanda (ver analisis.md); las
dimensiones de los modelos son las reales del codigo.

(Antes este generador vivia en el scratchpad de la sesion y se perdio en una
limpieza; por eso ahora esta versionado en el repo.)
"""
import html as _html
from xml.etree import ElementTree

E = _html.escape


class Fig:
    def __init__(self, fid, w=840):
        self.fid = fid
        self.w = w
        self.cx = w / 2
        self.parts = []
        self.maxy = 0.0

    def _up(self, y):
        self.maxy = max(self.maxy, y)

    def el(self, tag, **attrs):
        a = ' '.join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        self.parts.append(f'<{tag} {a}/>')

    def text(self, x, y, s, cls, anchor='middle'):
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{E(s)}</text>')
        self._up(y + 4)

    def box(self, y, title=None, lines=(), cx=None, w=320, cls='bx-data', badge=None):
        cx = self.cx if cx is None else cx
        pad, lh, th = 11, 15.5, (18 if title else 0)
        gap = 5 if (title and lines) else 0
        h = pad * 2 + th + gap + len(lines) * lh + (0 if lines else (0 if title else 8))
        x = cx - w / 2
        self.el('rect', x=f'{x:.1f}', y=f'{y:.1f}', width=w, height=f'{h:.1f}', rx=9,
                **{'class': f'bx {cls}'})
        cy = y + pad
        if title:
            self.text(cx, cy + 12, title, 'b-title')
            cy += th + gap
        for i, ln in enumerate(lines):
            self.text(cx, cy + 11 + i * lh, ln, 'b-line')
        if badge:
            bx, by = x + w - 16, y
            self.el('rect', x=f'{bx - 16:.1f}', y=f'{by - 9:.1f}', width=32, height=18, rx=9,
                    **{'class': 'badge'})
            self.text(bx, by + 4.5, badge, 'badge-t')
        bot = y + h
        self._up(bot)
        return {'x': x, 'y': y, 'w': w, 'h': h, 'cx': cx, 'bot': bot}

    def tokens(self, y, cells, cx=None, h=30, gap=4):
        # cells: (label, kind, width)
        cx = self.cx if cx is None else cx
        total = sum(c[2] for c in cells) + gap * (len(cells) - 1)
        x = cx - total / 2
        placed = []
        for label, kind, w in cells:
            if kind == 'k-ell':
                self.text(x + w / 2, y + h / 2 + 4, label, 'tkt k-ell-t')
            else:
                self.el('rect', x=f'{x:.1f}', y=f'{y:.1f}', width=w, height=h, rx=6,
                        **{'class': f'tk {kind}'})
                self.text(x + w / 2, y + h / 2 + 3.5, label, f'tkt {kind}-t')
            placed.append({'x': x, 'w': w, 'kind': kind})
            x += w + gap
        bot = y + h
        self._up(bot)
        return {'x': cx - total / 2, 'y': y, 'w': total, 'h': h, 'cx': cx,
                'bot': bot, 'cells': placed}

    def arrow(self, y, dy=36, labels=(), x=None):
        x = self.cx if x is None else x
        self.parts.append(
            f'<line x1="{x:.1f}" y1="{y + 2:.1f}" x2="{x:.1f}" y2="{y + dy - 2:.1f}" '
            f'class="flow" marker-end="url(#arr-{self.fid})"/>')
        n = len(labels)
        mid = y + dy / 2
        for i, lab in enumerate(labels):
            self.text(x + 12, mid + (i - (n - 1) / 2) * 13.5 + 3.5, lab, 'shape', anchor='start')
        self._up(y + dy)
        return y + dy

    def bracket(self, x1, x2, y, label, above=False, cls='shape'):
        if above:
            self.parts.append(f'<path d="M{x1:.1f} {y:.1f} v-4 H{x2:.1f} v4" class="brk"/>')
            self.text((x1 + x2) / 2, y - 9, label, cls)
        else:
            self.parts.append(f'<path d="M{x1:.1f} {y:.1f} v4 H{x2:.1f} v-4" class="brk"/>')
            self.text((x1 + x2) / 2, y + 17, label, cls)
            self._up(y + 21)

    def strike(self, cell, row):
        x, w, y, h = cell['x'], cell['w'], row['y'], row['h']
        self.parts.append(
            f'<line x1="{x + 4:.1f}" y1="{y + 4:.1f}" x2="{x + w - 4:.1f}" y2="{y + h - 4:.1f}" class="strike"/>')
        self.parts.append(
            f'<line x1="{x + w - 4:.1f}" y1="{y + 4:.1f}" x2="{x + 4:.1f}" y2="{y + h - 4:.1f}" class="strike"/>')

    def elbow(self, x0, y0, x1, y1, label=None, label_dy=0, side='right'):
        # baja desde (x0,y0), dobla a x1 y entra hacia abajo en (x1,y1)
        ym = y1 - 14
        self.parts.append(
            f'<path d="M{x0:.1f} {y0 + 2:.1f} V{ym:.1f} H{x1:.1f} V{y1 - 2:.1f}" '
            f'class="flow" marker-end="url(#arr-{self.fid})"/>')
        if label:
            if side == 'right':
                self.text(x0 + 12, (y0 + ym) / 2 + label_dy, label, 'shape', anchor='start')
            else:
                self.text(x0 - 12, (y0 + ym) / 2 + label_dy, label, 'shape', anchor='end')
        self._up(y1)

    def outnode(self, y, side=None):
        cx = self.cx
        self.el('circle', cx=f'{cx:.1f}', cy=f'{y + 14:.1f}', r=13, **{'class': 'signode'})
        self.text(cx, y + 18.5, 'σ', 'sigt')
        self._up(y + 27)
        y2 = self.arrow(y + 27, 24)
        self.el('rect', x=f'{cx - 100:.1f}', y=f'{y2:.1f}', width=200, height=29, rx=14.5,
                **{'class': 'outpill'})
        self.text(cx, y2 + 18.5, 'p(bought) ∈ (0, 1)', 'outt')
        if side:
            self.text(cx + 118, y2 + 18, side, 'shape', anchor='start')
        self._up(y2 + 29)
        return y2 + 29

    def render(self):
        h = self.maxy + 22
        defs = (f'<defs><marker id="arr-{self.fid}" viewBox="0 0 10 10" refX="8" refY="5" '
                f'markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse">'
                f'<path d="M0.5,1 L9,5 L0.5,9 z" class="arrhead"/></marker></defs>')
        return (f'<svg viewBox="0 0 {self.w} {h:.0f}" class="fig" role="img" '
                f'xmlns="http://www.w3.org/2000/svg">{defs}{"".join(self.parts)}</svg>')


# ---------------------------------------------------------------- figuras

def fig_preproc():
    f = Fig('p0')
    y = 20
    b = f.box(y, title='supermarket_products.csv', w=590, cls='bx-data', lines=(
        '10.000 impresiones (fila = un producto mostrado en una búsqueda) × 22 columnas',
        '2.012 búsquedas (query_id) de 1 a 8 productos · target: bought (13% positivos)'))
    y = f.arrow(b['bot'])
    b = f.box(y, title='Derivar features (sin mirar el target)', w=590, cls='bx-proc', lines=(
        'listing_status ← regex al sufijo "( … )" del final del título · 21 niveles',
        'price_rel ← (price − filter_price_min) / (filter_price_max − filter_price_min)',
        'allergens: NaN → "None" (sin alérgenos declarados)'))
    y = f.arrow(b['bot'])
    b = f.box(y, title='Split por query_id — 70 / 15 / 15', w=470, cls='bx-proc', lines=(
        'las filas de una misma búsqueda nunca se reparten entre train / val / test',))
    y = f.arrow(b['bot'], 40, labels=('vocabularios y estadísticos: SOLO con train',))
    b = f.box(y, title='Preprocessor (ajustado con train, aplicado a los tres splits)', w=590,
              cls='bx-proc', lines=(
        '7 vocabularios categóricos — índice 0 = UNK para niveles no vistos',
        'z-score en las 6 numéricas (log1p antes en price y net_weight_oz)',
        'texto: title + "\\n" + description → vocabulario de 67 chars (PAD=0, UNK=1)'))
    y = f.arrow(b['bot'])
    f.tokens(y, [('x_cat  (B, 7)', 'k-tab', 150), ('x_num  (B, 6)', 'k-num', 150),
                 ('x_text  (B, 256)', 'k-txt', 165), ('y  (B,)', 'k-out', 105)], h=32)
    return f.render()


CAT_CELLS = [('status', 'k-tab', 48), ('categ', 'k-tab', 48), ('marca', 'k-tab', 48),
             ('almac', 'k-tab', 48), ('unidad', 'k-tab', 48), ('origen', 'k-tab', 48),
             ('alérg', 'k-tab', 48)]
NUM_CELLS = [('price', 'k-num', 48), ('p_rel', 'k-num', 48), ('min', 'k-num', 48),
             ('max', 'k-num', 48), ('peso', 'k-num', 48), ('nutri', 'k-num', 48)]

FTK_LINES = (
    'cada categórica → su PROPIA tabla:  nn.Embedding(cardᵢ, 32)',
    'cardinalidades: status 21 · categ 13 · marca 16 · almac 4 · unidad 6 · origen 11 · alérg 9',
    'cada numérica → xⱼ · wⱼ + bⱼ (recta aprendida por feature) — o bins por cuantiles')


def fig_feat():
    f = Fig('fa')
    y = 20
    b = f.box(y, title='1 impresión', w=430, cls='bx-data',
              lines=('x_cat: 7 índices categóricos · x_num: 6 valores (z-score)',))
    y = f.arrow(b['bot'])
    b = f.box(y, title='FeatureTokenizer', w=560, cls='bx-tok', lines=FTK_LINES)
    y = f.arrow(b['bot'])
    row = f.tokens(y, [('CLS', 'k-cls', 44)] + CAT_CELLS + NUM_CELLS)
    y = f.arrow(row['bot'], 56, labels=('(B, 14, 32)',
                                        'sin positional: un set de features no tiene orden',
                                        'sin máscara: los 14 tokens son siempre reales'))
    b = f.box(y, title='Bloque Transformer', w=540, cls='bx-blk', badge='×2', lines=(
        'LayerNorm → atención multi-cabeza (4 cabezas de dim 8) → + residual',
        'cada token arma Q, K, V · pesos = softmax(Q·Kᵀ / √8) · mezcla los V de los otros',
        'LayerNorm → FFN: Linear 32→128 → ReLU → Linear 128→32 → + residual'))
    y = f.arrow(b['bot'], labels=('(B, 14, 32)',))
    b = f.box(y, w=350, cls='bx-proc', lines=('LayerNorm final → quedarse con el token [CLS]',))
    y = f.arrow(b['bot'], labels=('(B, 32)',))
    b = f.box(y, w=190, cls='bx-mlp', lines=('Linear 32 → 1',))
    y = f.arrow(b['bot'], 30, labels=('logit (B,)',))
    f.outnode(y, side='loss: BCE (pos_weight opcional)')
    return f.render()


def fig_feat_ancho():
    """Igual que fig_feat() pero en dos columnas, para la diapositiva apaisada de la
    presentacion: entrada y tokenizacion a la izquierda, transformer y cabezal a la derecha,
    unidas por un puente horizontal a la altura de la fila de tokens."""
    f = Fig('fa', w=1560)
    A, B = 400, 1150

    b = f.box(20, title='1 impresión', w=430, cx=A, cls='bx-data',
              lines=('x_cat: 7 índices categóricos · x_num: 6 valores (z-score)',))
    y = f.arrow(b['bot'], x=A)
    b = f.box(y, title='FeatureTokenizer', w=620, cx=A, cls='bx-tok', lines=FTK_LINES)
    y = f.arrow(b['bot'], x=A)
    row = f.tokens(y, [('CLS', 'k-cls', 44)] + CAT_CELLS + NUM_CELLS, cx=A)
    for i, s in enumerate(('(B, 14, 32)',
                           'sin positional: un set de features no tiene orden',
                           'sin máscara: los 14 tokens son siempre reales')):
        f.text(A, row['bot'] + 20 + i * 15, s, 'shape')

    mid = row['y'] + row['h'] / 2  # el puente sale y entra a media altura
    blk = f.box(mid - 45.75, title='Bloque Transformer', w=560, cx=B, cls='bx-blk', badge='×2',
                lines=('LayerNorm → atención multi-cabeza (4 cabezas de dim 8) → + residual',
                       'cada token arma Q, K, V · pesos = softmax(Q·Kᵀ / √8) · mezcla los V',
                       'LayerNorm → FFN: Linear 32→128 → ReLU → Linear 128→32 → + residual'))
    f.parts.append(f'<line x1="{row["x"] + row["w"] + 6:.1f}" y1="{mid:.1f}" '
                   f'x2="{blk["x"] - 4:.1f}" y2="{mid:.1f}" class="flow" '
                   f'marker-end="url(#arr-fa)"/>')

    y = f.arrow(blk['bot'], x=B, labels=('(B, 14, 32)',))
    b = f.box(y, w=380, cx=B, cls='bx-proc',
              lines=('LayerNorm final → quedarse con el token [CLS]',))
    y = f.arrow(b['bot'], x=B, labels=('(B, 32)',))
    b = f.box(y, w=190, cx=B, cls='bx-mlp', lines=('Linear 32 → 1',))
    y = f.arrow(b['bot'], 30, x=B, labels=('logit (B,)',))
    f.cx = B  # outnode dibuja centrado en self.cx
    f.outnode(y, side='loss: BCE (pos_weight opcional)')
    return f.render()


def fig_mlp():
    f = Fig('fm')
    y = 20
    b = f.box(y, title='1 impresión', w=430, cls='bx-data',
              lines=('x_cat: 7 índices categóricos · x_num: 6 valores (z-score)',))
    y = f.arrow(b['bot'])
    b = f.box(y, title='FeatureTokenizer — idéntico al del transformer tabular', w=500,
              cls='bx-tok',
              lines=('mismos embeddings de entrada: lo único que cambia es qué los mezcla',))
    y = f.arrow(b['bot'])
    row = f.tokens(y, CAT_CELLS + NUM_CELLS)
    y = f.arrow(row['bot'], 44, labels=('flatten: 13 × 32 → (B, 416)',
                                        'sin atención: cada peso ve TODO concatenado'))
    b = f.box(y, title='MLP denso (como en SIA)', w=390, cls='bx-mlp', lines=(
        'Linear 416 → 256 → ReLU → Dropout 0.1',
        'Linear 256 → 64 → ReLU → Dropout 0.1',
        'Linear 64 → 1'))
    y = f.arrow(b['bot'], 30, labels=('logit (B,)',))
    f.outnode(y)
    return f.render()


def _char_cells(chars, kind='k-chr', w=22):
    return [(c, kind, w) for c in chars]


def fig_text():
    f = Fig('fc')
    y = 20
    b = f.box(y, title='title + "\\n" + description  (texto crudo, sin parsear)', w=640,
              cls='bx-txt', lines=(
        '"Riverbend Cultured Half And Half - 24 oz (Best Seller)"',
        '"Cultured half and half … One of the most repurchased items in its aisle."',
        '229 caracteres en este ejemplo · p95 del dataset = 243 → se trunca a 256'))
    y = f.arrow(b['bot'], 44, labels=('cada carácter → su índice en el vocabulario de 67',
                                      'UNK = 1 para chars no vistos · PAD = 0 al final'))
    y += 26  # aire para la llave de arriba
    cells = ([('CLS', 'k-cls', 44)] + _char_cells('Riverb') + [('⋯', 'k-ell', 30)]
             + _char_cells('(Best', 'k-hot') + [('␣', 'k-hot', 22)] + _char_cells('Seller)', 'k-hot')
             + [('⋯', 'k-ell', 30), ('PAD', 'k-pad', 42), ('PAD', 'k-pad', 42)])
    row = f.tokens(y, cells)
    hot = [c for c in row['cells'] if c['kind'] == 'k-hot']
    f.bracket(hot[0]['x'], hot[-1]['x'] + hot[-1]['w'], row['y'] - 6,
              'la señal de estado viaja escondida acá (--strip-status la borra)', above=True)
    y = f.arrow(row['bot'], 36, labels=('[CLS] + 256 chars → (B, 257, 32)',))
    b = f.box(y, w=580, cls='bx-proc', lines=(
        '+ embedding posicional aprendido (257 × 32): en texto el ORDEN importa',
        'máscara de padding: los PAD no reciben atención (score −10⁹ antes del softmax)'))
    y = f.arrow(b['bot'])
    b = f.box(y, title='Bloque Transformer', w=540, cls='bx-blk', badge='×2', lines=(
        'atención de 257 × 257 por cabeza: acá el cómputo es REAL (por esto la GPU)',
        '4 cabezas de dim 8 → FFN 32→128→32 · pre-LN · residuales'))
    y = f.arrow(b['bot'], labels=('(B, 257, 32)',))
    b = f.box(y, w=350, cls='bx-proc', lines=('LayerNorm final → quedarse con el token [CLS]',))
    y = f.arrow(b['bot'], labels=('(B, 32)',))
    b = f.box(y, w=190, cls='bx-mlp', lines=('Linear 32 → 1',))
    y = f.arrow(b['bot'], 30, labels=('logit (B,)',))
    f.outnode(y)
    return f.render()


def fig_hybrid():
    f = Fig('fh')
    y = 20
    bl = f.box(y, title='1 impresión (tabular)', cx=225, w=330, cls='bx-data',
               lines=('x_cat (7) · x_num (6)',))
    br = f.box(y, title='title + "\\n" + description', cx=615, w=340, cls='bx-txt',
               lines=('256 índices de caracteres',))
    row_y = max(bl['bot'], br['bot']) + 58
    cells = ([('CLS', 'k-cls', 44), ('status', 'k-tab', 48), ('categ', 'k-tab', 48),
              ('⋯', 'k-ell', 30), ('nutri', 'k-num', 48)]
             + _char_cells('Riv') + [('⋯', 'k-ell', 30), ('PAD', 'k-pad', 42)])
    row = f.tokens(row_y, cells)
    feat_x1, feat_x2 = row['cells'][1]['x'], row['cells'][4]['x'] + row['cells'][4]['w']
    chr_x1, chr_x2 = row['cells'][5]['x'], row['cells'][-1]['x'] + row['cells'][-1]['w']
    f.elbow(225, bl['bot'], (feat_x1 + feat_x2) / 2, row_y, label='FeatureTokenizer → 13 tokens')
    f.elbow(615, br['bot'], (chr_x1 + chr_x2) / 2, row_y,
            label='char embedding (67 × 32) → 256 tokens', side='left')
    f.bracket(feat_x1, feat_x2, row['bot'] + 5, '13 feature-tokens')
    f.bracket(chr_x1, chr_x2, row['bot'] + 5, '256 char-tokens')
    y = f.arrow(row['bot'] + 26, 44, labels=('UNA sola secuencia: (B, 1 + 13 + 256 = 270, 32)',
                                             '+ PE aprendido (270 × 32) sobre TODA la secuencia'))
    b = f.box(y, title='Bloque Transformer', w=560, cls='bx-blk', badge='×2', lines=(
        'la atención puede CRUZAR texto ↔ features en la misma capa',
        'p. ej.: el token price_rel puede atender a los chars de "(Best Seller)"',
        '4 cabezas de dim 8 · FFN 32→128→32 · máscara solo sobre los PAD del texto'))
    y = f.arrow(b['bot'], labels=('(B, 270, 32)',))
    b = f.box(y, w=350, cls='bx-proc', lines=('LayerNorm final → quedarse con el token [CLS]',))
    y = f.arrow(b['bot'], labels=('(B, 32)',))
    b = f.box(y, w=190, cls='bx-mlp', lines=('Linear 32 → 1',))
    y = f.arrow(b['bot'], 30, labels=('logit (B,)',))
    f.outnode(y)
    return f.render()


def fig_fusion():
    f = Fig('fx', w=880)
    f.cx = 440
    CL, CR = 235, 660
    bl = f.box(20, title='title + "\\n" + description', cx=CL, w=340, cls='bx-txt',
               lines=('256 chars — o 64 PALABRAS (--text-tokens words)',))
    br = f.box(20, title='1 impresión (tabular)', cx=CR, w=310, cls='bx-data',
               lines=('x_cat (7) · x_num (6)',))
    yl = f.arrow(bl['bot'], x=CL, dy=32)
    tb = f.box(yl, title='Torre de texto interna', cx=CL, w=340, cls='bx-blk', badge='×2',
               lines=('CLS + tokens + PE → bloques → LN',
                      'CLS = resumen del texto (B, 32)',
                      'opcional: inicializar con word2vec'))
    yr = f.arrow(br['bot'], x=CR, dy=32)
    ftk = f.box(yr, title='FeatureTokenizer', cx=CR, w=320, cls='bx-tok',
                lines=('13 tokens de 32 dims',))
    row_y = max(tb['bot'], ftk['bot']) + 58
    row = f.tokens(row_y, [('CLS', 'k-cls', 44), ('status', 'k-tab', 48), ('categ', 'k-tab', 48),
                           ('⋯', 'k-ell', 30), ('nutri', 'k-num', 48), ('TXT', 'k-hot', 48)])
    feat_mid = (row['cells'][1]['x'] + row['cells'][4]['x'] + 48) / 2
    txt_cell = row['cells'][-1]
    f.elbow(CR, ftk['bot'], feat_mid, row_y, label='13 tokens', side='right')
    f.elbow(CL, tb['bot'], txt_cell['x'] + 24, row_y, label='el resumen: UN token', side='left')
    f.bracket(row['x'], row['x'] + row['w'], row['bot'] + 5,
              'CLS + 13 features + 1 resumen = 15 tokens · sin PE (sigue siendo un set)')
    y = f.arrow(row['bot'] + 26, 36, labels=('(B, 15, 32)',))
    b = f.box(y, title='Bloque Transformer', w=560, cls='bx-blk', badge='×2', lines=(
        'la atención cruza features ↔ RESUMEN del texto',
        'sin dilución: 1 token de texto en vez de 256 (el mal medido del híbrido)'))
    y = f.arrow(b['bot'], labels=('(B, 15, 32)',))
    b = f.box(y, w=350, cls='bx-proc', lines=('LayerNorm final → quedarse con el token [CLS]',))
    y = f.arrow(b['bot'], labels=('(B, 32)',))
    b = f.box(y, w=190, cls='bx-mlp', lines=('Linear 32 → 1',))
    y = f.arrow(b['bot'], 30, labels=('logit (B,)',))
    f.outnode(y)
    return f.render()


def fig_tower():
    f = Fig('ft', w=880)
    f.cx = 440
    CL, CR = 235, 660
    y = 20
    bl = f.box(y, title='title + "\\n" + description', cx=CL, w=340, cls='bx-txt',
               lines=('256 índices de caracteres',))
    br = f.box(y, title='1 impresión (tabular)', cx=CR, w=310, cls='bx-data',
               lines=('x_cat (7) · x_num (6)',))
    yl = f.arrow(bl['bot'], x=CL, dy=32)
    rowl = f.tokens(yl, [('CLS', 'k-cls', 40)] + _char_cells('Riv', w=20)
                    + [('⋯', 'k-ell', 26), ('PAD', 'k-pad', 38)], cx=CL)
    yl = f.arrow(rowl['bot'], x=CL, dy=40, labels=('+ PE (257 × 32)', 'máscara sobre PAD'))
    tb = f.box(yl, title='Bloque Transformer', cx=CL, w=330, cls='bx-blk', badge='×2',
               lines=('la atención SOLO ve caracteres', 'nunca ve lo tabular'))
    yl = f.arrow(tb['bot'], x=CL, dy=32)
    emb = f.box(yl, cx=CL, w=340, cls='bx-cls',
                lines=('LayerNorm → [CLS] = embedding del texto',))
    yr = f.arrow(br['bot'], x=CR, dy=32)
    ftk = f.box(yr, title='FeatureTokenizer', cx=CR, w=320, cls='bx-tok',
                lines=('13 tokens de 32 dims',))
    yr = f.arrow(ftk['bot'], x=CR, dy=32)
    rowr = f.tokens(yr, [('status', 'k-tab', 48), ('categ', 'k-tab', 48), ('⋯', 'k-ell', 30),
                         ('nutri', 'k-num', 48)], cx=CR)
    ct = max(emb['bot'], rowr['bot']) + 40
    f.elbow(CL, emb['bot'], 440 - 62, ct, label='(B, 32)')
    f.elbow(CR, rowr['bot'], 440 + 62, ct, label='flatten → (B, 416)')
    b = f.box(ct, w=330, cls='bx-proc',
              lines=('⊕ concatenar → (B, 32 + 416 = 448)',))
    y = f.arrow(b['bot'])
    b = f.box(y, title='el CLASIFICADOR es un MLP', w=340, cls='bx-mlp', lines=(
        'Linear 448 → 128 → ReLU → Dropout 0.1', 'Linear 128 → 1'))
    y = f.arrow(b['bot'], 30, labels=('logit (B,)',))
    f.outnode(y)
    return f.render()


def fig_listwise():
    f = Fig('fl')
    y = 20
    b = f.box(y, title='UNA búsqueda completa (no una impresión suelta)', w=570, cls='bx-data',
              lines=('los 1–8 productos que compiten en la misma página de resultados',
                     'en este ejemplo la página tiene 6 productos → 2 slots de padding'))
    y = f.arrow(b['bot'], 40, labels=('por producto: x_cat (7) + x_num (6)',))
    b = f.box(y, title='Colapsar cada producto a UN token', w=570, cls='bx-tok', lines=(
        'FeatureTokenizer → 13 tokens de 32 → flatten (416) → Linear 416 → 32',
        'el producto entero queda resumido en un vector de 32 dims'))
    y = f.arrow(b['bot'], 40, labels=('(Q, 8, 32) — Q = búsquedas del batch',))
    row = f.tokens(y, [(f'prod {i}', 'k-prod', 62) for i in range(1, 7)]
                   + [('PAD', 'k-pad', 62), ('PAD', 'k-pad', 62)])
    f.bracket(row['x'], row['x'] + row['w'], row['bot'] + 5,
              'prod_mask marca los slots reales · sin positional (el CSV no trae orden de página)')
    y = f.arrow(row['bot'] + 26, 36)
    b = f.box(y, title='Bloque Transformer', w=560, cls='bx-blk', badge='×2', lines=(
        'la atención corre ENTRE los productos de la misma página',
        '"¿me compran a MÍ, dado lo que aparece al lado?" — modela la competencia'))
    y = f.arrow(b['bot'], labels=('(Q, 8, 32)',))
    b = f.box(y, w=440, cls='bx-mlp',
              lines=('LayerNorm → Linear 32 → 1 aplicado a CADA producto',))
    y = f.arrow(b['bot'], 36, labels=('8 logits por página → σ',))
    orow = f.tokens(y, [(f'p{s}', 'k-out', 46) for s in '₁₂₃₄₅₆']
                    + [('–', 'k-pad', 46), ('–', 'k-pad', 46)], h=28)
    f.bracket(orow['x'], orow['x'] + orow['w'], orow['bot'] + 5,
              'un p(bought) por producto · la BCE se calcula SOLO sobre los slots reales')
    return f.render()


def fig_listwise_texto():
    f = Fig('flt')
    y = 20
    b = f.box(y, title='UNA búsqueda completa', w=560, cls='bx-data',
              lines=('los 1–8 productos que compiten en la misma página de resultados',))
    y = f.arrow(b['bot'], 40, labels=('por producto: x_cat (7) + x_num (6) + su TEXTO (256 chars)',))
    b = f.box(y, title='Colapsar cada producto: tabular + resumen de su texto', w=580, cls='bx-tok',
              lines=('FeatureTokenizer → 13 tokens de 32 → flatten (416)',
                     'TextEncoder (torre de chars ×2) → CLS = resumen del texto (32)',
                     'concat (448) → Linear 448 → 32: el producto entero, CON su texto, en un vector'))
    y = f.arrow(b['bot'], 40, labels=('(Q, 8, 32) · máscara derivada: slot vacío = texto todo-PAD',))
    row = f.tokens(y, [(f'p+t {i}', 'k-prod', 62) for i in range(1, 7)]
                   + [('PAD', 'k-pad', 62), ('PAD', 'k-pad', 62)])
    f.bracket(row['x'], row['x'] + row['w'], row['bot'] + 5,
              'cada token de producto ahora VE su título y descripción · sin positional')
    y = f.arrow(row['bot'] + 26, 36)
    b = f.box(y, title='Bloque Transformer', w=560, cls='bx-blk', badge='×2', lines=(
        'la atención corre ENTRE los productos de la página, como en listwise',
        'pero ahora nadie compite "ciego" a la señal del sufijo del título'))
    y = f.arrow(b['bot'], labels=('(Q, 8, 32)',))
    b = f.box(y, w=440, cls='bx-mlp',
              lines=('LayerNorm → Linear 32 → 1 aplicado a CADA producto',))
    y = f.arrow(b['bot'], 36, labels=('8 logits por página → σ',))
    orow = f.tokens(y, [(f'p{s}', 'k-out', 46) for s in '₁₂₃₄₅₆']
                    + [('–', 'k-pad', 46), ('–', 'k-pad', 46)], h=28)
    f.bracket(orow['x'], orow['x'] + orow['w'], orow['bot'] + 5,
              'un p(bought) por producto · BCE solo sobre los slots reales')
    return f.render()


def fig_familias():
    f = Fig('ff', w=840)
    y = 26
    rl = f.tokens(y, [('CLS', 'k-cls', 42), ('status', 'k-tab', 50), ('categ', 'k-tab', 48),
                      ('⋯', 'k-ell', 28), ('nutri', 'k-num', 48)], cx=225)
    f.strike(rl['cells'][1], rl)
    f.bracket(rl['x'], rl['x'] + rl['w'], rl['bot'] + 5,
              '--drop-features listing_status: saca el token de estado')
    rr = f.tokens(y, _char_cells('(Best', 'k-hot') + [('⋯', 'k-ell', 26)]
                  + _char_cells('r)', 'k-hot') + [('.', 'k-chr', 22), ('␣', 'k-hot', 22)]
                  + _char_cells('One', 'k-hot') + [('⋯', 'k-ell', 26)], cx=610)
    for cell in rr['cells']:
        if cell['kind'] == 'k-hot':
            f.strike(cell, rr)
    f.bracket(rr['x'], rr['x'] + rr['w'], rr['bot'] + 5,
              '--strip-status: borra sufijo del título y última oración')
    return f.render()


# ---------------------------------------------------------------- html

FIGS = {
    'feat_ancho': fig_feat_ancho(),
    'p0': fig_preproc(), 'feat': fig_feat(), 'mlp': fig_mlp(), 'text': fig_text(),
    'hybrid': fig_hybrid(), 'fusion': fig_fusion(), 'tower': fig_tower(),
    'listwise': fig_listwise(), 'listwise_texto': fig_listwise_texto(),
    'familias': fig_familias(),
}

for name, svg in FIGS.items():
    ElementTree.fromstring(svg)  # valida XML; explota si algo quedó mal formado

DARK_TOKENS = """
  --bg:#0F141A; --card:#171E27; --figbg:#121924; --grid:#202C3B;
  --ink:#E6EBF1; --ink2:#C0CAD6; --muted:#8D9BAB; --line:#2A3543;
  --arrow:#68788A; --boxbr:#5C6C7E;
  --tab-bg:#143530; --tab-br:#34A98F; --tab-tx:#8FD9C8;
  --num-bg:#0F2822; --num-br:#27806B; --num-tx:#79C4B2;
  --txt-bg:#3B2F12; --txt-br:#C79A2F; --txt-tx:#ECCB7B;
  --hot-bg:#57431A; --hot-br:#E8B84B; --hot-tx:#F5D98F;
  --cls-bg:#2B2350; --cls-br:#9678E8; --cls-tx:#C7B5F5;
  --pad-br:#55626F; --pad-tx:#7C8B9A;
  --blk-bg:#1C2531; --blk-br:#7386A0;
  --mlp-bg:#17293E; --mlp-br:#4C8BD4; --mlp-tx:#A9CBF0;
  --prod-bg:#1F9781; --prod-tx:#06231C;
  --out:#E85585; --out-tx:#230A13;
"""

CSS = """
:root{
  --bg:#EFF2F5; --card:#FFFFFF; --figbg:#F8FAFC; --grid:#DFE6EC;
  --ink:#1B2530; --ink2:#3D4B5A; --muted:#5F7183; --line:#D8DFE7;
  --arrow:#8E9CAA; --boxbr:#8A99A9;
  --tab-bg:#DFF0EB; --tab-br:#1C8A76; --tab-tx:#0B5B4C;
  --num-bg:#F0F8F5; --num-br:#66B4A2; --num-tx:#166553;
  --txt-bg:#F8ECCB; --txt-br:#B58117; --txt-tx:#7C5709;
  --hot-bg:#F2D592; --hot-br:#8A5F06; --hot-tx:#6B4A05;
  --cls-bg:#E7DEF9; --cls-br:#7052C9; --cls-tx:#4A338F;
  --pad-br:#9FACB9; --pad-tx:#7D8C9B;
  --blk-bg:#EDF1F6; --blk-br:#64778F;
  --mlp-bg:#DCE9F8; --mlp-br:#2E68AC; --mlp-tx:#1C4B85;
  --prod-bg:#1C8A76; --prod-tx:#FFFFFF;
  --out:#C22B5E; --out-tx:#FFFFFF;
  --sans:"Segoe UI","Noto Sans","Liberation Sans",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"Cascadia Code","JetBrains Mono","Fira Mono","DejaVu Sans Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){ %DARK% } }
:root[data-theme="dark"]{ %DARK% }

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 var(--sans);
  -webkit-font-smoothing:antialiased}
.wrap{max-width:920px;margin:0 auto;padding:40px 20px 64px}
header.top{margin-bottom:26px}
.eyebrow{font:600 10.5px var(--mono);letter-spacing:.16em;text-transform:uppercase;
  color:var(--muted)}
h1{font:700 32px/1.15 var(--sans);margin:8px 0 10px;letter-spacing:-.01em;text-wrap:balance}
.lede{color:var(--ink2);max-width:66ch;margin:0}
nav.jump{display:flex;flex-wrap:wrap;gap:7px;margin:18px 0 0}
nav.jump a{font:11px var(--mono);color:var(--ink2);text-decoration:none;
  border:1px solid var(--line);background:var(--card);border-radius:999px;padding:4px 11px}
nav.jump a:hover,nav.jump a:focus-visible{border-color:var(--boxbr);color:var(--ink)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--mlp-br);outline-offset:2px}

.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:22px 24px 20px;margin:0 0 22px}
.card h2{font:700 21px/1.2 var(--sans);margin:6px 0 2px;letter-spacing:-.01em}
.card h2 .soft{color:var(--muted);font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 14px}
.chips span{font:10.5px var(--mono);color:var(--ink2);border:1px solid var(--line);
  background:var(--figbg);border-radius:999px;padding:3px 9px;font-variant-numeric:tabular-nums}
.chips span.cmd{color:var(--mlp-tx);border-color:var(--mlp-br)}
.chips span.rank{background:var(--tab-br);border-color:var(--tab-br);color:#fff;font-weight:700}
.figpanel{background:var(--figbg);border:1px solid var(--line);border-radius:10px;
  background-image:radial-gradient(var(--grid) 1px,transparent 1.3px);
  background-size:18px 18px;padding:12px 10px;overflow-x:auto}
.figpanel svg{display:block;width:100%;height:auto;min-width:740px}
.caption{color:var(--ink2);font-size:13.5px;margin:13px 2px 0;max-width:78ch}
.caption strong{color:var(--ink)}
.caption code, .lede code, li code{font:12px var(--mono);background:var(--figbg);
  border:1px solid var(--line);border-radius:5px;padding:1px 5px;white-space:nowrap}

.legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin:12px 0 4px}
.legend span{display:inline-flex;align-items:center;gap:7px;font:12px var(--mono);
  color:var(--ink2)}
.sw{width:14px;height:14px;border-radius:4px;flex:none;border:1.4px solid}
.sw-tab{background:var(--tab-bg);border-color:var(--tab-br)}
.sw-num{background:var(--num-bg);border-color:var(--num-br)}
.sw-txt{background:var(--txt-bg);border-color:var(--txt-br)}
.sw-hot{background:var(--hot-bg);border-color:var(--hot-br)}
.sw-cls{background:var(--cls-bg);border-color:var(--cls-br)}
.sw-pad{background:transparent;border-color:var(--pad-br);border-style:dashed}
.sw-prod{background:var(--prod-bg);border-color:var(--prod-bg)}
.sw-blk{background:var(--blk-bg);border-color:var(--blk-br)}
.sw-mlp{background:var(--mlp-bg);border-color:var(--mlp-br)}
.sw-out{background:var(--out);border-color:var(--out)}

.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;min-width:820px;font-size:13px}
th{font:600 10.5px var(--mono);letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);
  background:var(--figbg)}
td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top;color:var(--ink2)}
tr:last-child td{border-bottom:none}
td:first-child{color:var(--ink);font-weight:600;white-space:nowrap}
td.n{font:12.5px var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.c{font:11.5px var(--mono);white-space:nowrap}
tr.ref td{color:var(--muted);font-weight:400}
tr.ref td:first-child{font-weight:400;font-style:italic}

footer{color:var(--muted);font-size:12.5px;margin-top:30px}
footer code{font:11.5px var(--mono)}

/* ------- svg ------- */
.fig text{font-family:var(--sans)}
.b-title{font:600 12.5px var(--sans);fill:var(--ink)}
.b-line{font:11px var(--sans);fill:var(--ink2)}
.shape{font:10.5px var(--mono);fill:var(--muted)}
.bx{stroke-width:1.3}
.bx-data{fill:var(--card);stroke:var(--boxbr)}
.bx-proc{fill:var(--figbg);stroke:var(--boxbr);stroke-dasharray:none}
.bx-tok{fill:var(--tab-bg);stroke:var(--tab-br)}
.bx-txt{fill:var(--txt-bg);stroke:var(--txt-br)}
.bx-blk{fill:var(--blk-bg);stroke:var(--blk-br);stroke-width:1.6}
.bx-mlp{fill:var(--mlp-bg);stroke:var(--mlp-br)}
.bx-cls{fill:var(--cls-bg);stroke:var(--cls-br)}
.badge{fill:var(--card);stroke:var(--blk-br);stroke-width:1.2}
.badge-t{font:700 11px var(--mono);fill:var(--ink)}
.tk{stroke-width:1.3}
.tkt{font:9.5px var(--mono)}
.k-tab{fill:var(--tab-bg);stroke:var(--tab-br)} .k-tab-t{fill:var(--tab-tx)}
.k-num{fill:var(--num-bg);stroke:var(--num-br)} .k-num-t{fill:var(--num-tx)}
.k-chr{fill:var(--txt-bg);stroke:var(--txt-br)} .k-chr-t{fill:var(--txt-tx)}
.k-hot{fill:var(--hot-bg);stroke:var(--hot-br);stroke-width:1.7} .k-hot-t{fill:var(--hot-tx)}
.k-cls{fill:var(--cls-bg);stroke:var(--cls-br)} .k-cls-t{fill:var(--cls-tx);font-weight:700}
.k-pad{fill:none;stroke:var(--pad-br);stroke-dasharray:4 3} .k-pad-t{fill:var(--pad-tx)}
.k-prod{fill:var(--prod-bg);stroke:var(--prod-bg)} .k-prod-t{fill:var(--prod-tx);font-weight:700}
.k-out{fill:var(--out);stroke:var(--out)} .k-out-t{fill:var(--out-tx);font-weight:700}
.k-ell-t{fill:var(--muted);font-size:13px}
.flow{stroke:var(--arrow);stroke-width:1.5;fill:none}
.arrhead{fill:var(--arrow)}
.brk{stroke:var(--arrow);stroke-width:1.1;fill:none}
.strike{stroke:var(--out);stroke-width:2.2;stroke-linecap:round}
.signode{fill:var(--card);stroke:var(--out);stroke-width:1.6}
.sigt{font:italic 700 13px Georgia,serif;fill:var(--out)}
.outpill{fill:var(--out)}
.outt{font:600 12px var(--mono);fill:var(--out-tx)}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
""".replace('%DARK%', DARK_TOKENS)


def arch_section(aid, eyebrow, title, soft, chips, fig, caption):
    ch = ''.join(f'<span class="{c[1]}">{E(c[0])}</span>' if isinstance(c, tuple)
                 else f'<span>{E(c)}</span>' for c in chips)
    return f'''
<section class="card" id="{aid}">
  <div class="eyebrow">{E(eyebrow)}</div>
  <h2>{E(title)} <span class="soft">— {E(soft)}</span></h2>
  <div class="chips">{ch}</div>
  <div class="figpanel">{fig}</div>
  <p class="caption">{caption}</p>
</section>'''


SECTIONS = []

SECTIONS.append(f'''
<section class="card" id="p0">
  <div class="eyebrow">PANEL 0 · COMPARTIDO POR TODAS</div>
  <h2>Preprocesamiento <span class="soft">— del CSV a los tensores</span></h2>
  <div class="chips"><span class="cmd">btr/data.py</span><span>igual para todas las arquitecturas</span><span>fit solo con train</span></div>
  <div class="figpanel">{FIGS['p0']}</div>
  <p class="caption">Quedan <strong>afuera de la entrada</strong>: <code>cart</code> (leakage: bought ⟹ cart al 100%),
  <code>query_id</code> (solo particiona), <code>timestamp</code> (ruido, ver EDA), <code>package_size</code> y
  <code>dimensions_in</code> (redundantes con el peso), <code>filter_category</code> y <code>filter_storage_type</code>
  (los productos siempre los cumplen; el rango de precio sí entra, vía <code>price_rel</code>), e
  <code>ingredients</code> (v1). Todos verificables por ablación: <code>--extra-features</code> los reintroduce
  (medido: no aportan — <code>feat_extras</code> Δ≈0).</p>
</section>''')

SECTIONS.append(arch_section(
    'feat', 'FORMULACIÓN A · FEATURES COMO TOKENS · ESTILO FT-TRANSFORMER',
    'Transformer tabular', 'cada feature es un token',
    [('#1 · mejor 0.824 (feat_ordinal — MODELO FINAL)', 'rank'),
     ('--arch transformer --formulation features', 'cmd'), '14 tokens', 'd=32 · 4 cabezas · 2 bloques',
     '28.289 params', 'suite: feat_base + variantes'],
    FIGS['feat'],
    '''La pregunta que responde: <strong>¿la atención entre features aporta algo?</strong> Cada feature entra con
    identidad propia (su tabla de embedding); la atención aprende interacciones entre ellos — p. ej.
    <em>price_rel × categoría</em>, la U invertida del precio — y el [CLS] junta todo para clasificar.
    Sin positional encoding porque un conjunto de features no tiene orden (medido: Δ ≈ 0).
    Resultado: <strong>0.794 ± 0.033</strong> (6 seeds); 0.816 con 1 cabeza — y el <strong>MODELO FINAL del TP
    es este mismo esqueleto con encoding ORDINAL</strong> de categóricas: <strong>0.824 ± 0.018</strong>
    (<code>feat_ordinal</code>, 26k parámetros). La 3ª tanda cerró la búsqueda: sumarle capacidad al ordinal
    EMPEORA (−0.02 a −0.05) — el prior simple era el punto. Su mapa de atención: el CLS pone 0.75 de su
    atención en <code>status</code> (ver <code>graficos/</code>).'''))

SECTIONS.append(arch_section(
    'mlp', 'BASELINE SIN ATENCIÓN · EL CONTROL DEL EXPERIMENTO',
    'MLP denso', 'mismos embeddings, sin atención',
    [('#2 · mejor 0.797 (mlp_onehot)', 'rank'),
     ('--arch mlp', 'cmd'), 'flatten 416', '126.209 params (4,5× el transformer)',
     'suite: mlp_base · mlp_onehot'],
    FIGS['mlp'],
    '''El control que le da sentido a A: usa <strong>exactamente los mismos tokens de entrada</strong> pero los
    concatena y los mezcla con un MLP denso, como en SIA. Con la misma representación, la atención le saca
    <strong>+0.048</strong> (apareado, gana 5/6 seeds). Matiz honesto de la 2ª tanda: con one-hot crudo el MLP
    mejora a 0.797 (<code>mlp_onehot</code>, +0.047 sobre sus embeddings, 6/6) — parte del déficit era la
    entrada, no la falta de atención; contra ese MLP mejorado, el mejor transformer aporta ~+0.027.'''))

SECTIONS.append(arch_section(
    'text', 'FORMULACIÓN C · CARACTERES COMO TOKENS · LA DEMO DE LA CÁTEDRA',
    'Transformer de texto', 'cada carácter es un token',
    [('#8 · mejor 0.652 (text_base)', 'rank'),
     ('--arch transformer --formulation text', 'cmd'), '257 tokens', '35.713 params',
     'PE obligatorio', 'suite: text_base · text_d64 · text_len96 · text_words · text_intrinseco'],
    FIGS['text'],
    '''La adaptación directa de la demo (chars como tokens), pasada de <em>decoder que genera</em> a
    <em>encoder que clasifica</em>: sin máscara causal, con [CLS] y un solo logit. No se le da nada parseado:
    tiene que descubrir solo que la señal vive en el sufijo del título — y en la última oración de la
    descripción, que repite el estado en prosa. Atención 257×257 por cabeza: acá el cómputo es real y por
    esto la suite exige la 3070. Resultado: <strong>0.652 ± 0.039</strong> — encontró el sufijo solo (≫ techo
    intrínseco 0.16) pero leer chars crudos con 36k parámetros no alcanza al tabular. Curiosidades medidas:
    MÁS paciencia lo empeora (sobreajuste de selección por val) y truncar al título también
    (<code>text_len96</code> −0.027: la copia redundante de la descripción ayudaba). La variante
    <code>text_words</code> (palabras, 64 tokens) empata con chars (−0.006): la secuencia corta no compensa
    la tabla de embeddings grande.'''))

SECTIONS.append(arch_section(
    'hybrid', 'FORMULACIÓN A + C · TODO EN UNA SECUENCIA',
    'Transformer híbrido', 'features y caracteres juntos',
    [('#7 · mejor 0.736 (sin_regex, paciencia 20)', 'rank'),
     ('--arch transformer --formulation hybrid', 'cmd'), '270 tokens', '39.073 params',
     'suite: hybrid_full · hybrid_sin_regex · hybrid_status_campo · hybrid_intrinseco'],
    FIGS['hybrid'],
    '''Los 13 feature-tokens y los 256 char-tokens conviven en <strong>una misma secuencia</strong>, así que la
    atención puede cruzar texto ↔ tabular en la misma capa. La variante <code>hybrid_sin_regex</code> le saca el
    token <code>listing_status</code> parseado y contesta una pregunta linda: ¿la atención recupera sola desde los
    caracteres lo que nosotros extrajimos con una regex? Resultado: <strong>SÍ</strong> — quitarle el token parseado
    no le cuesta nada (0.735 vs 0.735 con pac20), aunque el híbrido completo (0.705–0.735) queda debajo del
    tabular puro: los 256 chars diluyen la atención sobre los 13 features. Limpiar el sufijo duplicado del
    texto (<code>hybrid_status_campo</code>, idea de Fer) sube a 0.733 — el canal doble era ruido — y la
    <strong>fusión</strong> (abajo) ataca la dilución de raíz.'''))

SECTIONS.append(arch_section(
    'fusion', 'FORMULACIÓN 5B · REVISIÓN EXTERNA · EL RESUMEN COMO TOKEN',
    'Transformer fusión', 'el texto, comprimido a un token',
    [('#4 · mejor 0.775 (fusion_base — empate con la torre)', 'rank'),
     ('--arch transformer --formulation fusion', 'cmd'), '15 tokens', '63.969 params',
     'suite: fusion_base · fusion_words · fusion_words_w2v'],
    FIGS['fusion'],
    '''El punto medio exacto que faltaba entre el híbrido y la torre, sugerido por la revisión externa:
    la torre comprime el texto a su [CLS] y ese vector entra como <strong>token 15</strong> de la secuencia
    tabular. La atención puede cruzar texto ↔ features (lo que a la torre le falta) pero al nivel del
    <strong>resumen</strong>, sin que 256 caracteres diluyan a los 13 features (lo que al híbrido lo mata,
    medido: −0.06). Resultado (3ª tanda): <strong>0.775 ± 0.036</strong> —
    <strong>+0.069 sobre el híbrido (gana 6/6)</strong>: la compresión cura la dilución por completo. Pero
    <strong>empata EXACTO con la torre (Δ −0.0002)</strong>: cruzar por atención o por concat da igual una vez
    que el texto llega comprimido. Palabras: peor que chars (la tabla de embeddings grande sobreajusta);
    <strong>w2v-init la mejora +0.010</strong> (el pre-entrenamiento regulariza — clase 1 → clase 2) sin
    alcanzar a chars: a esta escala, el tokenizador chico de la demo era el correcto.'''))

SECTIONS.append(arch_section(
    'tower', 'FORMULACIÓN C2 · TORRE DE TEXTO · PROPUESTA DE FER',
    'Torre de texto + MLP', 'el transformer solo hace embeddings',
    [('#3 · mejor 0.775 (tower_base)', 'rank'),
     ('--arch tower', 'cmd'), '257 + 13 tokens', '96.225 params',
     'suite: tower_base · tower_sin_regex · tower_status_campo'],
    FIGS['tower'],
    '''La idea original de Fer: el transformer trabaja <strong>solo como encoder de texto</strong> (estilo
    BERT de la clase 2) y produce un embedding de 32 dims; el clasificador de verdad es un MLP. Resultado:
    <strong>0.775 ± 0.022</strong>, la mejor de las que ven texto — pero su embedding único NO recupera la señal
    del regex (sin_regex: −0.04, gana 1/6): el cuello de botella de comprimir 256 chars en 32 dims es real.
    Contraste perfecto con el híbrido (que recupera, pero diluido).'''))

SECTIONS.append(arch_section(
    'listwise', 'FORMULACIÓN B · PRODUCTOS COMO TOKENS · LISTWISE',
    'Transformer de página', 'cada producto de la query es un token',
    [('#6 · mejor 0.740 (paciencia 20)', 'rank'),
     ('--arch listwise', 'cmd'), '8 tokens (batch en queries)', '41.601 params',
     'suite: listwise_base'],
    FIGS['listwise'],
    '''La única formulación que ve la <strong>página completa</strong>: colapsa cada producto a un vector y la
    atención corre entre los productos que compiten en la misma búsqueda. Resultado: <strong>0.698 ± 0.044</strong>,
    y <strong>0.740</strong> con paciencia 20 (el grupo MÁS beneficiado por entrenar más: +0.041). Con el texto
    adentro del token de producto (<code>listwise_texto</code>, propuesta de Junior): <strong>0.749 ± 0.060</strong>
    (+0.005, gana 3/4 seeds) — el texto lo ayuda *algo*, pero la formulación queda lejos del tabular: la
    competencia de página es débil, como dijo el EDA (§2.5).'''))

SECTIONS.append(arch_section(
    'listwise_texto', 'FORMULACIÓN B+ · PROPUESTA #1 DE JUNIOR · EL PRODUCTO LEE SU TEXTO',
    'Listwise + texto', 'la página, sin productos ciegos',
    [('#5 · mejor 0.749 (4 seeds)', 'rank'),
     ('--arch listwise --listwise-texto', 'cmd'), '8 tokens (c/u con su texto)', '78.305 params',
     'suite: listwise_texto'],
    FIGS['listwise_texto'],
    '''La propuesta #1 de Junior, respondida: el listwise original competía <strong>ciego</strong> a la señal
    más fuerte (el sufijo del título); acá cada producto colapsa tabular + el resumen de su texto (una
    TextEncoder por producto) antes de que la página compita. Resultado: <strong>0.749 ± 0.060</strong>
    (+0.005 sobre listwise con paciencia 20, gana 3/4 seeds; con 2 seeds daba +0.016). El texto lo ayuda
    <em>algo</em> — estaba parcialmente ciego, no mal concebido — pero el techo de la formulación es la
    competencia débil del dataset (EDA §2.5). Seeds 46–47 pendientes por resume; no cambian la lectura.'''))

# orden de la página = ranking (mejor → peor); p0 primero, familias al final
RANKING = ['p0', 'feat', 'mlp', 'tower', 'fusion', 'listwise_texto', 'listwise',
           'hybrid', 'text', 'familias']
import re as _re_orden
SECTIONS.sort(key=lambda s: RANKING.index(_re_orden.search(r'id="(\w+)"', s).group(1)))

SECTIONS.append(f'''
<section class="card" id="familias">
  <div class="eyebrow">EJE TRANSVERSAL · NO SON ARQUITECTURAS NUEVAS</div>
  <h2>Dos familias <span class="soft">— catálogo vs. intrínseco (producto nuevo)</span></h2>
  <div class="chips"><span class="cmd">--drop-features listing_status</span><span class="cmd">--strip-status</span><span>aplican sobre casi todas las de arriba</span></div>
  <div class="figpanel">{FIGS['familias']}</div>
  <p class="caption">La discusión del "(Best Seller)" no se resuelve eligiendo: se corren <strong>las dos familias</strong>
  con la misma arquitectura. <strong>Catálogo</strong> usa todo (el estado existe al momento de la impresión);
  <strong>intrínseco</strong> recorta el estado de la entrada — el token parseado <em>y</em> su copia escondida en el
  texto — y simula el producto nuevo, sin historial. Medido de punta a punta: la familia intrínseca queda en
  <strong>0.14–0.16 ≈ el techo del GBM sin estado (0.162)</strong>, con <code>text_intrinseco</code> como el más
  bajo — el strip funcionó, no quedó puerta trasera. Combinando los ejes salió el <strong>2×2 del estado</strong>
  (idea de Fer): full 0.705 · sin_regex 0.711 · <strong>status_campo 0.733</strong> · intrínseco 0.150.</p>
</section>''')

TABLE = '''
<section class="card" id="tabla">
  <div class="eyebrow">RESUMEN · ORDEN = RANKING POR MEJOR VARIANTE</div>
  <h2>Todas, lado a lado</h2>
  <div class="chips"><span>d_model 32 · 4 cabezas · 2 bloques · AdamW 1e-3 · early stopping por PR-AUC de val · 6 seeds</span></div>
  <div class="tablewrap"><table>
  <thead><tr><th>#</th><th>arquitectura</th><th>un token es…</th><th>secuencia</th><th>la atención cruza</th>
  <th>parámetros</th><th>base (6 seeds)</th><th>mejor variante</th><th>comando</th></tr></thead>
  <tbody>
  <tr><td>1</td><td>A · tabular</td><td>un feature</td><td class="n">14</td><td>features ↔ features</td>
      <td class="n">28.289</td><td class="n">0.794 ± 0.033</td><td class="n">0.824 (ordinal) ★</td><td class="c">--formulation features</td></tr>
  <tr><td>2</td><td>MLP (control)</td><td>— (sin atención)</td><td class="n">416 flat</td><td>—</td>
      <td class="n">126.209</td><td class="n">0.746 ± 0.036</td><td class="n">0.797 (one-hot)</td><td class="c">--arch mlp</td></tr>
  <tr><td>3</td><td>C2 · torre</td><td>un carácter (en la torre)</td><td class="n">257 + 13</td><td>solo chars ↔ chars</td>
      <td class="n">96.225</td><td class="n">0.775 ± 0.022</td><td class="n">0.775 (base)</td><td class="c">--arch tower</td></tr>
  <tr><td>4</td><td>5B · fusión</td><td>feature o RESUMEN del texto</td><td class="n">15</td><td>features ↔ resumen</td>
      <td class="n">63.969</td><td class="n">0.775 ± 0.036</td><td class="n">0.775 (base)</td><td class="c">--formulation fusion</td></tr>
  <tr><td>5</td><td>B+ · listwise + texto</td><td>un producto CON su texto</td><td class="n">8</td><td>producto ↔ producto</td>
      <td class="n">78.305</td><td class="n">0.749 ± 0.060 (4 seeds)</td><td class="n">—</td><td class="c">--arch listwise --listwise-texto</td></tr>
  <tr><td>6</td><td>B · listwise</td><td>un producto entero</td><td class="n">8</td><td>producto ↔ producto</td>
      <td class="n">41.601</td><td class="n">0.698 ± 0.044</td><td class="n">0.740 (paciencia 20)</td><td class="c">--arch listwise</td></tr>
  <tr><td>7</td><td>A+C · híbrido</td><td>feature o carácter</td><td class="n">270</td><td>texto ↔ tabular</td>
      <td class="n">39.073</td><td class="n">0.705 ± 0.062</td><td class="n">0.736 (sin_regex, p20)</td><td class="c">--formulation hybrid</td></tr>
  <tr><td>8</td><td>C · texto</td><td>un carácter</td><td class="n">257</td><td>chars ↔ chars</td>
      <td class="n">35.713</td><td class="n">0.652 ± 0.039</td><td class="n">0.652 (base)</td><td class="c">--formulation text</td></tr>
  <tr class="ref"><td>—</td><td>GBM (referencia)</td><td>árboles, sin red</td><td class="n">—</td><td>—</td>
      <td class="n">—</td><td class="n">0.762</td><td class="n">—</td><td class="c">eda/verificaciones.py</td></tr>
  <tr class="ref"><td>—</td><td>logística (referencia)</td><td>lineal</td><td class="n">—</td><td>—</td>
      <td class="n">—</td><td class="n">0.660</td><td class="n">—</td><td class="c">eda/verificaciones.py</td></tr>
  </tbody></table></div>
  <p class="caption">Números de la suite en la 3070 (test PR-AUC, 6 seeds, protocolo base). El transformer tabular
  <strong>supera al GBM y al MLP</strong> (atención: +0.048 apareado, gana 5/6 seeds) y el <strong>modelo final del TP</strong> es
  <strong><code>feat_ordinal</code> 0.824 ± 0.018</strong> (categóricas como su rango de BTR en train; elegido por
  empate técnico en validación + parsimonia — 26k parámetros — y confirmado por test; ver
  <code>analisis.md §8.2</code>). Ninguna variante con texto supera al tabular puro; el
  hallazgo textual es otro: el híbrido <strong>recupera desde los chars</strong> la señal del regex (sin_regex ≈
  full) y la torre no (−0.04). Análisis completo: <code>analisis.md</code>.</p>
</section>'''

HTML = f'''<title>Zoo de arquitecturas BTR</title>
<style>{CSS}</style>
<div class="wrap">
<header class="top">
  <div class="eyebrow">TP1 · 73.69 LLM · REFERENCIA VISUAL</div>
  <h1>Zoo de arquitecturas BTR</h1>
  <p class="lede">Las maneras implementadas de estimar <code>p(bought)</code> por impresión, dibujadas
  completas y <strong>ordenadas por ranking</strong> (de la que mejor dio a la que peor, por la mejor variante
  de cada una — PR-AUC test, 6 seeds). Todos los diagramas usan las dimensiones reales del código
  (<code>btr/model.py</code>) y comparten el mismo código de color, así que las diferencias entre
  arquitecturas son exactamente lo que cambia de figura a figura.</p>
  <nav class="jump">
    <a href="#p0">0 · preprocesamiento</a><a href="#feat">#1 tabular</a><a href="#mlp">#2 MLP</a>
    <a href="#tower">#3 torre</a><a href="#fusion">#4 fusión</a><a href="#listwise_texto">#5 listwise+texto</a>
    <a href="#listwise">#6 listwise</a><a href="#hybrid">#7 híbrido</a><a href="#text">#8 texto</a>
    <a href="#familias">familias</a><a href="#tabla">tabla</a>
  </nav>
</header>

<section class="card" id="leyenda">
  <div class="eyebrow">CÓMO LEER LOS DIAGRAMAS</div>
  <div class="legend">
    <span><i class="sw sw-cls"></i>[CLS] aprendido</span>
    <span><i class="sw sw-tab"></i>token categórico</span>
    <span><i class="sw sw-num"></i>token numérico</span>
    <span><i class="sw sw-txt"></i>token de carácter</span>
    <span><i class="sw sw-hot"></i>señal de estado / resumen</span>
    <span><i class="sw sw-prod"></i>producto colapsado</span>
    <span><i class="sw sw-pad"></i>padding (enmascarado)</span>
    <span><i class="sw sw-blk"></i>bloque de atención</span>
    <span><i class="sw sw-mlp"></i>capa lineal / MLP</span>
    <span><i class="sw sw-out"></i>salida</span>
  </div>
  <p class="caption">Las flechas anotan el shape del tensor: <code>(B, 14, 32)</code> =
  (batch, tokens, d_model). Config base en todos: d_model 32, 4 cabezas de dim 8, 2 bloques pre-LN,
  dropout 0.1 — cada una tiene además sus variantes en <code>experimentos.py</code> (52 configs × 6 seeds).</p>
</section>

{''.join(SECTIONS)}
{TABLE}

<footer>Generado desde el código real del repo con <code>zoo.py</code>: <code>btr/model.py</code> ·
<code>btr/data.py</code> · <code>experimentos.py</code> — actualizado tras la 3ª tanda GPU (18/08/2026).
La versión para el repo (mermaid, la renderiza GitHub) está en <code>diagramas.md</code>; el análisis con
todos los números, en <code>analisis.md</code>.</footer>
</div>
'''

if __name__ == '__main__':
    from pathlib import Path
    out = Path(__file__).resolve().parent / 'zoo.html'
    out.write_text(HTML)
    print(f'escrito {out.name} ({len(HTML) / 1024:.0f} KB)')
