"""Modelo Transformer para prediccion de BTR.

Los bloques Head / MultiHeadAttention / FeedForward / Block estan tomados de la
demo de la catedra (decoder-only a nivel caracteres, basada en el tutorial de
Karpathy), con dos adaptaciones justificadas en propuesta.md (seccion 5):

1. La mascara causal es opcional (`causal=False` por defecto): nuestro problema
   es clasificacion, no generacion autoregresiva, asi que corresponde atencion
   bidireccional (encoder-only, como BERT).
2. El escalado de los scores de atencion se hace UNA sola vez por sqrt(d_k),
   como en el paper. La demo escalaba dos veces (multiplicaba por
   k.shape[-1]**-0.5 y ademas dividia por head_size**0.5).

El modelo soporta tres formulaciones de "que es un token" (propuesta.md, 4):
  'features': cada feature tabular es un token (estilo FT-Transformer). Secuencia
              corta (14), sin positional encoding por defecto (los features no
              tienen orden; cada uno tiene identidad propia por su tabla).
  'text':     cada CARACTER de title+description es un token, igual que la demo.
              Secuencia larga (~250) -> aca el positional encoding es NECESARIO
              (el texto si tiene orden) y el costo computacional es real (GPU).
  'hybrid':   [CLS] + tokens de features + tokens de caracteres en una misma
              secuencia; la atencion puede cruzar texto con features.

En los tres casos la salida es la del token [CLS]: Linear(d_model -> 1 logit),
sigmoide = p(bought), con binary cross-entropy (en la demo: vocab_size logits
por posicion + cross-entropy contra el proximo caracter).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

PAD_IDX = 0  # debe coincidir con data.PAD_IDX


class Head(nn.Module):
    """Una cabeza de self-attention (demo, con mascara causal opcional)."""

    def __init__(self, head_size, n_embd, context_length, dropout, causal=False):
        super().__init__()
        self.causal = causal
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        if causal:
            self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # x: (batch, seq, n_embd); padding_mask: (batch, seq) con True en posiciones reales
        _, seq_len, _ = x.shape
        k = self.key(x)    # (batch, seq, head_size)
        q = self.query(x)  # (batch, seq, head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (batch, seq, seq)
        if self.causal:
            wei = wei.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
        if padding_mask is not None:
            # -1e9 en lugar de -inf: una fila totalmente enmascarada (token de
            # padding atendiendo) daria NaN con -inf; con -1e9 da uniforme y su
            # salida se descarta igual.
            wei = wei.masked_fill(~padding_mask[:, None, :], -1e9)
        wei = F.softmax(wei, dim=-1)
        if getattr(self, 'guardar_atencion', False):  # para eda/atencion.py (solo eval)
            self.ultima_atencion = wei.detach()
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v  # (batch, seq, head_size)


class HeadPorFeature(nn.Module):
    """Cabeza de self-attention con W_q/W_k/W_v PROPIOS POR POSICION (idea de Fer).

    En texto las posiciones son intercambiables y compartir W_q/W_k/W_v es el
    sesgo inductivo correcto. Aca la posicion ES el feature (token 1 = status,
    token 8 = price, siempre): desatar los pesos deja que cada feature haga sus
    propias preguntas/claves/valores. q_t = x_t @ Wq[t], via einsum. Solo tiene
    sentido con secuencia de identidad fija (formulation features, T=14).
    """

    def __init__(self, head_size, n_embd, context_length, dropout):
        super().__init__()
        def par():
            w = nn.Parameter(torch.empty(context_length, n_embd, head_size))
            nn.init.normal_(w, mean=0.0, std=0.02)
            return w
        self.wq, self.wk, self.wv = par(), par(), par()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # x: (batch, T, d); W*: (T, d, hs) -> por posicion
        q = torch.einsum('btd,tdh->bth', x, self.wq)
        k = torch.einsum('btd,tdh->bth', x, self.wk)
        v = torch.einsum('btd,tdh->bth', x, self.wv)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        if padding_mask is not None:
            wei = wei.masked_fill(~padding_mask[:, None, :], -1e9)
        wei = F.softmax(wei, dim=-1)
        if getattr(self, 'guardar_atencion', False):
            self.ultima_atencion = wei.detach()
        wei = self.dropout(wei)
        return wei @ v


class HeadPorFeatureGate(nn.Module):
    """Especializacion POR FEATURE con presupuesto minimo (contrapeso de la idea
    de los pesos desatados): en vez de un W_q/W_k/W_v entero por posicion, cada
    posicion tiene un vector de COMPUERTAS diagonal que modula la entrada a los
    W compartidos:  q_t = (x_t * g_q[t]) @ W_q. ~1% de los parametros del
    desatado completo; con g inicializado en 1, el modelo arranca siendo
    EXACTAMENTE el compartido y aprende solo la desviacion por feature.
    """

    def __init__(self, head_size, n_embd, context_length, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.gq = nn.Parameter(torch.ones(context_length, n_embd))
        self.gk = nn.Parameter(torch.ones(context_length, n_embd))
        self.gv = nn.Parameter(torch.ones(context_length, n_embd))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        q = self.query(x * self.gq)
        k = self.key(x * self.gk)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        if padding_mask is not None:
            wei = wei.masked_fill(~padding_mask[:, None, :], -1e9)
        wei = F.softmax(wei, dim=-1)
        if getattr(self, 'guardar_atencion', False):
            self.ultima_atencion = wei.detach()
        wei = self.dropout(wei)
        return wei @ self.value(x * self.gv)


class FeedForwardPorFeature(nn.Module):
    """FFN con parametros PROPIOS POR POSICION: 14 MLPs, una por feature (+CLS)."""

    def __init__(self, n_embd, context_length, dropout):
        super().__init__()
        def par(*shape):
            w = nn.Parameter(torch.empty(*shape))
            nn.init.normal_(w, mean=0.0, std=0.02)
            return w
        self.w1 = par(context_length, n_embd, 4 * n_embd)
        self.b1 = nn.Parameter(torch.zeros(context_length, 4 * n_embd))
        self.w2 = par(context_length, 4 * n_embd, n_embd)
        self.b2 = nn.Parameter(torch.zeros(context_length, n_embd))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.relu(torch.einsum('btd,tdh->bth', x, self.w1) + self.b1)
        return self.dropout(torch.einsum('bth,thd->btd', h, self.w2) + self.b2)


class MultiHeadAttention(nn.Module):
    """Varias cabezas en paralelo + proyeccion (igual que la demo)."""

    def __init__(self, num_heads, head_size, n_embd, context_length, dropout, causal=False,
                 por_feature=False):
        super().__init__()
        if por_feature == 'full':  # W_q/W_k/W_v propios por posicion
            self.heads = nn.ModuleList(
                [HeadPorFeature(head_size, n_embd, context_length, dropout)
                 for _ in range(num_heads)]
            )
        elif por_feature == 'gate':  # compuertas diagonales por posicion (barato)
            self.heads = nn.ModuleList(
                [HeadPorFeatureGate(head_size, n_embd, context_length, dropout)
                 for _ in range(num_heads)]
            )
        else:
            self.heads = nn.ModuleList(
                [Head(head_size, n_embd, context_length, dropout, causal)
                 for _ in range(num_heads)]
            )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        out = torch.cat([h(x, padding_mask) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """MLP interno del bloque: expansion x4 + no linealidad (igual que la demo)."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Bloque transformer pre-LN con conexiones residuales (igual que la demo)."""

    def __init__(self, n_embd, n_head, context_length, dropout, causal=False,
                 per_feature='none'):
        super().__init__()
        head_size = n_embd // n_head
        pf_sa = 'full' if per_feature in ('qkv', 'both') else \
            ('gate' if per_feature == 'gate' else None)
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, context_length, dropout, causal,
                                     por_feature=pf_sa)
        if per_feature in ('ffn', 'both'):
            self.ffwd = FeedForwardPorFeature(n_embd, context_length, dropout)
        else:
            self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, padding_mask=None):
        x = x + self.sa(self.ln1(x), padding_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class FeatureTokenizer(nn.Module):
    """Convierte una fila tabular en una secuencia de tokens de d_model.

    Es el analogo del token_embedding_table de la demo: alli habia UN vocabulario
    de 65 caracteres compartido; aca cada feature categorica tiene su propia
    tabla (nn.Embedding) y cada numerica su propia proyeccion afin a d_model
    (estilo FT-Transformer). El indice 0 de cada tabla queda reservado para
    valores no vistos en train (UNK).

    numeric_mode:
      'linear': token_i = x_i * w_i + b_i  (w_i, b_i vectores aprendidos)
      'bins':   x_i se discretiza en cuantiles (bin_edges de train) y se hace
                lookup en una tabla, igual que una categorica. Captura efectos
                no monotonicos (la U invertida del precio) sin depender del MLP.

    cat_encoding (propuesta 6.1; one-hot+lineal ≡ embedding, por eso no esta aca):
      'embedding': una tabla aprendida por columna (column embeddings, el default)
      'target':    nivel -> BTR promedio suavizado de train (cat_tables, float);
                   el escalar entra con una proyeccion afin como una numerica
      'ordinal':   nivel -> RANGO del nivel al ordenar por ese BTR de train,
                   normalizado a [0,1]: conserva el orden pero no las magnitudes
                   (un orden "a mano" por semantica seria indefendible: el EDA
                   mostro que el wording no predice el tier)
      'freq':      nivel -> frecuencia relativa en train; idem proyeccion afin
      'hashing':   nivel -> bucket = hash(valor) % B (cat_tables, long) y lookup
                   en una tabla de B entradas por feature ("truco del modulo":
                   util con miles de niveles; aca mide cuanto duele la colision)

    cat_modes permite un encoding POR feature (p. ej. solo listing_status en
    ordinal y el resto embedding): lista alineada a cat_cardinalities; si se
    omite, todas usan cat_encoding. cat_tables: dict {posicion: tensor lookup}
    para las que lo requieren.
    """

    SCALAR_MODES = ('target', 'freq', 'ordinal')  # nivel -> escalar -> afin a d_model

    def __init__(self, cat_cardinalities, n_numeric, d_model,
                 numeric_mode='linear', bin_edges=None,
                 cat_encoding='embedding', cat_tables=None, hash_buckets=8,
                 cat_modes=None):
        super().__init__()
        self.numeric_mode = numeric_mode
        self.cat_encoding = cat_encoding
        self.n_cat = len(cat_cardinalities)
        modes = list(cat_modes) if cat_modes else [cat_encoding] * self.n_cat
        assert len(modes) == self.n_cat, 'cat_modes debe alinear con cat_cardinalities'
        self.cat_modes = modes
        if all(m == 'embedding' for m in modes):
            # camino identico al original: los checkpoints viejos cargan tal cual
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(card, d_model) for card in cat_cardinalities]
            )
        else:
            cat_tables = cat_tables or {}
            embs = {}
            self.cat_scalar_weight = nn.ParameterDict()
            self.cat_scalar_bias = nn.ParameterDict()
            for i, (m, card) in enumerate(zip(modes, cat_cardinalities)):
                if m == 'embedding':
                    embs[str(i)] = nn.Embedding(card, d_model)
                elif m == 'hashing':
                    if i not in cat_tables:
                        raise ValueError(f'hashing en feature {i} requiere cat_tables[{i}]')
                    self.register_buffer(f'cat_table_{i}', cat_tables[i].long())
                    embs[str(i)] = nn.Embedding(hash_buckets, d_model)
                elif m in self.SCALAR_MODES:
                    if i not in cat_tables:
                        raise ValueError(f'{m} en feature {i} requiere cat_tables[{i}]')
                    self.register_buffer(f'cat_table_{i}', cat_tables[i].float())
                    w = nn.Parameter(torch.empty(d_model))
                    nn.init.normal_(w, mean=0.0, std=0.02)
                    self.cat_scalar_weight[str(i)] = w
                    self.cat_scalar_bias[str(i)] = nn.Parameter(torch.zeros(d_model))
                else:
                    raise ValueError(f'cat_encoding desconocido: {m}')
            self.cat_embeddings = nn.ModuleDict(embs)

        if numeric_mode == 'linear':
            self.num_weight = nn.Parameter(torch.empty(n_numeric, d_model))
            self.num_bias = nn.Parameter(torch.zeros(n_numeric, d_model))
            nn.init.normal_(self.num_weight, mean=0.0, std=0.02)
        elif numeric_mode == 'bins':
            if bin_edges is None:
                raise ValueError("numeric_mode='bins' requiere bin_edges (cuantiles de train)")
            # bin_edges: (n_numeric, n_bins-1) -> bucketize da valores en [0, n_bins-1]
            self.register_buffer('bin_edges', bin_edges)
            n_bins = bin_edges.shape[1] + 1
            self.num_bin_embeddings = nn.ModuleList(
                [nn.Embedding(n_bins, d_model) for _ in range(n_numeric)]
            )
        else:
            raise ValueError(f'numeric_mode desconocido: {numeric_mode}')

    @property
    def n_tokens(self):
        n_num = self.num_weight.shape[0] if self.numeric_mode == 'linear' \
            else len(self.num_bin_embeddings)
        return self.n_cat + n_num

    def _cat_tokens(self, x_cat):
        if isinstance(self.cat_embeddings, nn.ModuleList):  # todo embedding (camino original)
            return [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        out = []
        for i, m in enumerate(self.cat_modes):
            if m == 'embedding':
                out.append(self.cat_embeddings[str(i)](x_cat[:, i]))
            elif m == 'hashing':
                out.append(self.cat_embeddings[str(i)](getattr(self, f'cat_table_{i}')[x_cat[:, i]]))
            else:  # target / freq / ordinal: escalar por nivel -> afin a d_model
                v = getattr(self, f'cat_table_{i}')[x_cat[:, i]].unsqueeze(-1)
                out.append(v * self.cat_scalar_weight[str(i)] + self.cat_scalar_bias[str(i)])
        return out

    def forward(self, x_cat, x_num):
        # x_cat: (batch, n_cat) long; x_num: (batch, n_num) float
        tokens = self._cat_tokens(x_cat)
        if self.numeric_mode == 'linear':
            # (batch, n_num, 1) * (n_num, d_model) -> (batch, n_num, d_model)
            num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
            tokens = torch.stack(tokens, dim=1)
            tokens = torch.cat([tokens, num_tokens], dim=1)
        else:
            for i, emb in enumerate(self.num_bin_embeddings):
                idx = torch.bucketize(x_num[:, i], self.bin_edges[i])
                tokens.append(emb(idx))
            tokens = torch.stack(tokens, dim=1)
        return tokens  # (batch, n_cat + n_num, d_model)


def _bce_multitask(logits, cart_logits, targets, pos_weight, cart_lambda):
    """BCE del target real + lambda * BCE de cart (multi-task, propuesta de Junior).

    targets: (batch,) = solo bought, o (batch, 2) = [bought, cart]. cart entra
    UNICAMENTE como etiqueta auxiliar de entrenamiento — nunca como input — y su
    cabeza se ignora en inferencia (predict_proba expone solo p(bought)).
    """
    if targets.dim() == 2:
        y, y_cart = targets[:, 0], targets[:, 1]
    else:
        y, y_cart = targets, None
    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    if cart_lambda and y_cart is not None and cart_logits is not None:
        loss = loss + cart_lambda * F.binary_cross_entropy_with_logits(cart_logits, y_cart)
    return loss


class TextEncoder(nn.Module):
    """Torre de caracteres reutilizable: [CLS] + chars + PE -> bloques -> (B, d_model).

    Mismas piezas que TextTowerModel.encode_text, como modulo aparte para poder
    meterla DENTRO de otra arquitectura (hoy: el token de producto del listwise
    enriquecido con texto, propuesta #1 de junior_proposals.md).
    """

    def __init__(self, char_vocab_size, max_text_len, d_model, n_head=4, n_layer=2,
                 dropout=0.1):
        super().__init__()
        self.char_embedding_table = nn.Embedding(char_vocab_size, d_model, padding_idx=PAD_IDX)
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        seq_len = 1 + max_text_len
        self.position_embedding_table = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, seq_len, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x_text):
        batch_size = x_text.shape[0]
        x = torch.cat([self.cls.expand(batch_size, -1, -1),
                       self.char_embedding_table(x_text)], dim=1)
        mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=x_text.device),
                          x_text != PAD_IDX], dim=1)
        x = x + self.position_embedding_table(torch.arange(x.shape[1], device=x.device))
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)[:, 0]


class BTRTransformer(nn.Module):
    """Encoder-only transformer que estima p(bought) por impresion.

    Mismo esqueleto que TransformerLanguageModel de la demo (embeddings ->
    bloques -> layer norm final -> proyeccion), con la entrada segun la
    formulacion elegida (ver docstring del modulo) y salida [CLS] -> 1 logit.
    """

    def __init__(self, formulation='features', cat_cardinalities=None, n_numeric=0,
                 char_vocab_size=None, max_text_len=0, d_model=32, n_head=4,
                 n_layer=2, dropout=0.1, causal=False, pooling='cls',
                 use_positional=False, numeric_mode='linear', bin_edges=None,
                 pos_weight=None, cls_position='first', cat_encoding='embedding',
                 cat_tables=None, hash_buckets=8, cart_lambda=0.0, cat_modes=None,
                 per_feature='none'):
        super().__init__()
        assert d_model % n_head == 0, 'd_model debe ser multiplo de n_head'
        assert pooling in ('cls', 'mean')
        assert formulation in ('features', 'text', 'hybrid', 'fusion')
        assert cls_position in ('first', 'last')
        # con mascara causal el CLS en posicion 0 solo se ve a si mismo y el modelo
        # degenera a predecir la tasa base (medido: ROC 0.500 exacto, p constante).
        # El causal "bien hecho" pone el CLS al FINAL, como el ultimo token de GPT.
        if causal and cls_position == 'first' and pooling == 'cls':
            import warnings
            warnings.warn('causal con CLS en posicion 0 es degenerado; usar cls_position="last"')
        self.formulation = formulation
        self.pooling = pooling
        self.cls_position = cls_position
        self.cart_lambda = cart_lambda

        seq_len = 1  # [CLS]
        self.tokenizer = None
        if formulation in ('features', 'hybrid', 'fusion'):
            self.tokenizer = FeatureTokenizer(
                cat_cardinalities, n_numeric, d_model, numeric_mode, bin_edges,
                cat_encoding, cat_tables, hash_buckets, cat_modes
            )
            seq_len += self.tokenizer.n_tokens
        self.char_embedding_table = None
        self.text_encoder = None
        if formulation in ('text', 'hybrid'):
            assert char_vocab_size and max_text_len, 'text/hybrid requieren vocabulario de chars'
            self.char_embedding_table = nn.Embedding(char_vocab_size, d_model, padding_idx=PAD_IDX)
            seq_len += max_text_len
        elif formulation == 'fusion':
            # 5b de la revision externa: una torre resume el texto a UN vector via su
            # CLS, y ese vector entra como UN token mas de la secuencia tabular — la
            # atencion cruza texto y features al nivel del RESUMEN, sin que 256 chars
            # diluyan a los 13 tokens (el problema medido del hybrid)
            assert char_vocab_size and max_text_len, 'fusion requiere vocabulario de texto'
            self.text_encoder = TextEncoder(char_vocab_size, max_text_len, d_model,
                                            n_head, n_layer, dropout)
            seq_len += 1
        self.seq_len = seq_len

        # [CLS] aprendido, como BERT: agrega la informacion para clasificar
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        # El texto tiene orden -> con chars el positional encoding es necesario;
        # con solo features es redundante (queda como ablacion, propuesta.md 3).
        if formulation in ('text', 'hybrid'):
            use_positional = True
        self.position_embedding_table = (
            nn.Embedding(seq_len, d_model) if use_positional else None
        )

        assert per_feature in ('none', 'qkv', 'ffn', 'both', 'gate')
        if per_feature != 'none':
            # pesos por posicion: solo con secuencia de identidad fija (features)
            assert formulation == 'features' and not causal, \
                'per-feature requiere formulation=features sin causal'
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, seq_len, dropout, causal, per_feature)
             for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, 1)
        # cabeza auxiliar multi-task sobre el mismo pooled (--cart-aux); se
        # descarta en inferencia
        self.cart_head = nn.Linear(d_model, 1) if cart_lambda else None
        # peso para la clase positiva en la BCE (desbalance 87/13), opcional
        self.register_buffer(
            'pos_weight',
            torch.tensor(float(pos_weight)) if pos_weight is not None else None,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # misma inicializacion que la demo
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_sequence(self, x_cat, x_num, x_text):
        """Arma [CLS | features | chars] (o el CLS al final) + su padding mask."""
        batch_size = (x_text if self.formulation == 'text' else x_cat).shape[0]
        device = (x_text if self.formulation == 'text' else x_cat).device
        parts, masks = [], []
        if self.tokenizer is not None:
            feat = self.tokenizer(x_cat, x_num)
            parts.append(feat)
            masks.append(torch.ones(batch_size, feat.shape[1], dtype=torch.bool, device=device))
        if self.text_encoder is not None:  # fusion: el resumen del texto como un token
            parts.append(self.text_encoder(x_text).unsqueeze(1))
            masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
        if self.char_embedding_table is not None:
            parts.append(self.char_embedding_table(x_text))
            masks.append(x_text != PAD_IDX)  # el padding del texto no recibe atencion
        cls = [self.cls.expand(batch_size, -1, -1)]
        cls_mask = [torch.ones(batch_size, 1, dtype=torch.bool, device=device)]
        if self.cls_position == 'first':
            parts, masks = cls + parts, cls_mask + masks
        else:  # 'last': con causal, el CLS al final es el unico que ve todo
            parts, masks = parts + cls, masks + cls_mask
        x = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        return x, (mask if self.char_embedding_table is not None else None)

    def forward(self, x_cat=None, x_num=None, x_text=None, targets=None):
        x, padding_mask = self._build_sequence(x_cat, x_num, x_text)
        if self.position_embedding_table is not None:
            positions = torch.arange(x.shape[1], device=x.device)
            x = x + self.position_embedding_table(positions)
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.ln_f(x)

        if self.pooling == 'cls':
            pooled = x[:, 0] if self.cls_position == 'first' else x[:, -1]
        else:  # promedio solo sobre posiciones reales
            m = (padding_mask if padding_mask is not None
                 else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device))
            pooled = (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
        logits = self.cls_head(pooled).squeeze(-1)  # (batch,)

        loss = None
        if targets is not None:
            cart_logits = self.cart_head(pooled).squeeze(-1) if self.cart_head is not None else None
            loss = _bce_multitask(logits, cart_logits, targets, self.pos_weight, self.cart_lambda)
        return logits, loss

    @torch.no_grad()
    def predict_proba(self, x_cat=None, x_num=None, x_text=None, batch_size=4096):
        """p(bought) por fila, en eval mode y por lotes (analogo a generate(), sin autoregresion)."""
        self.eval()
        n = (x_text if self.formulation == 'text' else x_cat).shape[0]
        sl = lambda t, a, b: None if t is None else t[a:b]
        probs = []
        for start in range(0, n, batch_size):
            end = start + batch_size
            logits, _ = self(sl(x_cat, start, end), sl(x_num, start, end), sl(x_text, start, end))
            probs.append(torch.sigmoid(logits))
        return torch.cat(probs)


class MLPBaseline(nn.Module):
    """Baseline SIN atencion, para la pregunta central "¿la atencion aporta?".

    Usa exactamente la misma entrada que el transformer tabular (los tokens del
    FeatureTokenizer) pero los concatena y los mezcla con un MLP denso, como en
    SIA. Si el transformer no supera a esto, la capa de atencion no se justifica.
    """

    def __init__(self, cat_cardinalities, n_numeric, d_model=32, dropout=0.1,
                 numeric_mode='linear', bin_edges=None, pos_weight=None,
                 cat_encoding='embedding', cat_tables=None, hash_buckets=8,
                 cart_lambda=0.0, cat_modes=None):
        super().__init__()
        self.cart_lambda = cart_lambda
        self.onehot = cat_encoding == 'onehot'
        if self.onehot:
            # one-hot crudo de las categoricas + tokens afines de las numericas:
            # mide que aporta EMBEBER (para el transformer one-hot+lineal ≡
            # embedding, propuesta 6.1; la comparacion solo tiene sentido aca)
            if numeric_mode != 'linear':
                raise ValueError('cat_encoding=onehot solo con numeric_mode=linear')
            self.cat_cardinalities = list(cat_cardinalities)
            self.tokenizer = FeatureTokenizer([], n_numeric, d_model,  # solo la parte numerica
                                              numeric_mode, bin_edges)
            in_dim = sum(cat_cardinalities) + n_numeric * d_model
        else:
            self.tokenizer = FeatureTokenizer(cat_cardinalities, n_numeric, d_model,
                                              numeric_mode, bin_edges,
                                              cat_encoding, cat_tables, hash_buckets, cat_modes)
            in_dim = self.tokenizer.n_tokens * d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, 8 * d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(8 * d_model, 2 * d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, 1),
        )
        self.cart_head = nn.Linear(2 * d_model, 1) if cart_lambda else None
        self.register_buffer(
            'pos_weight',
            torch.tensor(float(pos_weight)) if pos_weight is not None else None,
        )

    def _entrada(self, x_cat, x_num):
        if self.onehot:
            oh = [F.one_hot(x_cat[:, i], card).float()
                  for i, card in enumerate(self.cat_cardinalities)]
            num_tokens = x_num.unsqueeze(-1) * self.tokenizer.num_weight + self.tokenizer.num_bias
            return torch.cat(oh + [num_tokens.flatten(1)], dim=1)
        return self.tokenizer(x_cat, x_num).flatten(1)

    def forward(self, x_cat, x_num, x_text=None, targets=None):
        h = self._entrada(x_cat, x_num)                    # (batch, in_dim)
        # forward por capas para exponer la penultima activacion a la cabeza de cart
        pen = self.net[:-1](h)                             # (batch, 2*d_model)
        logits = self.net[-1](pen).squeeze(-1)             # (batch,)
        loss = None
        if targets is not None:
            cart_logits = self.cart_head(pen).squeeze(-1) if self.cart_head is not None else None
            loss = _bce_multitask(logits, cart_logits, targets, self.pos_weight, self.cart_lambda)
        return logits, loss


class TextTowerModel(nn.Module):
    """Transformer SOLO como encoder de texto; la clasificacion la hace un MLP.

    La torre de texto ([CLS] + chars -> bloques -> embedding del CLS) comprime
    title+description a UN vector de d_model. Ese vector se concatena con los
    feature-tokens tabulares aplanados y un MLP clasifica. A diferencia de la
    formulacion 'hybrid' del BTRTransformer, aca la atencion NO puede cruzar
    texto <-> features: compara "transformer como modulo de embedding" (estilo
    encoder-only de BERT, clase 2) contra "transformer como clasificador".
    """

    def __init__(self, cat_cardinalities, n_numeric, char_vocab_size, max_text_len,
                 d_model=32, n_head=4, n_layer=2, dropout=0.1, causal=False,
                 numeric_mode='linear', bin_edges=None, pos_weight=None,
                 cat_encoding='embedding', cat_tables=None, hash_buckets=8,
                 cart_lambda=0.0, cat_modes=None):
        super().__init__()
        self.cart_lambda = cart_lambda
        self.char_embedding_table = nn.Embedding(char_vocab_size, d_model, padding_idx=PAD_IDX)
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        seq_len = 1 + max_text_len
        self.position_embedding_table = nn.Embedding(seq_len, d_model)  # texto: PE necesario
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, seq_len, dropout, causal) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        self.tokenizer = FeatureTokenizer(cat_cardinalities, n_numeric, d_model,
                                          numeric_mode, bin_edges,
                                          cat_encoding, cat_tables, hash_buckets, cat_modes)
        head_in = d_model + self.tokenizer.n_tokens * d_model
        self.head = nn.Sequential(
            nn.Linear(head_in, 4 * d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, 1),
        )
        self.cart_head = nn.Linear(head_in, 1) if cart_lambda else None
        self.register_buffer(
            'pos_weight',
            torch.tensor(float(pos_weight)) if pos_weight is not None else None,
        )

    def encode_text(self, x_text):
        """(batch, max_text_len) -> (batch, d_model): el embedding [CLS] del texto."""
        batch_size = x_text.shape[0]
        x = torch.cat([self.cls.expand(batch_size, -1, -1),
                       self.char_embedding_table(x_text)], dim=1)
        mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=x_text.device),
                          x_text != PAD_IDX], dim=1)
        x = x + self.position_embedding_table(torch.arange(x.shape[1], device=x.device))
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)[:, 0]

    def forward(self, x_cat, x_num, x_text, targets=None):
        text_emb = self.encode_text(x_text)                     # (batch, d_model)
        tab = self.tokenizer(x_cat, x_num).flatten(1)           # (batch, n_tokens*d_model)
        h = torch.cat([text_emb, tab], dim=1)
        logits = self.head(h).squeeze(-1)
        loss = None
        if targets is not None:
            cart_logits = self.cart_head(h).squeeze(-1) if self.cart_head is not None else None
            loss = _bce_multitask(logits, cart_logits, targets, self.pos_weight, self.cart_lambda)
        return logits, loss


class ListwiseTransformer(nn.Module):
    """Formulacion B (propuesta.md 4B): los tokens son los PRODUCTOS de la pagina.

    Cada producto se colapsa a un vector d_model (feature-tokens aplanados +
    proyeccion) y la atencion corre ENTRE los productos de la misma query:
    ¿me compran a mi dado lo que aparece al lado? Salida: un logit POR producto
    (como los logits por posicion de la demo, pero con sigmoide) y la loss se
    calcula solo sobre los slots reales (padding_mask sobre el padding de la
    pagina). Sin positional encoding: el dataset no tiene columna de posicion
    en la pagina, asi que no hay orden que codificar.

    Con use_text (--listwise-texto, propuesta #1 de junior_proposals.md) cada
    producto suma el embedding de su title+description via una TextEncoder antes
    de la proyeccion: el tercer argumento del forward pasa a ser x_text (B, P, L)
    y la mascara de productos se reconstruye como (x_text != PAD).any(-1).
    Responde: ¿listwise pierde por la idea (comparar vecinos) o porque su token
    de producto era ciego a la senal mas fuerte (el sufijo del titulo)?
    """

    def __init__(self, cat_cardinalities, n_numeric, max_products, d_model=32,
                 n_head=4, n_layer=2, dropout=0.1, numeric_mode='linear',
                 bin_edges=None, pos_weight=None, use_text=False,
                 char_vocab_size=None, max_text_len=0):
        super().__init__()
        self.use_text = use_text
        self.tokenizer = FeatureTokenizer(cat_cardinalities, n_numeric, d_model,
                                          numeric_mode, bin_edges)
        proj_in = self.tokenizer.n_tokens * d_model
        if use_text:
            assert char_vocab_size and max_text_len, 'listwise-texto requiere vocabulario de chars'
            self.text_encoder = TextEncoder(char_vocab_size, max_text_len, d_model,
                                            n_head, n_layer, dropout)
            proj_in += d_model
        self.product_proj = nn.Linear(proj_in, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, max_products, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.logit_head = nn.Linear(d_model, 1)
        self.register_buffer(
            'pos_weight',
            torch.tensor(float(pos_weight)) if pos_weight is not None else None,
        )

    def forward(self, x_cat, x_num, mask_or_text, targets=None):
        # x_cat: (batch, P, n_cat); x_num: (batch, P, n_num)
        # mask_or_text: prod_mask (batch, P) bool, o x_text (batch, P, L) long
        batch_size, n_products, _ = x_cat.shape
        tokens = self.tokenizer(x_cat.flatten(0, 1), x_num.flatten(0, 1))  # (B*P, T, d)
        flat = tokens.flatten(1)                                           # (B*P, T*d)
        if self.use_text:
            x_text = mask_or_text
            prod_mask = (x_text != PAD_IDX).any(-1)      # slots de padding = texto todo-PAD
            text_emb = self.text_encoder(x_text.flatten(0, 1))             # (B*P, d)
            flat = torch.cat([flat, text_emb], dim=1)
        else:
            prod_mask = mask_or_text
        products = self.product_proj(flat)                                 # (B*P, d)
        x = products.view(batch_size, n_products, -1)                      # (B, P, d)
        for block in self.blocks:
            x = block(x, prod_mask)
        logits = self.logit_head(self.ln_f(x)).squeeze(-1)                 # (B, P)
        loss = None
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits[prod_mask], targets[prod_mask], pos_weight=self.pos_weight
            )
        return logits, loss


ARCHITECTURES = {
    'transformer': BTRTransformer,
    'mlp': MLPBaseline,
    'tower': TextTowerModel,
    'listwise': ListwiseTransformer,
}


def load_checkpoint(path, device='cpu'):
    """Recarga un checkpoint de pesos/ y devuelve (model en eval mode, preprocessor).

    El preprocessor permite transformar filas nuevas exactamente igual que en el
    entrenamiento (mismos vocabularios y estadisticos de train):

        model, prep = load_checkpoint('pesos/features_d32_h4_l2_linear_seed42.pt')
        x_cat, x_num, x_text, _ = prep.transform(df_nuevo)
        probs = model.predict_proba(x_cat, x_num, x_text)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cls = ARCHITECTURES[ckpt.get('arch', 'transformer')]
    extra = {}
    if ckpt.get('cat_tables') is not None:  # encodings target/freq/hashing
        extra['cat_tables'] = ckpt['cat_tables']
    model = cls(**ckpt['model_config'], bin_edges=ckpt['bin_edges'], **extra).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['preprocessor']
