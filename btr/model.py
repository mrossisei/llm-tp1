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

Formulaciones de "que es un token" del BTRTransformer (propuesta.md, 4):
  'features':   cada feature tabular es un token (estilo FT-Transformer). Secuencia
                corta (14), sin positional encoding (los features no tienen orden;
                cada uno tiene identidad propia por su posicion y su tabla).
                ES EL MODELO FINAL (con encoding ordinal de las categoricas).
  'ing_fusion': los ingredientes como CONJUNTO: un IngredientEncoder (otro
                transformer chico) resume la lista a UN token que entra a la
                secuencia tabular — la alternativa que presentamos.
  'ing_hybrid': un token POR ingrediente directo en la secuencia tabular.
  'ing':        SOLO los ingredientes (control: ¿tienen senal por si solos?).
Transfer learning (clase 3): con text_emb_dim > 0 el embedding del TITULO que
produce un modelo preentrenado (precomputado, congelado) entra como un token
mas; con hf_model el encoder de Hugging Face entra al grafo y se fine-tunea.

En todos los casos la salida es la del token [CLS]: Linear(d_model -> 1 logit),
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


class MultiHeadAttention(nn.Module):
    """Varias cabezas en paralelo + proyeccion (igual que la demo)."""

    def __init__(self, num_heads, head_size, n_embd, context_length, dropout, causal=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, context_length, dropout, causal) for _ in range(num_heads)]
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

    def __init__(self, n_embd, n_head, context_length, dropout, causal=False):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, context_length, dropout, causal)
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

    cat_tables: dict {posicion: tensor lookup} para los encodings que lo requieren
    (target/ordinal/freq/hashing; ajustados con train en btr.train.build_cat_tables).
    """

    SCALAR_MODES = ('target', 'freq', 'ordinal')  # nivel -> escalar -> afin a d_model

    def __init__(self, cat_cardinalities, n_numeric, d_model,
                 numeric_mode='linear', bin_edges=None,
                 cat_encoding='embedding', cat_tables=None, hash_buckets=8, n_cyclic=0):
        super().__init__()
        self.numeric_mode = numeric_mode
        self.cat_encoding = cat_encoding
        self.n_cat = len(cat_cardinalities)
        # variables CICLICAS (--tiempo ciclico): cada una llega como el par (sin, cos) de su
        # angulo, al final de x_num, y se vuelve UN token con una proyeccion afin R^2 -> d_model
        # (la misma idea que la recta por numerica, en dos dimensiones): asi la hora 23 y la
        # 0 quedan cerca, cosa que un escalar ordenado no puede
        self.n_cyclic = n_cyclic
        self.n_numeric = n_numeric
        if n_cyclic:
            self.cyc_weight = nn.Parameter(torch.empty(n_cyclic, 2, d_model))
            self.cyc_bias = nn.Parameter(torch.zeros(n_cyclic, d_model))
            nn.init.normal_(self.cyc_weight, mean=0.0, std=0.02)
        modes = [cat_encoding] * self.n_cat
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
        return self.n_cat + n_num + getattr(self, 'n_cyclic', 0)

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
        # x_cat: (batch, n_cat) long; x_num: (batch, n_num [+ 2 por ciclica]) float
        n_cyc = getattr(self, 'n_cyclic', 0)
        x_cyc = None
        if n_cyc:
            x_cyc = x_num[:, -2 * n_cyc:].reshape(-1, n_cyc, 2)
            x_num = x_num[:, :-2 * n_cyc]
        tokens = self._tokens(x_cat, x_num)
        if n_cyc:
            cyc_tokens = torch.einsum('bkc,kcd->bkd', x_cyc, self.cyc_weight) + self.cyc_bias
            tokens = torch.cat([tokens, cyc_tokens], dim=1)
        return tokens

    def _tokens(self, x_cat, x_num):
        tokens = self._cat_tokens(x_cat)
        if self.numeric_mode == 'linear':
            # (batch, n_num, 1) * (n_num, d_model) -> (batch, n_num, d_model)
            num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
            if not tokens:  # sin categoricas (control "solo el titulo": --drop-features all)
                return num_tokens
            tokens = torch.stack(tokens, dim=1)
            tokens = torch.cat([tokens, num_tokens], dim=1)
        else:
            for i, emb in enumerate(self.num_bin_embeddings):
                idx = torch.bucketize(x_num[:, i], self.bin_edges[i])
                tokens.append(emb(idx))
            tokens = torch.stack(tokens, dim=1)
        return tokens  # (batch, n_cat + n_num, d_model)


class IngredientEncoder(nn.Module):
    """Encoder de CONJUNTO para la lista de ingredientes (9na tanda, idea de Fer).

    [ING] + un token POR INGREDIENTE -> bloques de self-attention -> embedding
    del [ING]: comprime la lista a UN vector de d_model que entra como un token
    mas del transformer tabular (ing_fusion). Decisiones deliberadas:
      - vocabulario de INGREDIENTES de train (data.Preprocessor): UNK para los
        no vistos, nada de test entra al vocabulario; embeddings APRENDIDOS
      - SIN positional encoding: la lista no tiene orden conocido (el orden en
        que estan escritos no significa nada) -> la salida es invariante al
        orden de la lista, como corresponde a un conjunto
      - atencion BIDIRECCIONAL todos-contra-todos (sin mascara causal: en un
        conjunto no existe "lo de atras"); el PAD no recibe atencion (mascara)
    """

    def __init__(self, ing_vocab_size, max_ingredients, d_model, n_head=4, n_layer=1,
                 dropout=0.1, d_out=None):
        super().__init__()
        self.ing_embedding_table = nn.Embedding(ing_vocab_size, d_model, padding_idx=PAD_IDX)
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        seq_len = 1 + max_ingredients
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, seq_len, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        # el encoder puede tener su propia dimension (--ing-d-model); su [ING] se proyecta a la
        # del transformer principal para entrar como un token mas
        self.proj = nn.Linear(d_model, d_out) if d_out and d_out != d_model else None

    def forward(self, x_ing):
        # x_ing: (batch, max_ingredients) long, 0 = PAD
        batch_size = x_ing.shape[0]
        x = torch.cat([self.cls.expand(batch_size, -1, -1),
                       self.ing_embedding_table(x_ing)], dim=1)
        mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=x_ing.device),
                          x_ing != PAD_IDX], dim=1)
        for block in self.blocks:
            x = block(x, mask)
        out = self.ln_f(x)[:, 0]
        return self.proj(out) if self.proj is not None else out


class BTRTransformer(nn.Module):
    """Encoder-only transformer que estima p(bought) por impresion.

    Mismo esqueleto que TransformerLanguageModel de la demo (embeddings ->
    bloques -> layer norm final -> proyeccion), con la entrada segun la
    formulacion elegida (ver docstring del modulo) y salida [CLS] -> 1 logit.
    """

    def __init__(self, formulation='features', cat_cardinalities=None, n_numeric=0,
                 d_model=32, n_head=4, n_layer=2, dropout=0.1, causal=False, pooling='cls',
                 use_positional=False, numeric_mode='linear', bin_edges=None,
                 pos_weight=None, cls_position='first', cat_encoding='embedding',
                 cat_tables=None, hash_buckets=8, ing_vocab_size=None, max_ingredients=0,
                 ing_layer=1, ing_d_model=None, ing_head=None, n_cyclic=0,
                 text_emb_dim=0, hf_model=''):
        super().__init__()
        assert d_model % n_head == 0, 'd_model debe ser multiplo de n_head'
        assert pooling in ('cls', 'mean')
        assert formulation in ('features', 'ing', 'ing_fusion', 'ing_hybrid')
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

        seq_len = 1  # [CLS]
        self.tokenizer = None
        if formulation in ('features', 'ing_fusion', 'ing_hybrid'):
            self.tokenizer = FeatureTokenizer(
                cat_cardinalities, n_numeric, d_model, numeric_mode, bin_edges,
                cat_encoding, cat_tables, hash_buckets, n_cyclic=n_cyclic
            )
            seq_len += self.tokenizer.n_tokens
        # los INGREDIENTES como conjunto (idea de Fer).
        #   'ing_fusion': el IngredientEncoder resume la lista a UN vector via su
        #       [ING] y ese vector entra como UN token mas de la secuencia tabular
        #       (el mecanismo de fusion, con encoder propio de conjunto).
        #   'ing' / 'ing_hybrid': un token POR ingrediente directo en la secuencia
        #       principal (sin encoder aparte); 'ing' es solo-ingredientes.
        # En ambas: sin positional encoding para la lista (no tiene orden conocido)
        # y el PAD de la lista no recibe atencion.
        self.ing_encoder = None
        self.ing_embedding_table = None
        if formulation == 'ing_fusion':
            assert ing_vocab_size and max_ingredients, \
                'ing_fusion requiere vocabulario de ingredientes (use_ingredients en data)'
            self.ing_encoder = IngredientEncoder(ing_vocab_size, max_ingredients,
                                                 ing_d_model or d_model, ing_head or n_head,
                                                 ing_layer, dropout, d_out=d_model)
            seq_len += 1
        elif formulation in ('ing', 'ing_hybrid'):
            assert ing_vocab_size and max_ingredients, \
                f'{formulation} requiere vocabulario de ingredientes (use_ingredients en data)'
            self.ing_embedding_table = nn.Embedding(ing_vocab_size, d_model,
                                                    padding_idx=PAD_IDX)
            seq_len += max_ingredients
        # transfer learning desde un preentrenado EXTERNO: el titulo del producto
        # entra como UN token = proyeccion aprendida del embedding de un modelo
        # conocido. Dos regimenes de la clase 3: text_emb_dim > 0 = feature
        # extraction (el embedding viene precomputado y congelado en el slot
        # x_text, float); hf_model = fine-tuning (el encoder HF entra al grafo y
        # x_text son sus input_ids).
        self.temb_proj = None
        _hf = None
        if text_emb_dim or hf_model:
            assert formulation == 'features', 'preentrenado de texto: solo formulation features'
            if hf_model:
                from transformers import AutoModel
                _hf = AutoModel.from_pretrained(hf_model)
                text_emb_dim = _hf.config.hidden_size
            self.temb_proj = nn.Linear(text_emb_dim, d_model)
            seq_len += 1
        self.seq_len = seq_len

        # [CLS] aprendido, como BERT: agrega la informacion para clasificar
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        # sin positional encoding por defecto: los features no tienen orden
        # (queda como ablacion, propuesta.md 3)
        self.position_embedding_table = (
            nn.Embedding(seq_len, d_model) if use_positional else None
        )

        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, seq_len, dropout, causal) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, 1)
        # peso para la clase positiva en la BCE (desbalance 87/13), opcional
        self.register_buffer(
            'pos_weight',
            torch.tensor(float(pos_weight)) if pos_weight is not None else None,
        )
        self.apply(self._init_weights)
        # el encoder HF se cuelga DESPUES del apply: _init_weights no debe pisar
        # los pesos preentrenados (seria tirar el transfer learning a la basura)
        self.hf_encoder = _hf

    def _init_weights(self, module):
        # misma inicializacion que la demo
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_sequence(self, x_cat, x_num, x_text):
        """Arma [CLS | features | extras] (o el CLS al final) + su padding mask."""
        batch_size = (x_text if self.formulation == 'ing' else x_cat).shape[0]
        device = (x_text if self.formulation == 'ing' else x_cat).device
        parts, masks = [], []
        if self.tokenizer is not None:
            feat = self.tokenizer(x_cat, x_num)
            parts.append(feat)
            masks.append(torch.ones(batch_size, feat.shape[1], dtype=torch.bool, device=device))
        if self.temb_proj is not None:  # el titulo, embebido por un preentrenado, como un token
            if self.hf_encoder is not None:
                # fine-tuning: x_text son input_ids del tokenizer HF (pad = 0);
                # mean pooling + L2, la receta de sentence-transformers
                mask_t = x_text != 0
                h = self.hf_encoder(input_ids=x_text, attention_mask=mask_t).last_hidden_state
                mt = mask_t.unsqueeze(-1).float()
                vec = (h * mt).sum(1) / mt.sum(1).clamp_min(1e-9)
                vec = nn.functional.normalize(vec, dim=1)
            else:
                vec = x_text  # feature extraction: el embedding ya viene calculado
            parts.append(self.temb_proj(vec).unsqueeze(1))
            masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
        if self.ing_encoder is not None:  # ing_fusion: el resumen del conjunto como un token
            parts.append(self.ing_encoder(x_text).unsqueeze(1))
            masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
        if self.ing_embedding_table is not None:  # ing / ing_hybrid: un token por ingrediente
            parts.append(self.ing_embedding_table(x_text))
            masks.append(x_text != PAD_IDX)  # el padding de la lista no recibe atencion
        cls = [self.cls.expand(batch_size, -1, -1)]
        cls_mask = [torch.ones(batch_size, 1, dtype=torch.bool, device=device)]
        if self.cls_position == 'first':
            parts, masks = cls + parts, cls_mask + masks
        else:  # 'last': con causal, el CLS al final es el unico que ve todo
            parts, masks = parts + cls, masks + cls_mask
        x = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        return x, (mask if self.ing_embedding_table is not None else None)

    def _pooled(self, x_cat=None, x_num=None, x_text=None):
        """Secuencia -> bloques -> ln_f -> vector pooled (la representacion pre-cabeza)."""
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
        return pooled

    def forward(self, x_cat=None, x_num=None, x_text=None, targets=None):
        pooled = self._pooled(x_cat, x_num, x_text)
        logits = self.cls_head(pooled).squeeze(-1)  # (batch,)

        loss = None
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        return logits, loss

    @torch.no_grad()
    def predict_proba(self, x_cat=None, x_num=None, x_text=None, batch_size=4096):
        """p(bought) por fila, en eval mode y por lotes (analogo a generate(), sin autoregresion)."""
        self.eval()
        n = (x_text if self.formulation == 'ing' else x_cat).shape[0]
        sl = lambda t, a, b: None if t is None else t[a:b]
        probs = []
        for start in range(0, n, batch_size):
            end = start + batch_size
            logits, _ = self(sl(x_cat, start, end), sl(x_num, start, end), sl(x_text, start, end))
            probs.append(torch.sigmoid(logits))
        return torch.cat(probs)


ARCHITECTURES = {'transformer': BTRTransformer}

# formulaciones cuyo tercer tensor es la lista de ingredientes (no texto)
ING_FORMULATIONS = ('ing', 'ing_fusion', 'ing_hybrid')


def load_checkpoint(path, device='cpu'):
    """Recarga un checkpoint de pesos/ y devuelve (model en eval mode, preprocessor).

    El preprocessor permite transformar filas nuevas exactamente igual que en el
    entrenamiento (mismos vocabularios y estadisticos de train):

        model, prep = load_checkpoint('pesos/features_d32_h4_l2_linear_seed42.pt')
        x_cat, x_num, x_text, _ = prep.transform(df_nuevo)
        probs = model.predict_proba(x_cat, x_num, x_text)
    """
    import inspect
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get('arch', 'transformer') not in ARCHITECTURES:
        raise ValueError(f"{path}: arquitectura {ckpt['arch']!r} ya no esta en el codigo")
    cls = ARCHITECTURES[ckpt.get('arch', 'transformer')]
    extra = {}
    if ckpt.get('cat_tables') is not None:  # encodings target/freq/hashing/ordinal
        extra['cat_tables'] = ckpt['cat_tables']
    # los checkpoints guardan la config completa de su epoca; las variantes que
    # quedaron fuera del codigo (multi-task, per-feature, ablaciones, transfer
    # externo) se ignoran si estaban apagadas y se rechazan si no
    apagado = {'cart_lambda': 0.0, 'per_feature': 'none', 'sin_residual': False,
               'sin_layernorm': False, 'feature_dropout': 0.0, 'cat_modes': None}
    ignorar = {'char_vocab_size', 'max_text_len'}   # el vocabulario de chars ya no existe
    config = {}
    for k, v in ckpt['model_config'].items():
        if k in ignorar:
            continue
        if k in apagado:
            if v != apagado[k]:
                raise ValueError(f'{path}: usa {k}={v!r}, una variante que ya no esta en el codigo')
        elif k in inspect.signature(cls).parameters:
            config[k] = v
        else:
            raise ValueError(f'{path}: parametro desconocido {k!r} en model_config')
    model = cls(**config, bin_edges=ckpt['bin_edges'], **extra).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['preprocessor']
