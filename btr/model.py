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
    """

    def __init__(self, cat_cardinalities, n_numeric, d_model,
                 numeric_mode='linear', bin_edges=None):
        super().__init__()
        self.numeric_mode = numeric_mode
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in cat_cardinalities]
        )
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
        return len(self.cat_embeddings) + n_num

    def forward(self, x_cat, x_num):
        # x_cat: (batch, n_cat) long; x_num: (batch, n_num) float
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
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
                 pos_weight=None):
        super().__init__()
        assert d_model % n_head == 0, 'd_model debe ser multiplo de n_head'
        assert pooling in ('cls', 'mean')
        assert formulation in ('features', 'text', 'hybrid')
        self.formulation = formulation
        self.pooling = pooling

        seq_len = 1  # [CLS]
        self.tokenizer = None
        if formulation in ('features', 'hybrid'):
            self.tokenizer = FeatureTokenizer(
                cat_cardinalities, n_numeric, d_model, numeric_mode, bin_edges
            )
            seq_len += self.tokenizer.n_tokens
        self.char_embedding_table = None
        if formulation in ('text', 'hybrid'):
            assert char_vocab_size and max_text_len, 'text/hybrid requieren vocabulario de chars'
            self.char_embedding_table = nn.Embedding(char_vocab_size, d_model, padding_idx=PAD_IDX)
            seq_len += max_text_len
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

    def _init_weights(self, module):
        # misma inicializacion que la demo
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_sequence(self, x_cat, x_num, x_text):
        """Arma [CLS | features | chars] segun la formulacion + su padding mask."""
        batch_size = (x_text if self.formulation == 'text' else x_cat).shape[0]
        device = (x_text if self.formulation == 'text' else x_cat).device
        parts = [self.cls.expand(batch_size, -1, -1)]
        masks = [torch.ones(batch_size, 1, dtype=torch.bool, device=device)]
        if self.tokenizer is not None:
            feat = self.tokenizer(x_cat, x_num)
            parts.append(feat)
            masks.append(torch.ones(batch_size, feat.shape[1], dtype=torch.bool, device=device))
        if self.char_embedding_table is not None:
            parts.append(self.char_embedding_table(x_text))
            masks.append(x_text != PAD_IDX)  # el padding del texto no recibe atencion
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
            pooled = x[:, 0]
        else:  # promedio solo sobre posiciones reales
            m = (padding_mask if padding_mask is not None
                 else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device))
            pooled = (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
        logits = self.cls_head(pooled).squeeze(-1)  # (batch,)

        loss = None
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight
            )
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
                 numeric_mode='linear', bin_edges=None, pos_weight=None):
        super().__init__()
        self.tokenizer = FeatureTokenizer(cat_cardinalities, n_numeric, d_model,
                                          numeric_mode, bin_edges)
        in_dim = self.tokenizer.n_tokens * d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, 8 * d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(8 * d_model, 2 * d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, 1),
        )
        self.register_buffer(
            'pos_weight',
            torch.tensor(float(pos_weight)) if pos_weight is not None else None,
        )

    def forward(self, x_cat, x_num, x_text=None, targets=None):
        tokens = self.tokenizer(x_cat, x_num)              # (batch, n_tokens, d_model)
        logits = self.net(tokens.flatten(1)).squeeze(-1)   # (batch,)
        loss = None
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets,
                                                      pos_weight=self.pos_weight)
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
                 numeric_mode='linear', bin_edges=None, pos_weight=None):
        super().__init__()
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
                                          numeric_mode, bin_edges)
        head_in = d_model + self.tokenizer.n_tokens * d_model
        self.head = nn.Sequential(
            nn.Linear(head_in, 4 * d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, 1),
        )
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
        logits = self.head(torch.cat([text_emb, tab], dim=1)).squeeze(-1)
        loss = None
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets,
                                                      pos_weight=self.pos_weight)
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
    """

    def __init__(self, cat_cardinalities, n_numeric, max_products, d_model=32,
                 n_head=4, n_layer=2, dropout=0.1, numeric_mode='linear',
                 bin_edges=None, pos_weight=None):
        super().__init__()
        self.tokenizer = FeatureTokenizer(cat_cardinalities, n_numeric, d_model,
                                          numeric_mode, bin_edges)
        self.product_proj = nn.Linear(self.tokenizer.n_tokens * d_model, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, max_products, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.logit_head = nn.Linear(d_model, 1)
        self.register_buffer(
            'pos_weight',
            torch.tensor(float(pos_weight)) if pos_weight is not None else None,
        )

    def forward(self, x_cat, x_num, prod_mask, targets=None):
        # x_cat: (batch, P, n_cat); x_num: (batch, P, n_num); prod_mask: (batch, P)
        batch_size, n_products, _ = x_cat.shape
        tokens = self.tokenizer(x_cat.flatten(0, 1), x_num.flatten(0, 1))  # (B*P, T, d)
        products = self.product_proj(tokens.flatten(1))                    # (B*P, d)
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
    model = cls(**ckpt['model_config'], bin_edges=ckpt['bin_edges']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['preprocessor']
