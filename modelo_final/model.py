"""El modelo final: FT-Transformer con encoding ordinal de categoricas.

Arquitectura (26.177 parametros):

    13 features --TokenizadorOrdinal--> 13 tokens de d_model=32
    [CLS] + 13 tokens  (14 en total; SIN positional encoding: un conjunto de
                        features no tiene orden — medido: agregarlo da delta ~0)
    -> 2 bloques Transformer pre-LN (atencion multi-cabeza de 4 cabezas de
       dim 8 + FFN 32->128->32, conexiones residuales)
    -> LayerNorm -> tomar el estado final del [CLS] -> Linear(32 -> 1)
    -> sigmoide = p(bought)  ·  loss: binary cross-entropy

Los bloques (Head / MultiHeadAttention / FeedForward / Block) siguen la demo
de la catedra (decoder a nivel caracteres estilo Karpathy) con dos adaptaciones
justificadas:
  1. SIN mascara causal: esto es clasificacion de un conjunto, no generacion
     autoregresiva -> atencion bidireccional (encoder-only, como BERT). Ademas,
     medimos que causal con el [CLS] adelante DEGENERA (el CLS solo se ve a si
     mismo y predice una constante) y que, bien hecho (CLS al final), la
     bidireccionalidad no cambia el resultado.
  2. El escalado de los scores se hace UNA sola vez por sqrt(d_k), como el
     paper (la demo escalaba dos veces).

El tokenizador es el analogo del token_embedding_table de la demo: proyecta
cada feature al mismo espacio R^d donde la atencion puede compararlos.
  - Numericas:   token_j = x_j * w_j + b_j  (vectores aprendidos por feature).
  - Categoricas: el nivel se reemplaza por su RANGO ordinal en [0,1] (tabla
    ajustada en train, ver data.py) y ese escalar entra con la misma proyeccion
    afin que una numerica. Elegido por experimento: supera al embedding
    aprendido clasico (0.824 vs 0.798) porque inyecta como prior el orden
    nivel->propension que el embedding tendria que aprender, con muchos menos
    parametros -> menos overfitting en un dataset de 10k filas.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from data import CAT_FEATURES, NUM_FEATURES


class Head(nn.Module):
    """Una cabeza de self-attention (Q, K, V propios; softmax(QK^T / sqrt(d_k)) V)."""

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k = self.key(x)                                       # (batch, seq, head_size)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (batch, seq, seq)
        wei = F.softmax(wei, dim=-1)
        if getattr(self, 'guardar_atencion', False):          # para analisis/interpretabilidad
            self.ultima_atencion = wei.detach()
        wei = self.dropout(wei)
        return wei @ self.value(x)                            # (batch, seq, head_size)


class MultiHeadAttention(nn.Module):
    """Varias cabezas en paralelo + proyeccion de salida."""

    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """MLP interno del bloque: expansion x4 + ReLU."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Bloque Transformer pre-LN con conexiones residuales."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TokenizadorOrdinal(nn.Module):
    """13 features -> 13 tokens de d_model (categoricas ordinales + numericas afines)."""

    def __init__(self, cat_tables, n_numeric, d_model):
        super().__init__()
        self.n_cat = len(cat_tables)
        for i, t in enumerate(cat_tables):
            self.register_buffer(f'cat_table_{i}', t.float())
        self.cat_scalar_weight = nn.ParameterDict()
        self.cat_scalar_bias = nn.ParameterDict()
        for i in range(self.n_cat):
            w = nn.Parameter(torch.empty(d_model))
            nn.init.normal_(w, mean=0.0, std=0.02)
            self.cat_scalar_weight[str(i)] = w
            self.cat_scalar_bias[str(i)] = nn.Parameter(torch.zeros(d_model))
        self.num_weight = nn.Parameter(torch.empty(n_numeric, d_model))
        self.num_bias = nn.Parameter(torch.zeros(n_numeric, d_model))
        nn.init.normal_(self.num_weight, mean=0.0, std=0.02)

    @property
    def n_tokens(self):
        return self.n_cat + self.num_weight.shape[0]

    def forward(self, x_cat, x_num):
        tokens = []
        for i in range(self.n_cat):  # rango ordinal en [0,1] -> proyeccion afin a d_model
            v = getattr(self, f'cat_table_{i}')[x_cat[:, i]].unsqueeze(-1)
            tokens.append(v * self.cat_scalar_weight[str(i)] + self.cat_scalar_bias[str(i)])
        num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        return torch.cat([torch.stack(tokens, dim=1), num_tokens], dim=1)  # (batch, 13, d)


class ModeloBTR(nn.Module):
    """[CLS] + feature-tokens -> bloques Transformer -> p(bought) por impresion."""

    def __init__(self, cat_tables, n_numeric=len(NUM_FEATURES), d_model=32, n_head=4,
                 n_layer=2, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.tokenizer = TokenizadorOrdinal(cat_tables, n_numeric, d_model)
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))   # token de lectura, como BERT
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x_cat, x_num, targets=None):
        tokens = self.tokenizer(x_cat, x_num)                        # (batch, 13, d)
        x = torch.cat([self.cls.expand(x_cat.shape[0], -1, -1), tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        logits = self.cls_head(self.ln_f(x)[:, 0]).squeeze(-1)       # estado final del CLS
        loss = None
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        return logits, loss

    @torch.no_grad()
    def predict_proba(self, x_cat, x_num, batch_size=4096):
        self.eval()
        probs = []
        for s in range(0, x_cat.shape[0], batch_size):
            logits, _ = self(x_cat[s:s + batch_size], x_num[s:s + batch_size])
            probs.append(torch.sigmoid(logits))
        return torch.cat(probs)


def guardar(path, model, prep, seed, metricas):
    """Checkpoint en formato PLANO (sin clases pickladas): portable y auditable."""
    torch.save({
        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'config': {'d_model': 32, 'n_head': 4, 'n_layer': 2, 'dropout': 0.1},
        'vocabs': prep.vocabs,
        'num_mean': prep.num_mean.tolist(),
        'num_std': prep.num_std.tolist(),
        'cat_tables': [t.tolist() for t in prep.cat_tables],
        'seed': seed,
        'metricas': metricas,
    }, path)


def cargar(path, device='cpu'):
    """Recarga (modelo en eval mode, preprocessor) desde un checkpoint plano."""
    import numpy as np
    from data import Preprocessor
    ckpt = torch.load(path, map_location=device, weights_only=False)
    prep = Preprocessor(vocabs=ckpt['vocabs'],
                        num_mean=np.array(ckpt['num_mean']),
                        num_std=np.array(ckpt['num_std']),
                        cat_tables=[torch.tensor(t) for t in ckpt['cat_tables']])
    model = ModeloBTR(prep.cat_tables, **ckpt['config']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, prep
