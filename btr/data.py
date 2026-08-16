"""Carga y preprocesamiento del dataset de eventos de busqueda.

Implementa las decisiones de propuesta.md (secciones 2 y 6):
- `listing_status` se extrae del sufijo entre parentesis del titulo (la senal
  dominante del dataset, ver EDA) y `price_rel` es la posicion del precio dentro
  del rango filtrado (efecto no lineal en U invertida).
- `cart` NO es feature (leakage: bought => cart). `query_id` solo se usa para
  particionar. timestamp / package_size / dimensions / filtros duplicados quedan
  afuera de la v1 (redundantes o sin senal; reintroducibles via ablacion).
- Split por query_id (group split): filas de la misma busqueda no se reparten
  entre train/val/test.
- Vocabularios y estadisticos (medias/desvios/cuantiles) se calculan SOLO con
  train; el indice 0 de cada vocabulario es UNK para niveles no vistos.
- Para las formulaciones de texto (ver propuesta.md 4C) se codifica
  title + '\n' + description a nivel caracteres, igual que la demo de la
  catedra: vocabulario de chars de train, PAD=0, UNK=1, truncado a max_text_len
  (el titulo entra entero: el sufijo de estado esta al final y mide <= 81 chars).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

CAT_FEATURES = [
    'listing_status', 'category', 'brand', 'storage_type',
    'unit_of_measure', 'country_of_origin', 'allergens',
]
NUM_FEATURES = [
    'price', 'price_rel', 'filter_price_min', 'filter_price_max',
    'net_weight_oz', 'nutrition_score',
]
# columnas descartadas en la v1 por redundancia (propuesta 2.6), reintroducibles
# via --extra-features para verificar empiricamente que no aportan:
EXTRA_FEATURES = {          # nombre -> tipo ('num'|'cat'); derivadas en load_dataset
    'volume_in3': 'num',    # producto de dimensions_in (L x W x H)
    'pkg_qty': 'num',       # numero de package_size ("24 oz" -> 24)
    'pkg_unit': 'cat',      # unidad de package_size (oz / ct / ...)
    'n_ingredients': 'num', # cantidad de items en ingredients
}
LOG_FEATURES = ['price', 'net_weight_oz', 'volume_in3', 'pkg_qty']  # sesgadas -> log1p antes del z-score
TARGET = 'bought'
TARGET_AUX = 'cart'  # target auxiliar para multi-task (--cart-aux); NUNCA es input
PAD_IDX, UNK_IDX = 0, 1  # reservados del vocabulario de caracteres
MAX_TEXT_LEN = 256       # p95 de title+description = 243 chars
STATUS_SUFFIX_RE = r'\s*\([^)]+\)$'  # "(Best Seller)" etc. al final del titulo


def strip_status_from_text(title, description):
    """Versión 'intrínseca' del texto: saca el sufijo de estado del título y la
    última oración de la descripción (que repite el mismo estado en prosa).
    Necesario para el experimento 'producto nuevo': sin esto, la variante de
    texto re-aprende la popularidad desde la descripción aunque saquemos el
    sufijo (ver propuesta.md 2.3.1)."""
    import re as _re
    title = _re.sub(STATUS_SUFFIX_RE, '', title)
    sentences = description.rstrip('.').split('. ')
    if len(sentences) > 1:
        description = '. '.join(sentences[:-1]) + '.'
    return title, description


def load_dataset(csv_path):
    """Lee el CSV y deriva los features nuevos (sin mirar el target)."""
    df = pd.read_csv(csv_path)
    df['listing_status'] = df['title'].str.extract(r'\(([^)]+)\)$')[0].fillna('None')
    df['allergens'] = df['allergens'].fillna('None')  # "sin alergenos declarados"
    span = df['filter_price_max'] - df['filter_price_min']
    df['price_rel'] = (df['price'] - df['filter_price_min']) / span
    # derivados de las columnas descartadas (solo entran con --extra-features):
    dims = df['dimensions_in'].str.extract(r'([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)')
    df['volume_in3'] = dims.astype(float).prod(axis=1)
    pkg = df['package_size'].str.extract(r'([\d.]+)\s*(\w+)')
    df['pkg_qty'] = pkg[0].astype(float)
    df['pkg_unit'] = pkg[1].fillna('None')
    df['n_ingredients'] = df['ingredients'].str.count(',') + 1
    return df


def split_by_query(df, seed=42, val_frac=0.15, test_frac=0.15):
    """Particiona por query_id para que una misma busqueda no cruce splits."""
    rng = np.random.default_rng(seed)
    # a numpy: shufflear el ExtensionArray de pandas no esta garantizado (warning de pandas 3)
    queries = df['query_id'].unique().to_numpy()
    rng.shuffle(queries)
    n_test = int(len(queries) * test_frac)
    n_val = int(len(queries) * val_frac)
    test_q = set(queries[:n_test])
    val_q = set(queries[n_test:n_test + n_val])
    assert not (test_q & val_q), 'splits no disjuntos: bug en el particionado'
    is_test = df['query_id'].isin(test_q)
    is_val = df['query_id'].isin(val_q)
    return df[~is_test & ~is_val], df[is_val], df[is_test]


@dataclass
class Preprocessor:
    """Vocabularios + estadisticos aprendidos de train, aplicables a cualquier split.

    Los campos nuevos usan getattr con default en los metodos: los checkpoints de
    pesos/ guardan instancias pickladas viejas que no los tienen.
    """
    vocabs: dict          # feature -> {valor: indice}, 0 reservado a UNK
    num_mean: np.ndarray  # (n_num,)
    num_std: np.ndarray   # (n_num,)
    char_vocab: dict      # caracter -> indice (PAD=0, UNK=1)
    max_text_len: int
    strip_status: bool = False  # True -> texto sin sufijo/oración de estado (modelo intrínseco)
    cat_features: list = None   # None -> CAT_FEATURES (instancias viejas)
    num_features: list = None   # None -> NUM_FEATURES
    include_cart: bool = False  # True -> y de shape (N, 2): [bought, cart] (--cart-aux)

    @property
    def cats(self):
        return getattr(self, 'cat_features', None) or CAT_FEATURES

    @property
    def nums(self):
        return getattr(self, 'num_features', None) or NUM_FEATURES

    @classmethod
    def fit(cls, train_df, max_text_len=MAX_TEXT_LEN, strip_status=False,
            extra_features=(), include_cart=False):
        desconocidos = set(extra_features) - set(EXTRA_FEATURES)
        if desconocidos:
            raise ValueError(f'extra_features desconocidos: {sorted(desconocidos)} '
                             f'(validos: {sorted(EXTRA_FEATURES)})')
        cats = CAT_FEATURES + [f for f in extra_features if EXTRA_FEATURES[f] == 'cat']
        nums = NUM_FEATURES + [f for f in extra_features if EXTRA_FEATURES[f] == 'num']
        vocabs = {
            f: {v: i + 1 for i, v in enumerate(sorted(train_df[f].unique()))}
            for f in cats
        }
        num = cls._numeric_raw_static(train_df, nums)
        chars = sorted(set('\n'.join(train_df['title']) + '\n'.join(train_df['description'])))
        char_vocab = {c: i + 2 for i, c in enumerate(chars)}  # 0=PAD, 1=UNK
        return cls(vocabs=vocabs, num_mean=num.mean(axis=0), num_std=num.std(axis=0) + 1e-8,
                   char_vocab=char_vocab, max_text_len=max_text_len, strip_status=strip_status,
                   cat_features=cats, num_features=nums, include_cart=include_cart)

    @property
    def char_vocab_size(self):
        return len(self.char_vocab) + 2  # + PAD + UNK

    @staticmethod
    def _numeric_raw_static(df, num_features):
        num = df[num_features].to_numpy(dtype=np.float64).copy()
        for f in LOG_FEATURES:
            if f in num_features:
                j = num_features.index(f)
                num[:, j] = np.log1p(num[:, j])
        return num

    def _numeric_raw(self, df):
        return self._numeric_raw_static(df, self.nums)

    @property
    def cat_cardinalities(self):
        return [len(self.vocabs[f]) + 1 for f in self.cats]  # +1 por UNK

    def transform(self, df):
        """df -> (x_cat long, x_num float32, x_text long, y float32) listos para el modelo.

        Con include_cart, y es (N, 2): columna 0 = bought (target real), columna 1 =
        cart (solo etiqueta auxiliar de entrenamiento, jamas input).
        """
        x_cat = np.stack(
            [df[f].map(lambda v, f=f: self.vocabs[f].get(v, 0)).to_numpy() for f in self.cats],
            axis=1,
        )
        x_num = (self._numeric_raw(df) - self.num_mean) / self.num_std
        x_text = np.stack([
            self._encode_text(t, d) for t, d in zip(df['title'], df['description'])
        ])
        if getattr(self, 'include_cart', False):
            y = df[[TARGET, TARGET_AUX]].to_numpy()
        else:
            y = df[TARGET].to_numpy()
        return (
            torch.tensor(x_cat, dtype=torch.long),
            torch.tensor(x_num, dtype=torch.float32),
            torch.tensor(x_text, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )

    def _encode_text(self, title, description):
        """title + '\n' + description -> indices de caracteres, truncado y con PAD al final."""
        if self.strip_status:
            title, description = strip_status_from_text(title, description)
        text = (title + '\n' + description)[:self.max_text_len]
        ids = [self.char_vocab.get(c, UNK_IDX) for c in text]
        return np.array(ids + [PAD_IDX] * (self.max_text_len - len(ids)), dtype=np.int64)

    def bin_edges(self, train_df, n_bins=16):
        """Bordes por cuantiles de train para numeric_mode='bins' (en escala ya normalizada)."""
        x_num = (self._numeric_raw(train_df) - self.num_mean) / self.num_std
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.quantile(x_num, qs, axis=0).T  # (n_num, n_bins-1)
        return torch.tensor(edges, dtype=torch.float32)


def prepare(csv_path, seed=42, max_text_len=MAX_TEXT_LEN, strip_status=False,
            extra_features=(), include_cart=False):
    """Pipeline completo: carga -> split por query -> fit en train -> tensores."""
    df = load_dataset(csv_path)
    train_df, val_df, test_df = split_by_query(df, seed=seed)
    prep = Preprocessor.fit(train_df, max_text_len=max_text_len, strip_status=strip_status,
                            extra_features=extra_features, include_cart=include_cart)
    return prep, train_df, {
        'train': prep.transform(train_df),
        'val': prep.transform(val_df),
        'test': prep.transform(test_df),
    }


def make_query_tensors(df, prep, max_products=None, with_text=False):
    """Agrupa las filas por query para la formulación listwise (propuesta.md 4B).

    Devuelve tensores (Q, P, ...) con P = máximo de productos por página (8 en el
    dataset), padded con ceros y una máscara (Q, P) que marca los productos reales
    (el padding no recibe atención ni entra en la loss). El orden de los productos
    es el del CSV: no hay columna de posición/rank, así que no se asume orden.

    Con with_text (--listwise-texto) el tercer tensor es x_text (Q, P, L) en lugar
    de la máscara: los slots de padding quedan todo-PAD, así que la máscara se
    reconstruye como (x_text != PAD).any(-1) — todo título real tiene caracteres.
    """
    x_cat, x_num, x_text, y = prep.transform(df)
    sizes = df.groupby('query_id', sort=False).size()
    if max_products is None:
        max_products = int(sizes.max())
    n_queries = len(sizes)

    out_cat = torch.zeros(n_queries, max_products, x_cat.shape[1], dtype=torch.long)
    out_num = torch.zeros(n_queries, max_products, x_num.shape[1], dtype=torch.float32)
    out_y = torch.zeros(n_queries, max_products, dtype=torch.float32)
    out_mask = torch.zeros(n_queries, max_products, dtype=torch.bool)
    out_text = torch.zeros(n_queries, max_products, x_text.shape[1], dtype=torch.long)

    start = 0
    for q, size in enumerate(sizes):
        size = min(int(size), max_products)
        rows = slice(start, start + size)
        out_cat[q, :size] = x_cat[rows]
        out_num[q, :size] = x_num[rows]
        out_y[q, :size] = y[rows]
        out_mask[q, :size] = True
        out_text[q, :size] = x_text[rows]
        start += int(sizes.iloc[q])
    return out_cat, out_num, (out_text if with_text else out_mask), out_y


def prepare_listwise(csv_path, seed=42, with_text=False, max_text_len=MAX_TEXT_LEN,
                     strip_status=False):
    """Como prepare(), pero con tensores agrupados por query (para --arch listwise)."""
    df = load_dataset(csv_path)
    train_df, val_df, test_df = split_by_query(df, seed=seed)
    prep = Preprocessor.fit(train_df, max_text_len=max_text_len, strip_status=strip_status)
    max_products = int(df.groupby('query_id').size().max())
    return prep, max_products, {
        'train': make_query_tensors(train_df, prep, max_products, with_text),
        'val': make_query_tensors(val_df, prep, max_products, with_text),
        'test': make_query_tensors(test_df, prep, max_products, with_text),
    }
