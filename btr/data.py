"""Carga y preprocesamiento del dataset de eventos de busqueda.

Implementa las decisiones de propuesta.md (secciones 2 y 6):
- `listing_status` se extrae del sufijo entre parentesis del titulo (la senal
  dominante del dataset, ver EDA) y `price_rel` es la posicion del precio dentro
  del rango filtrado (efecto no lineal en U invertida).
- `cart` NO es feature (leakage: bought => cart). `query_id` solo se usa para
  particionar. timestamp / package_size / dimensions / filtros duplicados quedan
  afuera (redundantes o sin senal, ver EDA en propuesta.md).
- Split por query_id (group split): filas de la misma busqueda no se reparten
  entre train/val/test.
- Vocabularios y estadisticos (medias/desvios/cuantiles) se calculan SOLO con
  train; el indice 0 de cada vocabulario es UNK para niveles no vistos.
- Para las formulaciones de texto (ver propuesta.md 4C) se codifica
  title + '\n' + description a nivel caracteres, igual que la demo de la
  catedra: vocabulario de chars de train, PAD=0, UNK=1, truncado a max_text_len
  (el titulo entra entero: el sufijo de estado esta al final y mide <= 81 chars).
- Para las formulaciones de INGREDIENTES (9na tanda) la lista se codifica como
  CONJUNTO: vocabulario de ingredientes de train (PAD=0, UNK=1 para no vistos),
  un indice por item, padded a MAX_INGREDIENTS; viaja en el slot del texto.
"""

import re as _re_mod
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
LOG_FEATURES = ['price', 'net_weight_oz']  # sesgadas -> log1p antes del z-score
TARGET = 'bought'
PAD_IDX, UNK_IDX = 0, 1  # reservados del vocabulario de caracteres
MAX_TEXT_LEN = 256       # p95 de title+description = 243 chars
MAX_INGREDIENTS = 5      # largo maximo de la lista de ingredientes (maximo global del CSV)
STATUS_SUFFIX_RE = r'\s*\([^)]+\)$'  # "(Best Seller)" etc. al final del titulo


def _palabras(texto):
    """Tokenizacion word-level: minusculas, secuencias alfanumericas."""
    return _re_mod.findall(r'[a-z0-9]+', texto.lower())


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
    return df


def split_by_query(df, seed=42, val_frac=0.15, test_frac=0.15, cv_k=0, cv_fold=0):
    """Particiona por query_id para que una misma busqueda no cruce splits.

    Modo por defecto: holdout 70/15/15. Con cv_k > 0: GroupKFold manual — las
    queries (barajadas con seed) se parten en cv_k folds; test = el fold cv_fold
    y la validacion (para el early stopping) se recorta del resto. Corriendo los
    cv_k folds con la misma seed, cada query pasa por test exactamente una vez.
    """
    rng = np.random.default_rng(seed)
    # a numpy: shufflear el ExtensionArray de pandas no esta garantizado (warning de pandas 3)
    queries = df['query_id'].unique().to_numpy()
    rng.shuffle(queries)
    if cv_k:
        assert 0 <= cv_fold < cv_k, 'cv_fold fuera de rango'
        folds = np.array_split(queries, cv_k)
        test_q = set(folds[cv_fold])
        resto = np.concatenate([f for i, f in enumerate(folds) if i != cv_fold])
        val_q = set(resto[:int(len(resto) * val_frac)])
    else:
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
    text_tokens: str = 'chars'  # 'chars' (la demo) | 'words' (tokenización word-level)
    cat_features: list = None   # None -> CAT_FEATURES (instancias viejas)
    num_features: list = None   # None -> NUM_FEATURES
    ing_vocab: dict = None      # ingrediente -> indice (0=PAD, 1=UNK); formulaciones ing_*
    use_ingredients: bool = False  # True -> el 3er tensor es la lista de ingredientes, no texto

    @property
    def cats(self):
        return getattr(self, 'cat_features', None) or CAT_FEATURES

    @property
    def nums(self):
        return getattr(self, 'num_features', None) or NUM_FEATURES

    @classmethod
    def fit(cls, train_df, max_text_len=MAX_TEXT_LEN, strip_status=False,
            text_tokens='chars', use_ingredients=False):
        cats, nums = list(CAT_FEATURES), list(NUM_FEATURES)
        vocabs = {
            f: {v: i + 1 for i, v in enumerate(sorted(train_df[f].unique()))}
            for f in cats
        }
        num = cls._numeric_raw_static(train_df, nums)
        if text_tokens == 'words':
            # vocabulario de PALABRAS de train (5b de la revision externa): minusculas,
            # solo alfanumericos — "(Best Seller)" sobrevive como 'best','seller'
            palabras = sorted({w for t, d in zip(train_df['title'], train_df['description'])
                               for w in _palabras(t + ' ' + d)})
            char_vocab = {w: i + 2 for i, w in enumerate(palabras)}  # 0=PAD, 1=UNK
        else:
            chars = sorted(set('\n'.join(train_df['title']) + '\n'.join(train_df['description'])))
            char_vocab = {c: i + 2 for i, c in enumerate(chars)}  # 0=PAD, 1=UNK
        ing_vocab = None
        if use_ingredients:
            # vocabulario de INGREDIENTES de train (mismo contrato que el de chars:
            # nada de test entra al vocabulario; los no vistos caen en UNK)
            items = sorted({t.strip() for lst in train_df['ingredients'].str.split(',')
                            for t in lst})
            ing_vocab = {v: i + 2 for i, v in enumerate(items)}  # 0=PAD, 1=UNK
        return cls(vocabs=vocabs, num_mean=num.mean(axis=0), num_std=num.std(axis=0) + 1e-8,
                   char_vocab=char_vocab, max_text_len=max_text_len, strip_status=strip_status,
                   cat_features=cats, num_features=nums,
                   text_tokens=text_tokens, ing_vocab=ing_vocab, use_ingredients=use_ingredients)

    @property
    def char_vocab_size(self):
        return len(self.char_vocab) + 2  # + PAD + UNK

    @property
    def ing_vocab_size(self):
        return len(self.ing_vocab) + 2  # + PAD + UNK

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
        """df -> (x_cat long, x_num float32, x_text long, y float32) listos para el modelo."""
        x_cat = np.stack(
            [df[f].map(lambda v, f=f: self.vocabs[f].get(v, 0)).to_numpy() for f in self.cats],
            axis=1,
        )
        x_num = (self._numeric_raw(df) - self.num_mean) / self.num_std
        if getattr(self, 'use_ingredients', False):
            # las formulaciones ing_* viajan en el slot del texto (como --text-emb)
            x_text = np.stack([self._encode_ingredients(s) for s in df['ingredients']])
        else:
            x_text = np.stack([
                self._encode_text(t, d) for t, d in zip(df['title'], df['description'])
            ])
        y = df[TARGET].to_numpy()
        return (
            torch.tensor(x_cat, dtype=torch.long),
            torch.tensor(x_num, dtype=torch.float32),
            torch.tensor(x_text, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )

    def _encode_text(self, title, description):
        """title + description -> indices (de chars o de palabras), truncado, PAD al final."""
        if self.strip_status:
            title, description = strip_status_from_text(title, description)
        if getattr(self, 'text_tokens', 'chars') == 'words':
            toks = _palabras(title + ' ' + description)[:self.max_text_len]
        else:
            toks = (title + '\n' + description)[:self.max_text_len]
        ids = [self.char_vocab.get(t, UNK_IDX) for t in toks]
        return np.array(ids + [PAD_IDX] * (self.max_text_len - len(ids)), dtype=np.int64)

    def _encode_ingredients(self, ingredients):
        """'Wheat flour, Yeast, Salt' -> indices del vocabulario de train, PAD al final.

        La lista no tiene orden conocido (el orden escrito no significa nada), asi
        que quien consuma esto NO debe usar positional encoding (IngredientEncoder
        en btr/model.py trata la lista como conjunto).
        """
        ids = [self.ing_vocab.get(t.strip(), UNK_IDX)
               for t in ingredients.split(',')][:MAX_INGREDIENTS]
        return np.array(ids + [PAD_IDX] * (MAX_INGREDIENTS - len(ids)), dtype=np.int64)

    def bin_edges(self, train_df, n_bins=16):
        """Bordes por cuantiles de train para numeric_mode='bins' (en escala ya normalizada)."""
        x_num = (self._numeric_raw(train_df) - self.num_mean) / self.num_std
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.quantile(x_num, qs, axis=0).T  # (n_num, n_bins-1)
        return torch.tensor(edges, dtype=torch.float32)


def prepare(csv_path, seed=42, max_text_len=MAX_TEXT_LEN, strip_status=False,
            text_tokens='chars', train_frac=1.0, cv_k=0, cv_fold=0, use_ingredients=False):
    """Pipeline completo: carga -> split por query -> fit en train -> tensores.

    train_frac < 1 submuestrea las QUERIES de train (curva de aprendizaje; val y
    test quedan intactos para que las curvas sean comparables). El Preprocessor
    se ajusta sobre el train reducido, como corresponde.
    """
    df = load_dataset(csv_path)
    train_df, val_df, test_df = split_by_query(df, seed=seed, cv_k=cv_k, cv_fold=cv_fold)
    if train_frac < 1.0:
        qs = train_df['query_id'].unique().to_numpy()
        np.random.default_rng(seed + 1000).shuffle(qs)
        train_df = train_df[train_df['query_id'].isin(set(qs[:int(len(qs) * train_frac)]))]
    prep = Preprocessor.fit(train_df, max_text_len=max_text_len, strip_status=strip_status,
                            text_tokens=text_tokens, use_ingredients=use_ingredients)
    return prep, train_df, {
        'train': prep.transform(train_df),
        'val': prep.transform(val_df),
        'test': prep.transform(test_df),
    }

