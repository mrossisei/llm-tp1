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
- El tercer tensor de cada split ("slot de texto") lleva, segun la corrida: la
  lista de INGREDIENTES como conjunto (vocabulario de ingredientes de train,
  PAD=0, UNK=1 para no vistos, un indice por item, padded a MAX_INGREDIENTS), el
  embedding del titulo de un modelo preentrenado (transfer learning, lo arma
  btr.train) o nada (N, 0).
- titulo_sin_estado(): el titulo sin el sufijo "( ... )" de estado, lo unico que
  ve el modelo preentrenado (que no vuelva a leer el badge que ya extrae el regex).
"""

import re
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
PAD_IDX, UNK_IDX = 0, 1  # reservados del vocabulario de ingredientes
MAX_INGREDIENTS = 5      # largo maximo de la lista de ingredientes (maximo global del CSV)
STATUS_SUFFIX_RE = r'\s*\([^)]+\)$'  # "(Best Seller)" etc. al final del titulo


def titulo_sin_estado(title):
    """'Greek Yogurt - 24 oz (Top Rated)' -> 'Greek Yogurt - 24 oz': el titulo sin el badge."""
    return re.sub(STATUS_SUFFIX_RE, '', title).strip()


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
    char_vocab: dict = None     # (vestigial: checkpoints viejos con formulaciones de texto)
    max_text_len: int = 0       # (vestigial)
    strip_status: bool = False  # (vestigial)
    text_tokens: str = 'chars'  # (vestigial)
    cat_features: list = None   # None -> CAT_FEATURES (instancias viejas)
    num_features: list = None   # None -> NUM_FEATURES
    ing_vocab: dict = None      # ingrediente -> indice (0=PAD, 1=UNK); formulaciones ing_*
    use_ingredients: bool = False  # True -> el 3er tensor es la lista de ingredientes

    @property
    def cats(self):
        return getattr(self, 'cat_features', None) or CAT_FEATURES

    @property
    def nums(self):
        return getattr(self, 'num_features', None) or NUM_FEATURES

    @classmethod
    def fit(cls, train_df, use_ingredients=False):
        cats, nums = list(CAT_FEATURES), list(NUM_FEATURES)
        vocabs = {
            f: {v: i + 1 for i, v in enumerate(sorted(train_df[f].unique()))}
            for f in cats
        }
        num = cls._numeric_raw_static(train_df, nums)
        ing_vocab = None
        if use_ingredients:
            # vocabulario de INGREDIENTES de train (mismo contrato que el de chars:
            # nada de test entra al vocabulario; los no vistos caen en UNK)
            items = sorted({t.strip() for lst in train_df['ingredients'].str.split(',')
                            for t in lst})
            ing_vocab = {v: i + 2 for i, v in enumerate(items)}  # 0=PAD, 1=UNK
        return cls(vocabs=vocabs, num_mean=num.mean(axis=0), num_std=num.std(axis=0) + 1e-8,
                   cat_features=cats, num_features=nums,
                   ing_vocab=ing_vocab, use_ingredients=use_ingredients)

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
        """df -> (x_cat long, x_num float32, x_text long, y float32) listos para el modelo.

        x_text es la lista de ingredientes (use_ingredients) o un tensor vacio (N, 0):
        el slot lo puede ocupar despues el embedding del titulo (btr.train, --text-emb).
        """
        x_cat = np.stack(
            [df[f].map(lambda v, f=f: self.vocabs[f].get(v, 0)).to_numpy() for f in self.cats],
            axis=1,
        )
        x_num = (self._numeric_raw(df) - self.num_mean) / self.num_std
        if getattr(self, 'use_ingredients', False):
            x_text = np.stack([self._encode_ingredients(s) for s in df['ingredients']])
        else:
            x_text = np.zeros((len(df), 0), dtype=np.int64)
        y = df[TARGET].to_numpy()
        return (
            torch.tensor(x_cat, dtype=torch.long),
            torch.tensor(x_num, dtype=torch.float32),
            torch.tensor(x_text, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )

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


def prepare(csv_path, seed=42, train_frac=1.0, cv_k=0, cv_fold=0, use_ingredients=False,
            with_index=False):
    """Pipeline completo: carga -> split por query -> fit en train -> tensores.

    train_frac < 1 submuestrea las QUERIES de train (curva de aprendizaje; val y
    test quedan intactos para que las curvas sean comparables). El Preprocessor
    se ajusta sobre el train reducido, como corresponde. Con with_index devuelve
    ademas {split: indices de fila del CSV}, para alinear tablas externas (los
    embeddings precomputados del titulo) con los tensores de cada split.
    """
    df = load_dataset(csv_path)
    train_df, val_df, test_df = split_by_query(df, seed=seed, cv_k=cv_k, cv_fold=cv_fold)
    if train_frac < 1.0:
        qs = train_df['query_id'].unique().to_numpy()
        np.random.default_rng(seed + 1000).shuffle(qs)
        train_df = train_df[train_df['query_id'].isin(set(qs[:int(len(qs) * train_frac)]))]
    prep = Preprocessor.fit(train_df, use_ingredients=use_ingredients)
    dfs = {'train': train_df, 'val': val_df, 'test': test_df}
    splits = {k: prep.transform(d) for k, d in dfs.items()}
    if with_index:
        return prep, train_df, splits, {k: d.index.to_numpy() for k, d in dfs.items()}
    return prep, train_df, splits

