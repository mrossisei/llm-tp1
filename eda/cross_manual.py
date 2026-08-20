"""¿Cuanto del aporte de la atencion es UNA sola interaccion hecha a mano?

El EDA dice que la U invertida del precio esta condicionada al tier de estado.
Este script le da a una REGRESION LOGISTICA esa interaccion explicitamente
(one-hot(listing_status) x [price_rel, price_rel**2]) y mide cuanto del gap
lineal -> transformer se cierra. Si la logistica con el cross se acerca al
transformer, "lo que la atencion aprendio sola" es (en buena parte) ese cruce.

    .venv/bin/python eda/cross_manual.py     (CPU, ~1 min, 6 seeds)
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from btr.data import CAT_FEATURES, NUM_FEATURES, load_dataset, split_by_query  # noqa: E402

SEEDS = range(42, 48)


def matrices(train_df, otro_df, con_cross):
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    sca = StandardScaler()
    Xc_tr = enc.fit_transform(train_df[CAT_FEATURES])
    Xn_tr = sca.fit_transform(train_df[NUM_FEATURES])
    Xc_ot = enc.transform(otro_df[CAT_FEATURES])
    Xn_ot = sca.transform(otro_df[NUM_FEATURES])
    partes_tr, partes_ot = [Xc_tr, Xn_tr], [Xc_ot, Xn_ot]
    if con_cross:
        # one-hot(status) x [price_rel, price_rel^2]: la U invertida POR tier
        st = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        S_tr = st.fit_transform(train_df[['listing_status']])
        S_ot = st.transform(otro_df[['listing_status']])
        for df_, S_, partes in ((train_df, S_tr, partes_tr), (otro_df, S_ot, partes_ot)):
            pr = df_['price_rel'].to_numpy()[:, None]
            partes.append(S_ * pr)
            partes.append(S_ * pr ** 2)
    return np.hstack(partes_tr), np.hstack(partes_ot)


def main():
    df = load_dataset(REPO / 'supermarket_products.csv')
    res = {False: [], True: []}
    for seed in SEEDS:
        train_df, _, test_df = split_by_query(df, seed=seed)
        y_tr = train_df['bought'].to_numpy()
        y_te = test_df['bought'].to_numpy()
        for cross in (False, True):
            X_tr, X_te = matrices(train_df, test_df, cross)
            clf = LogisticRegression(max_iter=3000, C=1.0)
            clf.fit(X_tr, y_tr)
            res[cross].append(average_precision_score(y_te, clf.predict_proba(X_te)[:, 1]))
        print(f'  seed {seed}: sin cross {res[False][-1]:.4f} | con cross {res[True][-1]:.4f}',
              flush=True)

    sin, con = np.array(res[False]), np.array(res[True])
    d = con - sin
    print(f'\nlogística SIN cross:  {sin.mean():.4f} ± {sin.std():.3f}')
    print(f'logística CON cross:  {con.mean():.4f} ± {con.std():.3f}   '
          f'(Δ {d.mean():+.4f}, gana {(d > 0).sum()}/{len(d)})')
    print('\nreferencias (mismo protocolo): GBM 0.762 · MLP embeddings 0.746 · '
          'mlp_onehot 0.797 · transformer base 0.794 · feat_ordinal 0.824')
    gap_total = 0.824 - sin.mean()
    print(f'del gap logística → transformer final ({gap_total:.3f}), el cross manual '
          f'explica {d.mean() / gap_total:.0%}')


if __name__ == '__main__':
    main()
