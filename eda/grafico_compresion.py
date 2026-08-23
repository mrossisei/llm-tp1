"""Figura de la 7ma mini-tanda: el piso de la compresion, con y sin teacher.

    .venv/bin/python eda/grafico_compresion.py   ->  graficos/curva_compresion.png

Dos ramas de la curva "test PR-AUC vs parametros" (353 -> 26.177, eje log):
entrenar con las labels duras (plain) vs destilar las probabilidades del
deep-ensemble del mismo split (soft labels, clase 3). La lectura: el teacher
corre el piso de la compresion — el nivel campeon aguanta hasta 1.937
parametros destilando (plain ya degrado ahi), y a 353 ambas ramas caen juntas
(no hay donde guardar el conocimiento). Paleta validada del repo.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
TEAL, VIOLETA = '#0E9B7E', '#7052C9'   # destilada / plain (par validado CVD)

NIVELES = [  # (etiqueta, tag plain, tag destilada)
    ('d4·1b',  'min_d4l1_features_d4_h4_l1_linear_catordinal',
               'tl_distill_ens_d4l1_features_d4_h4_l1_linear_catordinal_dst1'),
    ('d8·1b',  'min_d8l1_features_d8_h4_l1_linear_catordinal',
               'tl_distill_ens_d8l1_features_d8_h4_l1_linear_catordinal_dst1'),
    ('d8·2b',  'min_d8_features_d8_h4_l2_linear_catordinal',
               'tl_distill_ens_d8_features_d8_h4_l2_linear_catordinal_dst1'),
    ('d16·1b', 'min_d16l1_features_d16_h4_l1_linear_catordinal',
               'tl_distill_ens_min_features_d16_h4_l1_linear_catordinal_dst1'),
    ('d32·2b', 'feat_ordinal_features_d32_h4_l2_linear_catordinal',
               'tl_distill_ens_features_d32_h4_l2_linear_catordinal_dst1'),
]


def grupo(tag):
    prs, params = [], None
    for s in range(42, 48):
        d = json.loads((REPO / 'resultados' / f'{tag}_seed{s}.json').read_text())
        prs.append(d['test']['pr_auc'])
        params = d['n_parametros']
    m = sum(prs) / len(prs)
    sd = (sum((x - m) ** 2 for x in prs) / len(prs)) ** 0.5
    return params, m, sd


def main():
    filas = [(et, *grupo(tp), *grupo(td)[1:]) for et, tp, td in NIVELES]
    xs = [f[1] for f in filas]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for idx, color, nombre in [(2, VIOLETA, 'labels duras (plain)'),
                               (4, TEAL, 'destilada del deep-ensemble (0.833)')]:
        ys = [f[idx] for f in filas]
        es = [f[idx + 1] for f in filas]
        ax.errorbar(xs, ys, yerr=es, color=color, lw=2, marker='o', ms=7,
                    capsize=3, label=nombre, zorder=3)
        ax.annotate(f'{ys[-1]:.3f}', (xs[-1], ys[-1]), textcoords='offset points',
                    xytext=(10, -4), fontsize=9, color='#444444')

    campeon = filas[-1][2]
    ax.axhline(campeon, color='#999999', lw=1, ls='--', zorder=1)
    ax.annotate(f'campeón d32 ({campeon:.3f})', (xs[0], campeon),
                textcoords='offset points', xytext=(0, 5), fontsize=9, color='#777777')
    # el punto de la historia: 1.937 params destilada = nivel campeon
    f8 = filas[2]
    ax.annotate('nivel campeón\ncon 1.937 params', (f8[1], f8[4]),
                textcoords='offset points', xytext=(-12, 22), fontsize=9,
                color=TEAL, ha='right')

    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{f[1]:,}\n{f[0]}' for f in filas], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel('parámetros (log)')
    ax.set_ylabel('test PR-AUC (media ± desvío, 6 seeds)')
    ax.set_title('El piso de la compresión: las soft labels lo corren un nivel\n'
                 '(plain degrada bajo 3.713 params; destilada aguanta hasta 1.937; '
                 'a 353 caen juntas)', fontsize=11)
    ax.grid(True, axis='y', alpha=0.25, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    fig.tight_layout()
    out = REPO / 'graficos' / 'curva_compresion.png'
    fig.savefig(out, dpi=140)
    print(f'escrito {out.relative_to(REPO)}')
    for f in filas:
        print(f'  {f[0]:7s} {f[1]:>7,} | plain {f[2]:.4f}±{f[3]:.3f} | dest {f[4]:.4f}±{f[5]:.3f}')


if __name__ == '__main__':
    main()
