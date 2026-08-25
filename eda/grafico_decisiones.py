"""Figura para la presentacion: UNA decision por eje experimental (plan de Juan).

    .venv/bin/python eda/grafico_decisiones.py  ->  graficos/decisiones_por_eje.png

Seis paneles, uno por eje que se vario sobre la base d32/h4/l2: formulacion
(que es un token), encoding de categoricas, cabezas, profundidad, capacidad
(d_model) y pre-entrenamiento propio. En cada panel el punto teal es la opcion
ELEGIDA para el modelo final; violeta las alternativas medidas; gris las
mediciones de contexto sobre otra base. Numeros = test PR-AUC (media +- desvio,
6 seeds) de resultados/, los mismos de analisis.md.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
TEAL, VIOLETA, GRIS = '#0E9B7E', '#7052C9', '#98A5B3'   # elegida / alternativa / contexto

# (etiqueta, media, desvio, rol)  rol: 'e'=elegida  'a'=alternativa  'c'=contexto
PANELES = [
    ('¿Qué es un token? (formulación)', [
        ('features (cada feature)', .7981, .0357, 'e'),
        ('fusion (features + resumen texto)', .7748, .0364, 'a'),
        ('tower (torre de texto aparte)', .7724, .0220, 'a'),
        ('listwise (productos de la query)', .7395, .0358, 'a'),
        ('hybrid (features + 256 chars)', .7349, .0307, 'a'),
        ('text (solo caracteres)', .6339, .0337, 'a'),
        ('MLP (control sin atención)', .7503, .0334, 'c'),
    ], 'features gana; la atención aporta +0.048 vs MLP'),
    ('Encoding de categóricas (2ª–3ª tanda)', [
        ('ordinal (rango por BTR)', .8239, .0178, 'e'),
        ('target (BTR suavizado)', .8134, .0245, 'a'),
        ('embedding aprendido', .7981, .0357, 'a'),
        ('one-hot (MLP)', .7973, .0318, 'c'),
        ('hashing (8 buckets)', .4981, .0412, 'a'),
        ('freq (frecuencia)', .2176, .0953, 'a'),
    ], 'el prior escalar y monótono gana; freq destruye la señal'),
    ('Cabezas de atención', [
        ('4 cabezas · ordinal', .8239, .0178, 'e'),
        ('1 cabeza · ordinal', .8000, .0306, 'a'),
        ('1 cabeza · embeddings', .8157, .0258, 'c'),
        ('2 cabezas · embeddings', .7953, .0099, 'c'),
        ('4 cabezas · embeddings', .7981, .0357, 'c'),
    ], 'interactúa con el encoding: decidido por val'),
    ('Profundidad (bloques)', [
        ('2 bloques · ordinal', .8239, .0178, 'e'),
        ('4 bloques · ordinal', .8110, .0471, 'a'),
        ('1 bloque · ordinal', .8009, .0499, 'a'),
        ('4 bloques · embeddings', .8030, .0180, 'c'),
        ('1 bloque · embeddings', .7754, .0416, 'c'),
    ], 'más capacidad no compra nada (26k → 51k params)'),
    ('d_model (capacidad, ordinal, 2 bloques)', [
        ('d32 · 26.177 params', .8239, .0178, 'e'),
        ('d16 · 6.945', .8154, .0233, 'a'),
        ('d8 · 1.937', .8142, .0248, 'a'),
        ('d16 · 1 bloque · 3.713', .8254, .0255, 'c'),
        ('d64 · embeddings · 105.729', .8149, .0269, 'c'),
    ], 'meseta amplia: la señal cabe en poquísimos parámetros'),
    ('Pre-entrenamiento propio (init)', [
        ('sin pre-entrenamiento · ordinal', .8239, .0178, 'e'),
        ('+ MLM 20 épocas · ordinal', .8168, .0298, 'a'),
        ('+ MLM 20 épocas · embeddings', .8089, .0342, 'c'),
        ('sin pre-entrenamiento · embeddings', .7937, .0335, 'c'),
        ('words + w2v-init (fusion)', .7571, .0542, 'c'),
    ], 'MLM ayuda a embeddings; sobre ordinal, no'),
]


def main():
    fig, axs = plt.subplots(2, 3, figsize=(14.5, 8.6))
    for ax, (titulo, filas, lectura) in zip(axs.flat, PANELES):
        n = len(filas)
        for i, (et, m, sd, rol) in enumerate(filas):
            y = n - 1 - i
            color = {'e': TEAL, 'a': VIOLETA, 'c': GRIS}[rol]
            ax.errorbar(m, y, xerr=sd, color=color, marker='o',
                        ms=7 if rol == 'e' else 5.5, capsize=2.5,
                        lw=1.8 if rol == 'e' else 1.3, zorder=3 if rol == 'e' else 2)
            ax.annotate(f'{m:.3f}', (m, y), textcoords='offset points',
                        xytext=(0, 7), fontsize=7.5, ha='center', color='#555555')
        ax.set_yticks(range(n), [f[0] for f in reversed(filas)], fontsize=8)
        ax.set_ylim(-0.7, n - 0.3)
        ax.tick_params(axis='x', labelsize=7.5)
        ax.set_title(titulo, fontsize=10.5, fontweight='bold', loc='left')
        ax.text(0, 1.005, ' ', transform=ax.transAxes)  # aire bajo el titulo
        ax.set_xlabel(lectura, fontsize=8, color='#555555')
        ax.grid(True, axis='x', alpha=0.25, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=TEAL, marker='o', lw=1.8, label='elegida para el modelo final'),
        Line2D([], [], color=VIOLETA, marker='o', lw=1.3, label='alternativa medida (misma base)'),
        Line2D([], [], color=GRIS, marker='o', lw=1.3, label='contexto (otra base)'),
    ], loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle('Una decisión por eje — test PR-AUC (media ± desvío, 6 seeds) · base d32/h4/l2',
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    out = REPO / 'graficos' / 'decisiones_por_eje.png'
    fig.savefig(out, dpi=140)
    print(f'escrito {out.relative_to(REPO)}')




def singles():
    """Versiones individuales tamano-diapositiva (una por eje, tipografia grande)."""
    nombres = ['formulacion', 'encoding', 'cabezas', 'bloques', 'dmodel', 'init']
    for nombre, (titulo, filas, lectura) in zip(nombres, PANELES):
        n = len(filas)
        fig, ax = plt.subplots(figsize=(7.6, 0.62 * n + 1.7))
        for i, (et, m, sd, rol) in enumerate(filas):
            y = n - 1 - i
            color = {'e': TEAL, 'a': VIOLETA, 'c': GRIS}[rol]
            ax.errorbar(m, y, xerr=sd, color=color, marker='o',
                        ms=10 if rol == 'e' else 8, capsize=4,
                        lw=2.4 if rol == 'e' else 1.8, zorder=3 if rol == 'e' else 2)
            ax.annotate(f'{m:.3f}', (m, y), textcoords='offset points',
                        xytext=(0, 10), fontsize=11, ha='center', color='#444444',
                        fontweight='bold' if rol == 'e' else 'normal')
        ax.set_yticks(range(n), [f[0] for f in reversed(filas)], fontsize=12)
        ax.set_ylim(-0.75, n - 0.25)
        ax.tick_params(axis='x', labelsize=10)
        ax.set_xlabel('test PR-AUC (media ± desvío, 6 seeds)',
                      fontsize=10.5, color='#555555')
        ax.grid(True, axis='x', alpha=0.25, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        out = REPO / 'graficos' / f'decision_{nombre}.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'escrito {out.relative_to(REPO)}')


if __name__ == '__main__':
    main()
    singles()
