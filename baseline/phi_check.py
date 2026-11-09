"""Empirical checks on the readout Φ (utils.complex_format_fn) for one format.

Run from repo root:  python -m baseline.phi_check [--format trispecific]
"""
import argparse

import torch
from torch_geometric.utils import to_dense_adj

from constants._connectivity import CONNECTIVITY_TYPES, CONNECTIVITY_WEIGHTS
from utils import complex_format_fn, seed_all

FABS = {'bispecific': 2, 'trispecific': 3, 'tetraspecific': 4, 'pentaspecific': 5}


def phi(f, ei, fmt):
    return float(complex_format_fn(f, ei, complex_format=fmt))


def closed_form_coeffs(ei, w):
    """Φ(f) = Σ_i c_i f_i with c_i = w_i (1 + outdeg_i + #2-paths_i) / n, read off the code."""
    A = to_dense_adj(ei, max_num_nodes=len(w))[0]
    ones = torch.ones(len(w))
    return w * (1 + A @ ones + A @ A @ ones) / len(w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--format', default='trispecific')
    p.add_argument('--trials', type=int, default=1000)
    args = p.parse_args()
    fmt, k = args.format, FABS[args.format]
    topos, weights = CONNECTIVITY_TYPES[fmt], CONNECTIVITY_WEIGHTS[fmt]
    n = len(weights[0])
    seed_all(0)

    def rand_f():
        f = torch.zeros(n)
        f[:k] = torch.rand(k) * 20  # Fab values in the range Ehrlich produces; Fc entries zeroed
        return f

    print(f'format={fmt}  topologies={len(topos)}  nodes={n} ({k} Fab + {n - k} Fc)')

    # Structure of A
    A0 = to_dense_adj(topos[0])[0]
    print(f'\nA (topology 0) symmetric? {torch.equal(A0, A0.T)}   edges stored: {topos[0].tolist()}')
    print('rows with nonzero out-degree are always the lower node index of each edge:',
          all(bool((ei[0] < ei[1]).all()) for ei in topos))

    # (1) Homogeneity and additivity across all topologies
    hom_abs, hom_rel, add_abs, add_rel, cf_abs = [], [], [], [], []
    alphas = torch.tensor([-3.0, -0.5, 0.0, 0.1, 2.0, 7.5])
    for t in range(args.trials):
        ti = t % len(topos)
        ei = topos[ti]
        f, g = rand_f(), rand_f()
        pf, pg = phi(f, ei, fmt), phi(g, ei, fmt)
        a = float(alphas[t % len(alphas)])
        lhs = phi(a * f, ei, fmt)
        hom_abs.append(abs(lhs - a * pf)); hom_rel.append(abs(lhs - a * pf) / max(abs(a * pf), 1e-12))
        lhs = phi(f + g, ei, fmt)
        add_abs.append(abs(lhs - (pf + pg))); add_rel.append(abs(lhs - (pf + pg)) / abs(pf + pg))
        cf_abs.append(abs(pf - float(closed_form_coeffs(ei, weights[ti]) @ f)))

    t = lambda xs: torch.tensor(xs)
    print(f'\nLinearity over {args.trials} random (f, g, α, topology) draws, float32:')
    print(f'  Φ(αf) - αΦ(f)        max abs {t(hom_abs).max():.3e}   max rel {t(hom_rel).max():.3e}')
    print(f'  Φ(f+g) - Φ(f) - Φ(g)  max abs {t(add_abs).max():.3e}   max rel {t(add_rel).max():.3e}')
    print(f'  Φ(f) - c·f (closed form)  max abs {t(cf_abs).max():.3e}')
    print(f'  Φ(0) = {phi(torch.zeros(n), topos[0], fmt)}  (no intercept)')

    # A worked example with concrete numbers
    ei, f, g = topos[0], torch.tensor([3., 5., 2.] + [0.] * (n - 3)), torch.tensor([1., 4., 6.] + [0.] * (n - 3))
    print(f'\nWorked example, topology 0, f={f.tolist()}, g={g.tolist()}:')
    print(f'  Φ(f)={phi(f, ei, fmt):.6f}  Φ(g)={phi(g, ei, fmt):.6f}  Φ(f+g)={phi(f + g, ei, fmt):.6f}  '
          f'Φ(f)+Φ(g)={phi(f, ei, fmt) + phi(g, ei, fmt):.6f}')
    print(f'  Φ(2.5f)={phi(2.5 * f, ei, fmt):.6f}  2.5Φ(f)={2.5 * phi(f, ei, fmt):.6f}')

    # (2) Per-node term decomposition and effective coefficients
    print('\nPer-topology effective coefficients c_i = w_i(1 + outdeg_i + #2paths_i)/n  (Fab entries only):')
    for ti, (ei, w) in enumerate(zip(topos, weights)):
        A = to_dense_adj(ei, max_num_nodes=n)[0]
        o = torch.ones(n)
        out1, out2 = (A @ o)[:k], (A @ A @ o)[:k]
        c = closed_form_coeffs(ei, w)[:k]
        print(f'  topo {ti:2d}  edges {[tuple(e) for e in ei.T.tolist()]}')
        print(f'           w_fab {[round(x, 2) for x in w[:k].tolist()]}  outdeg {out1.int().tolist()}  '
              f'2-paths {out2.int().tolist()}  c {[round(x, 3) for x in c.tolist()]}')

    # (3) Fc nodes: their weights are nonzero, so the zero f-entry is what removes them
    f = rand_f(); f_fc = f.clone(); f_fc[k:] = 1.0
    print(f'\nFc weights (topo 0): {[round(x, 2) for x in weights[0][k:].tolist()]}')
    print(f'Φ(f with Fc=0) = {phi(f, topos[0], fmt):.4f}   Φ(f with Fc=1) = {phi(f_fc, topos[0], fmt):.4f}')

    # (4) How much of Var(y) is topology? Best connectivity-blind predictor is E_topo[c]·f
    seed_all(1)
    F = torch.zeros(20000, n); F[:, :k] = torch.rand(20000, k) * 20
    C = torch.stack([closed_form_coeffs(ei, w) for ei, w in zip(topos, weights)])
    T = torch.randint(0, len(topos), (20000,))
    y = (C[T] * F).sum(1)
    y_blind = F @ C.mean(0)
    print(f'\nWith uniform Fab values and uniform topology: sd(y) = {y.std():.3f}; '
          f'RMSE of best topology-blind linear predictor c̄·f = {(y - y_blind).pow(2).mean().sqrt():.3f}')


if __name__ == '__main__':
    main()
