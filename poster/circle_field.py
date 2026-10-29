"""Draws circle_field.svg for the poster: the score of a smoothed circle and three queries.

An illustration, not a trained model: p is a mixture of 48 Gaussians (std SIGMA) on the unit
circle, and each query is gradient ascent of log p in the free coordinate, the other one clamped.
Run: python3 poster/circle_field.py
"""
import numpy as np
from pathlib import Path

SIGMA = 0.3
th = np.linspace(0, 2 * np.pi, 49)[:-1]
C = np.stack([np.cos(th), np.sin(th)], 1)

def score(z):
    d = C - z
    w = np.exp(-(d ** 2).sum(1) / (2 * SIGMA ** 2))
    w /= w.sum()
    return (w[:, None] * d).sum(0) / SIGMA ** 2

def solve(z0, free, steps=400, eta=0.02):
    z = np.array(z0, float); path = [z.copy()]
    for _ in range(steps):
        z[free] += eta * score(z)[free]
        path.append(z.copy())
    return np.array(path)

W, L = 520, 1.55                                  # pixels, half-width in data units
px = lambda x: (x + L) / (2 * L) * W
py = lambda y: (L - y) / (2 * L) * W
out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {W}" font-family="Inter, Helvetica, Arial, sans-serif">',
       '<defs><marker id="h" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0L6,3L0,6z" fill="#9aa7b8"/></marker>',
       '<marker id="hq" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L6,3L0,6z" fill="context-stroke"/></marker></defs>',
       f'<rect width="{W}" height="{W}" fill="#f7f8fb"/>']
for x in np.linspace(-1.4, 1.4, 15):
    for y in np.linspace(-1.4, 1.4, 15):
        s = score(np.array([x, y])); n = np.linalg.norm(s)
        if n < 1e-9: continue
        d = s / n * 0.075 * np.tanh(n / 2)
        out.append(f'<line x1="{px(x):.1f}" y1="{py(y):.1f}" x2="{px(x+d[0]):.1f}" y2="{py(y+d[1]):.1f}" stroke="#9aa7b8" stroke-width="1.6" marker-end="url(#h)"/>')
out.append(f'<circle cx="{px(0)}" cy="{py(0)}" r="{W/(2*L):.1f}" fill="none" stroke="#1f2a44" stroke-width="3"/>')
out.append(f'<text x="{px(-0.98):.0f}" y="{py(0.92):.0f}" font-size="20" font-style="italic" fill="#1f2a44">R</text>')

queries = [((0.6, 0.5), 1, "#d1495b"), ((0.6, -0.5), 1, "#d1495b"),
           ((-0.5, 0.6), 0, "#2e86ab"), ((1.05, 0.5), 1, "#e09f3e")]
out.append(f'<line x1="{px(0.6):.1f}" y1="{py(1.5):.1f}" x2="{px(0.6):.1f}" y2="{py(-1.5):.1f}" stroke="#d1495b" stroke-dasharray="5 5" stroke-width="1.5"/>')
out.append(f'<line x1="{px(-1.5):.1f}" y1="{py(0.6):.1f}" x2="{px(1.5):.1f}" y2="{py(0.6):.1f}" stroke="#2e86ab" stroke-dasharray="5 5" stroke-width="1.5"/>')
out.append(f'<line x1="{px(1.05):.1f}" y1="{py(1.5):.1f}" x2="{px(1.05):.1f}" y2="{py(-1.5):.1f}" stroke="#e09f3e" stroke-dasharray="5 5" stroke-width="1.5"/>')
ends = []
for z0, free, col in queries:
    p = solve(z0, free)
    pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in p[::4])
    out.append(f'<circle cx="{px(z0[0]):.1f}" cy="{py(z0[1]):.1f}" r="5" fill="white" stroke="{col}" stroke-width="2.5"/>')
    out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3" marker-end="url(#hq)"/>')
    out.append(f'<circle cx="{px(p[-1,0]):.1f}" cy="{py(p[-1,1]):.1f}" r="7" fill="{col}"/>')
    ends.append(p[-1])
lab = lambda x, y, t, c, a="start": out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="17" font-weight="600" fill="{c}" text-anchor="{a}">{t}</text>')
lab(px(0.6) + 6, py(1.5) + 18, "x = 0.6", "#d1495b")
lab(px(-1.5) + 6, py(0.6) - 8, "y = 0.6", "#2e86ab")
lab(px(1.05) + 6, py(-1.5) - 8, "x = 1.05", "#e09f3e")
out.append("</svg>")
here = Path(__file__).parent
(here / "circle_field.svg").write_text("\n".join(out))
for (z0, free, _), e in zip(queries, ends):
    print(z0, "free", "xy"[free], "->", np.round(e, 3))
