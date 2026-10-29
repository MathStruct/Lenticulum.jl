# Poster

A one-page poster of Lenticulum's core ideas, an HTML page with inline SVG diagrams.

- `poster.html`: open it in a browser. Printing gives A0 portrait (KaTeX loads from cdnjs, so
  the formulas need a network connection).
- `circle_field.py`: draws `circle_field.svg`, the score field in panel 2 (an illustration: a
  circle smoothed at σ = 0.3, not a trained model). Run `python3 poster/circle_field.py`.
- `poster_preview.png`: a screenshot (`firefox --headless --screenshot`).
