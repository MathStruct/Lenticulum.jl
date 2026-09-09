This whole repository is an Obsidian vault. 
Please download Obsidian and install the following Addons:
- Inline TikZ
- Wypst

you can write then formulas in either typst syntax or the normal Latex syntax.
You can write a TikZ diagram that gets automatically rendered in the form:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
X \arrow[r, "f"] \arrow[dr, "f"'] \arrow[d, "\mathrm{id}_X"'] & Y \arrow[d, "\mathrm{id}_Y"] \\
X \arrow[r, "f"'] & Y
\end{tikzcd}
\end{document}
```

If you implement something please put a markdown file with all the description of the theory, the implementation and an extensive description of implementation difficulties right next to the .jl file containing the implementation.

The theory notes for this library are indexed in [[Index]]. Start there.
