# Exp50--Exp51 technical report

This directory contains the LaTeX source for the working report
`A Direct Probabilistic Connection Between Halo Mass Assembly and the Stellar
Curves of Growth of Massive Galaxies in TNG300`.

The report is organized as one main file, one bibliography, and separate
section files. The generated PDF and staged figure copies are intentionally
gitignored because they can be rebuilt from the experiment products.

## Build

First generate the Exp50 and Exp51 experiment figures. Then copy their PDF
versions into this directory's `figures/` subdirectory, retaining the filenames
used by `report.tex`. From this directory, compile with:

```bash
latexmk -pdf report.tex
```

The compiled document is written to `build/report.pdf`. Remove auxiliary build
files with:

```bash
latexmk -C report.tex
```

## Organization

- `report.tex`: document setup, title material, and section ordering.
- `sections/`: scientific narrative, methods, results, audit, and appendices.
- `references.bib`: cited simulation, halo-history, interpolation, scoring, and
  symbolic-regression references.
- `latexmkrc`: reproducible build settings.
- `figures/`: staged experiment figures; gitignored.
- `build/`: the PDF and LaTeX auxiliary files; gitignored.

The numerical results remain sourced from the Exp50 and Exp51 output files.
This report interprets them but does not create a second independent result
record; the two experiment README files remain the concise scientific records.
