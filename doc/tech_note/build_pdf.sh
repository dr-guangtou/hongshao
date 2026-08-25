#!/bin/zsh
# Rebuild the deposition-model tech note PDF.
# Equations are LaTeX math ($$...$$); xelatex renders them.
# NOTE: pandoc will not treat `$...$` as math if the closing `$` is followed by
# a digit, which silently leaks the LaTeX command into text mode. Keep inline
# math away from adjacent digits.
cd /Users/shuang/Dropbox/work/project/massive/hongshao
pandoc doc/tech_note/deposition_model_2026-08.md \
  -o doc/tech_note/deposition_model_2026-08.pdf --pdf-engine=xelatex \
  -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true -V linkcolor=RoyalBlue \
  -V mainfont="Helvetica Neue" -V monofont="Menlo" --toc --toc-depth=2
