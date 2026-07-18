#!/bin/bash
# Render the tech notes to PDF (pandoc + xelatex).
#
# Usage: bash doc/tech_note/scripts/build_pdf.sh    (from the repo root)
#
# Output: doc/tech_note/pdf/{01,02,03}_*.pdf + tech_notes_combined.pdf
# (derived artifacts, gitignored — regenerate after editing the notes).
#
# Preprocessing before pandoc sees the Markdown:
#   * GIF references are swapped for a representative extracted frame
#     (PDF cannot animate; the caption points back to the repository GIF);
#   * cross-note .md links are flattened to plain text (they would be
#     dead file links inside a PDF).
set -euo pipefail

note_dir="$(cd "$(dirname "$0")/.." && pwd)"
repo_root="$(cd "$note_dir/../.." && pwd)"
out_dir="$note_dir/pdf"
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

mkdir -p "$out_dir" "$build_dir/figures"
cp "$note_dir"/figures/*.png "$build_dir/figures/"

# static frames standing in for the animations
(cd "$repo_root" && uv run python - "$note_dir" "$build_dir" <<'EOF'
import sys
from pathlib import Path

from PIL import Image

note_dir, build_dir = Path(sys.argv[1]), Path(sys.argv[2])
for gif_name, frame in (("note2_assembly.gif", 44),
                        ("note1_pca_sweep.gif", 12)):
    image = Image.open(note_dir / "figures" / gif_name)
    image.seek(frame)
    out = build_dir / "figures" / gif_name.replace(".gif", "_frame.png")
    image.convert("RGB").save(out)
EOF
)

for name in README 01_statistical_emulator 02_transport_kernel \
            03_two_channel_alternative; do
  sed -E \
    -e 's/\[([^]]*)\]\(0[0-9]_[a-z_]*\.md\)/\1/g' \
    -e 's/\[([^]]*)\]\(README\.md\)/\1/g' \
    -e 's/\[([^]]*)\]\(scripts\/\)/\1/g' \
    -e 's#\[`note1_pca_sweep\.gif`\]\(figures/note1_pca_sweep\.gif\)#the repository GIF `note1_pca_sweep.gif`#' \
    -e 's#!\[The assembly animated\]\(figures/note2_assembly\.gif\)#![The assembly (animation frame; the full GIF is in the repository)](figures/note2_assembly_frame.png)#' \
    "$note_dir/$name.md" > "$build_dir/$name.md"
done

pandoc_flags=(--pdf-engine=xelatex --resource-path="$build_dir"
              -V geometry:margin=2.7cm -V fontsize=11pt -V colorlinks=true
              -V urlcolor=NavyBlue -V linkcolor=NavyBlue)

for name in 01_statistical_emulator 02_transport_kernel \
            03_two_channel_alternative; do
  pandoc "$build_dir/$name.md" -o "$out_dir/$name.pdf" "${pandoc_flags[@]}"
  echo "wrote $out_dir/$name.pdf"
done

printf '\\newpage\n' > "$build_dir/pagebreak.md"
pandoc "$build_dir/README.md" "$build_dir/pagebreak.md" \
       "$build_dir/01_statistical_emulator.md" "$build_dir/pagebreak.md" \
       "$build_dir/02_transport_kernel.md" "$build_dir/pagebreak.md" \
       "$build_dir/03_two_channel_alternative.md" \
       -o "$out_dir/tech_notes_combined.pdf" "${pandoc_flags[@]}" \
       --toc --toc-depth=2 \
       -V title="HongShao tech notes — the emulators"
echo "wrote $out_dir/tech_notes_combined.pdf"
