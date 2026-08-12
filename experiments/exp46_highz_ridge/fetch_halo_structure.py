"""exp46 — fetch the TNG "Halo Structure" supplementary catalog (fitted NFW
concentration at twenty epochs).

Why: the project only holds `c200c` at z=0.4. The TNG *supplementary*
catalog "Halo Structure" (Anbajagane et al., arXiv:2109.02713) fits an NFW
profile with a free scale radius and reports c200c = R200c / r_s at twenty
epochs up to z=12 — covering all five profile epochs. Neither the standard
FoF Group catalog nor the Subhalo catalog carries any concentration field
(verified against the official TNG specification page), so this catalog is
the only route to a DIRECTLY FITTED concentration history.

Likely provenance note: the project's existing z=0.4 bundle carries
`c_200c`, `c_to_a_3d`, `b_to_a_3d`, `v_sigma_3d`, `acc_rate` — an almost
exact match to that paper's five properties. If it is the same catalog,
the other epochs will drop straight into the existing feature pipeline.

REQUIRES an API key at ``~/.tng_api_key`` (line ``TNG_API_KEY=...``), the
same convention exp27 used. The key is NOT in the repo and is currently
absent on this machine; get one from https://www.tng-project.org/users/
(free registration) and write the file.

Because the exact catalog filename is not recoverable from the truncated
public specification page, run ``list`` FIRST — with a key, the API returns
the available supplementary files and we pick the right name from that,
rather than guessing a URL.

Run:
    PYTHONPATH=. uv run python experiments/exp46_highz_ridge/\
fetch_halo_structure.py list            # discover available files
    PYTHONPATH=. uv run python experiments/exp46_highz_ridge/\
fetch_halo_structure.py get <filename>  # download one
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "outputs" / "supplementary"
KEY_FILE = Path.home() / ".tng_api_key"
SIM = "TNG300-1"
BASE = f"https://www.tng-project.org/api/{SIM}/files/"


def api_key():
    if not KEY_FILE.exists():
        raise SystemExit(
            f"No API key at {KEY_FILE}.\n"
            "  1. register (free) at https://www.tng-project.org/users/\n"
            f"  2. write the key:  echo 'TNG_API_KEY=<your key>' > {KEY_FILE}\n"
            "  3. re-run this script.\n"
            "The key must stay OUTSIDE the repo (exp27's convention).")
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith("TNG_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"no TNG_API_KEY= line in {KEY_FILE}")


def _get(url, key, timeout=180):
    req = urllib.request.Request(url, headers={"api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} on {url}\n  {e.read()[:400]!r}")


def cmd_list():
    """The catalog is a DIRECTORY endpoint whose members are named by
    snapshot (halo_structure.<snap>.hdf5), so list one level deeper."""
    key = api_key()
    body, _ = _get(BASE + "halo_structure/", key)
    files = json.loads(body)["files"]
    snaps = sorted(int(f.rstrip("/").split(".")[-2]) for f in files)
    print(f"halo_structure: {len(files)} files, by snapshot:\n  {snaps}")
    have = [s for _, s in EPOCH_SNAPS]
    print(f"\nprofile epochs needed: {have}")
    print("all present:", set(have) <= set(snaps))


def cmd_get(which):
    key = api_key()
    CACHE.mkdir(parents=True, exist_ok=True)
    snaps = ([s for _, s in EPOCH_SNAPS] if which == "epochs"
             else [int(which)])
    for s in snaps:
        name = f"halo_structure.{s}.hdf5"
        dst = CACHE / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  cached  {name}  ({dst.stat().st_size / 1e6:.1f} MB)")
            continue
        body, _ = _get(BASE + name, key, timeout=1800)
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.rename(dst)          # never leave a truncated file at the real name
        print(f"  fetched {name}  ({len(body) / 1e6:.1f} MB)")
    import h5py
    with h5py.File(CACHE / f"halo_structure.{snaps[-1]}.hdf5", "r") as f:
        print(f"\nstructure of halo_structure.{snaps[-1]}.hdf5:")

        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"   {name:<44s} {str(obj.shape):>14s} {obj.dtype}")
        f.visititems(show)
        if f.attrs:
            print("   attrs:", dict(list(f.attrs.items())[:10]))


EPOCH_SNAPS = ((0.4, 72), (0.7, 59), (1.0, 50), (1.5, 40), (2.0, 33))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        cmd_list()
    elif cmd == "get":
        cmd_get(sys.argv[2] if len(sys.argv) > 2 else "epochs")
    else:
        raise SystemExit(__doc__)
