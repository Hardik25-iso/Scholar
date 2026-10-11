"""Download the ONNX models Scholar runs, pinned and checksummed.

    python scripts/fetch_models.py [DEST] [--precision int8|fp32|all]

DEST defaults to MODELS_DIR, else ./models. Used by the Dockerfile (which bakes
int8 into the image) and by local development, so both read the exact same
bytes. Standard library only: the image needs no download tooling beyond the
Python it already has.

PINNED TO A COMMIT, NOT `main`. A model repository can be updated in place; a
silently changed weight file would change every embedding with no code change
to point at. Each file is fetched from an immutable revision URL and its
sha256 is checked, so a different byte anywhere fails the build instead.

The ONNX files are the repositories' own exports (sentence-transformers'
`onnx/` folder), so nothing is converted here and no exporter is a dependency.
"""
import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co/{repo}/resolve/{rev}/{path}"

# (repo, revision, local dir name)
EMBEDDER = ("sentence-transformers/all-mpnet-base-v2",
            "e8c3b32edf5434bc2275fc9bab85f82640a19130", "all-mpnet-base-v2")
RERANKER = ("cross-encoder/ms-marco-MiniLM-L-6-v2",
            "233902d25c440f23af6f7d6e94d2946bac0bee0a", "ms-marco-MiniLM-L-6-v2")

# path -> sha256, per model. tokenizer.json is needed at every precision.
FILES = {
    EMBEDDER: {
        "tokenizer.json": ("all", "b8be2c30ba5dd723a6d5ee26d013da103d5408d92ddcb23747622f9e48f1d842"),
        "onnx/model.onnx": ("fp32", "74187b16d9c946fea252e120cfd7a12c5779d8b8b86838a2e4c56573c47941bd"),
        "onnx/model_quint8_avx2.onnx": ("int8", "aa5c27172d77bbd1cbae3628cbac4b26d7c12adabff25d2d4285d0f29159b237"),
    },
    RERANKER: {
        "tokenizer.json": ("all", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
        "onnx/model.onnx": ("fp32", "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a"),
        "onnx/model_quint8_avx2.onnx": ("int8", "c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9"),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, partial: Path, attempts: int = 8) -> None:
    """Download with resume. A dropped connection mid-file is routine on a slow
    link, and urllib reports a short read as a normal end of stream — so the
    length is checked against Content-Range and the transfer resumed with a
    Range request until it is complete. The checksum is still the final word."""
    total = None
    for attempt in range(1, attempts + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if total is not None and have >= total:
            return
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:      # server ignored Range: restart
                    have = 0
                    partial.unlink(missing_ok=True)
                length = int(response.headers.get("Content-Length", "0") or 0)
                total = have + length if length else total
                with partial.open("ab") as out:
                    while block := response.read(1 << 20):
                        out.write(block)
        except (OSError, urllib.error.URLError) as exc:
            print(f"  interrupted at {partial.stat().st_size if partial.exists() else 0} bytes "
                  f"({exc}); resuming ({attempt}/{attempts})", flush=True)
            continue
        if total is None or partial.stat().st_size >= total:
            return
    raise SystemExit(f"could not download {url} after {attempts} attempts")


def fetch(dest: Path, precision: str) -> None:
    for (repo, rev, name), files in FILES.items():
        for rel, (needed_for, expected) in files.items():
            if needed_for not in ("all", precision) and precision != "all":
                continue
            target = dest / name / rel
            if target.exists() and _sha256(target) == expected:
                print(f"ok      {name}/{rel}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            url = HUB.format(repo=repo, rev=rev, path=rel)
            print(f"fetch   {name}/{rel}", flush=True)
            _download(url, partial)
            actual = _sha256(partial)
            if actual != expected:
                partial.unlink()
                raise SystemExit(f"checksum mismatch for {name}/{rel}: got {actual}, pinned {expected}")
            os.replace(partial, target)  # only a verified file ever has the real name


def main() -> int:
    parser = argparse.ArgumentParser(prog="python scripts/fetch_models.py")
    parser.add_argument("dest", nargs="?", default=os.getenv("MODELS_DIR", "models"))
    parser.add_argument("--precision", choices=("int8", "fp32", "all"), default="int8")
    args = parser.parse_args()
    fetch(Path(args.dest), args.precision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
