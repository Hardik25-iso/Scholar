"""Where the models live and how they are loaded — the one place that knows.

Both models run on ONNX Runtime from the files `scripts/fetch_models.py` pins
and checksums. There is no torch in the serving path: the same weights, exported
by the model authors, executed by a runtime a tenth of the size.

PRECISION. "int8" (the default) is the repositories' own dynamically-quantised
export; "fp32" is the unquantised export, kept for the parity tests and the
eval, which is how int8 earned its place. fp32 mpnet is 436 MB of weights alone,
so on a 512 MB instance int8 is not an optimisation but the only option.

SESSION OPTIONS are tuned for a small, shared CPU, and each one is a measured
memory or latency concern rather than a preference:
  * intra_op_num_threads=1 — on a 0.1-CPU slice extra threads do not run in
    parallel, they queue for the same quota and add contention.
  * enable_cpu_mem_arena=False — the arena keeps the PEAK activation buffer
    allocated forever after the first big batch. Without it, memory returns
    to the allocator between calls, which is the difference between a peak
    and a plateau under a hard RAM limit.
  * ORT_ENABLE_EXTENDED — the full ENABLE_ALL level adds layout transforms
    that mostly pay off on many cores and cost load-time memory here.
  * session.disable_prepacking — by default the runtime keeps a second,
    re-laid-out copy of every quantised weight matrix for faster matmuls.
    Measured in the image: the int8 mpnet session costs +218 MB with it and
    +122 MB without, for ~10% slower inference. On a 512 MB instance that
    copy is the difference between fitting and being OOM-killed at warm-up.

TOKENIZER TRAP. mpnet's tokenizer.json ships with truncation AND padding fixed
at 128 tokens. Loaded as-is, every chunk would be cut to a third of its length
with no error. Callers therefore always get a tokenizer with both DISABLED and
must opt in to exactly what they want — see `tokenizer()`.
"""
import threading
from functools import lru_cache
from pathlib import Path

import onnxruntime as ort
from tokenizers import Tokenizer

EMBEDDER_DIR = "all-mpnet-base-v2"
RERANKER_DIR = "ms-marco-MiniLM-L-6-v2"

_MODEL_FILES = {
    "int8": "onnx/model_quint8_avx2.onnx",
    "fp32": "onnx/model.onnx",
}

DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "models"


def models_dir() -> Path:
    # Read lazily: a module that names a model must be importable without a
    # configured deployment (see the note in reranker.py).
    from backend.config import settings

    return Path(settings.models_dir) if settings.models_dir else DEFAULT_MODELS_DIR


def precision() -> str:
    from backend.config import settings

    if settings.model_precision not in _MODEL_FILES:
        raise ValueError(
            f"MODEL_PRECISION must be one of {sorted(_MODEL_FILES)}, got {settings.model_precision!r}"
        )
    return settings.model_precision


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"model file missing: {path}. Run `python scripts/fetch_models.py "
            f"--precision {precision()}` (or set MODELS_DIR)."
        )
    return path


_sessions: dict[tuple[str, str], ort.InferenceSession] = {}
_sessions_lock = threading.Lock()


def session(model: str, model_precision: str) -> ort.InferenceSession:
    """One InferenceSession per (model, precision), built on first use.

    Built under a lock, NOT with lru_cache: lru_cache does not lock, so when
    the start-up warm-up thread and the first request both miss the cache,
    each builds its own session — two copies of a 120 MB model at once. That
    was the measured cause of a 513 MB peak on a 512 MB instance. A lock held
    across a multi-second model load is fine here: nothing can answer without
    the model anyway, and the second caller simply waits for the first's copy.
    """
    key = (model, model_precision)
    found = _sessions.get(key)
    if found is not None:
        return found
    with _sessions_lock:
        if key not in _sessions:
            _sessions[key] = _build_session(model, model_precision)
        return _sessions[key]


def _build_session(model: str, model_precision: str) -> ort.InferenceSession:
    path = _require(models_dir() / model / _MODEL_FILES[model_precision])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.add_session_config_entry("session.disable_prepacking", "1")
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=4)
def _raw_tokenizer(model: str) -> Tokenizer:
    return Tokenizer.from_file(str(_require(models_dir() / model / "tokenizer.json")))


def tokenizer(model: str) -> Tokenizer:
    """A FRESH tokenizer with truncation and padding switched off.

    Fresh because both settings are mutable state on the object: the chunker
    needs neither, the embedder needs both, and sharing one instance would let
    whichever ran last decide for the other. Construction is a cheap clone of
    the cached, parsed vocabulary.
    """
    tok = Tokenizer.from_str(_raw_tokenizer(model).to_str())
    tok.no_truncation()
    tok.no_padding()
    return tok
