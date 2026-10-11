"""Measure the image under Render-free-tier limits: peak memory and latency.

    python scripts/rss_probe.py --image scholar:onnx --label onnx-int8 \
        --database-url postgresql+psycopg://postgres:test@host.docker.internal:55432/scholar_probe

Runs the REAL image as a container capped like the target host —
`--memory=512m --memory-swap=512m --cpus=0.1` (no swap to hide behind; the CPU
quota as the worst case, since Render may let a free instance burst but nothing
promises it) — and drives it over HTTP the way a browser would:

  1. boot           container start -> /livez, then -> models warm (/health)
  2. index          the Attention paper (15 pages), upload -> job done
  3. ask x20        fixed questions; per-request wall time, p50 / p95
  4. index big      a synthetic MAX_PAGES-page dense PDF
  5. too big        a MAX_PAGES+1 page PDF — must fail fast, before parsing
  6. cold ask       after a container restart (process memory gone)

Generation is answered by a tiny fake OpenAI-compatible server on the host,
so /ask time is OUR CPU (retrieval, both models, reranking) — not Groq's
network latency, which no setting here can change.

MEMORY is sampled from `docker stats` (cgroup accounting via the daemon — a
`docker exec` would spend the container's own 0.1 CPU and skew the timings),
and the exact peak RSS is read from the uvicorn process's VmHWM.
An OOM kill under the 512 MB cap fails the run outright.

Writes docs/deploy-measurements/probe-<label>.json and prints a summary.
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "deploy-measurements"
PAPER = ROOT / "refferences" / "attention_is_all_you_need.pdf"
NAME = "scholar-probe"
PORT = 18001
LLM_PORT = 18999

QUESTIONS = [
    "What is the Transformer architecture based on?",
    "How many layers does the encoder stack have?",
    "What is scaled dot-product attention?",
    "Why is the dot product scaled by the square root of d_k?",
    "What is multi-head attention?",
    "How many attention heads are used in the base model?",
    "What positional encoding is used?",
    "What optimizer was used for training?",
    "What learning rate schedule was used?",
    "What BLEU score did the big model achieve on English-to-German?",
    "What BLEU score was achieved on English-to-French?",
    "How long did training take on the eight P100 GPUs?",
    "What regularization techniques were used?",
    "What is the value of label smoothing?",
    "What is the dimensionality of the model, d_model?",
    "What is the inner dimension of the feed-forward network?",
    "How does self-attention compare to recurrent layers in complexity?",
    "What dataset was used for English-German translation?",
    "How was English constituency parsing evaluated?",
    "What is the role of the decoder's masking?",
]


# ——— a fake LLM, so /ask measures retrieval, not a provider ———


class _FakeLLM(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass

    def _json(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # /v1/models
        self._json({"object": "list", "data": [{"id": "fake", "object": "model", "owned_by": "probe"}]})

    def do_POST(self):  # /v1/chat/completions (non-streaming)
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._json({
            "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Probe answer [1]."}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


def _start_fake_llm() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", LLM_PORT), _FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ——— docker ———


def _docker(*args: str, check: bool = True) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _mib(text: str) -> float:
    value = text.strip()
    for unit, factor in (("GiB", 1024), ("MiB", 1), ("KiB", 1 / 1024), ("B", 1 / 1048576)):
        if value.endswith(unit):
            return float(value[: -len(unit)]) * factor
    return float("nan")


class MemorySampler:
    """Polls `docker stats` in the background; records the max per phase."""

    def __init__(self) -> None:
        self.phase = "boot"
        self.peaks: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            out = _docker("stats", "--no-stream", "--format", "{{.MemUsage}}", NAME, check=False)
            if out:
                used = _mib(out.split("/")[0])
                self.peaks[self.phase] = max(self.peaks.get(self.phase, 0.0), used)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)


def _oom_killed() -> bool:
    return _docker("inspect", "-f", "{{.State.OOMKilled}}", NAME, check=False) == "true"


def _proc_peak() -> dict[str, float]:
    """Exact peak RSS (VmHWM) of the uvicorn process.

    Not PID 1: the image's shell-form CMD leaves `sh -c` as PID 1 with uvicorn
    as its child, and PID 1's numbers are the shell's ~1 MB. This kernel (the
    Docker Desktop VM) exposes no cgroup memory.peak, so VmHWM is the number.
    """
    script = "\n".join([
        "import glob, os",
        "for p in glob.glob('/proc/[0-9]*'):",
        "    try:",
        # This one-liner's own command line contains the search string, so
        # skip itself — otherwise it reports its own ~14 MB.
        "        if p.endswith('/' + str(os.getpid())): continue",
        "        if b'bin/uvicorn' in open(p + '/cmdline', 'rb').read():",
        "            print(open(p + '/status').read())",
        "    except OSError:",
        "        pass",
    ])
    status = _docker("exec", NAME, "python", "-c", script, check=False)
    found = {}
    for line in status.splitlines():
        key, _, rest = line.partition(":")
        if key in ("VmHWM", "VmRSS", "RssAnon") and rest.split():
            found[key + "_mib"] = round(int(rest.split()[0]) / 1024, 1)
    return found


# ——— the app ———


def _wait(client: httpx.Client, path: str, timeout: float, ok=lambda r: r.status_code == 200) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            if ok(client.get(path)):
                return time.perf_counter() - start
        except httpx.HTTPError:
            pass
        if _oom_killed():
            raise SystemExit("OOM-KILLED while waiting for " + path)
        time.sleep(1)
    raise SystemExit(f"timed out after {timeout}s waiting for {path}")


def _synthetic_pdf(pages: int, path: Path) -> None:
    import fitz

    paragraph = (
        "The committee reviewed the quarterly operating results, the capital plan and "
        "the risk register, and recorded its findings on liquidity, supplier concentration, "
        "regulatory change and the renewal terms of the master services agreement. "
    )
    doc = fitz.open()
    for n in range(pages):
        page = doc.new_page()
        page.insert_textbox(page.rect + (50, 50, -50, -50),
                            f"Section {n + 1}. " + paragraph * 9, fontsize=10)
    doc.save(str(path))
    doc.close()


def _upload_and_wait(client: httpx.Client, path: Path, timeout: float) -> dict:
    start = time.perf_counter()
    with path.open("rb") as fh:
        r = client.post("/papers", files={"file": (path.name, fh, "application/pdf")},
                        headers={"X-CSRF-Token": client.cookies.get("csrf_token", "")}, timeout=timeout)
    returned = time.perf_counter() - start
    if r.status_code >= 400:
        return {"status": f"http {r.status_code}", "detail": r.text[:200], "seconds": returned}
    job = r.json()
    while job["status"] in ("queued", "running"):
        if time.perf_counter() - start > timeout:
            return {"status": "timeout", "seconds": time.perf_counter() - start}
        if _oom_killed():
            return {"status": "OOM-KILLED", "seconds": time.perf_counter() - start}
        time.sleep(2)
        job = client.get(f"/papers/jobs/{job['id']}").json()
    return {"status": job["status"], "error": job.get("error"), "n_chunks": job.get("n_chunks"),
            "upload_returned_s": round(returned, 2), "seconds": round(time.perf_counter() - start, 2)}


def _ask_all(client: httpx.Client, questions: list[str]) -> dict:
    times = []
    for q in questions:
        start = time.perf_counter()
        r = client.post("/ask", json={"question": q},
                        headers={"X-CSRF-Token": client.cookies.get("csrf_token", "")}, timeout=300)
        times.append(time.perf_counter() - start)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    ordered = sorted(times)
    return {"n": len(times), "p50_s": round(statistics.median(times), 2),
            "p95_s": round(ordered[max(0, int(round(0.95 * len(ordered))) - 1)], 2),
            "max_s": round(max(times), 2), "first_s": round(times[0], 2)}


def main() -> int:
    p = argparse.ArgumentParser(prog="python scripts/rss_probe.py")
    p.add_argument("--image", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--database-url", required=True,
                   help="reachable FROM the container, e.g. host.docker.internal")
    p.add_argument("--memory", default="512m")
    p.add_argument("--cpus", default="0.1")
    p.add_argument("--max-pages", type=int, default=75)
    p.add_argument("--skip-big", action="store_true", help="skip the MAX_PAGES document")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra/overriding container environment (repeatable)")
    p.add_argument("--questions", type=int, default=len(QUESTIONS),
                   help="how many of the fixed questions to ask (each can take a while at 0.1 CPU)")
    args = p.parse_args()

    llm = _start_fake_llm()
    _docker("rm", "-f", NAME, check=False)
    env = {
        "SECRET_KEY": "probe-only", "DATABASE_URL": args.database_url,
        "STORAGE_BACKEND": "local", "DATA_ROOT": "/tmp/probe-data",
        "LLM_PROVIDER": "hosted", "LLM_BASE_URL": f"http://host.docker.internal:{LLM_PORT}/v1",
        "LLM_MODEL": "fake", "LLM_API_KEY": "fake", "QUERY_EXPANSION": "none",
        "MAX_UPLOAD_MB": "10", "MAX_PAGES": str(args.max_pages), "MAX_CHUNKS_PER_DOC": "400",
        "RETRIEVER_CACHE_SIZE": "2", "INDEX_IN_BACKGROUND": "true", "OCR_ENABLED": "false",
        "COOKIE_SECURE": "false", "ASK_RATE_LIMIT_PER_HOUR": "1000",
    }
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    flags = [f"--memory={args.memory}", f"--memory-swap={args.memory}", f"--cpus={args.cpus}",
             "-p", f"{PORT}:8001", "--add-host", "host.docker.internal:host-gateway"]
    for k, v in env.items():
        flags += ["-e", f"{k}={v}"]

    result: dict = {"image": args.image, "limits": {"memory": args.memory, "cpus": args.cpus},
                    "max_pages": args.max_pages, "env_overrides": args.env}
    sampler = MemorySampler()
    failure = None
    try:
        t0 = time.perf_counter()
        _docker("run", "-d", "--name", NAME, *flags, args.image)
        sampler.start()
        client = httpx.Client(base_url=f"http://localhost:{PORT}", timeout=120)
        result["boot_to_listening_s"] = round(_wait(client, "/healthz", 600), 1)
        _wait(client, "/health", 900)          # /health embeds a probe string: models loaded
        result["boot_to_models_ready_s"] = round(time.perf_counter() - t0, 1)

        r = client.post("/auth/register", json={"email": "probe@example.com", "password": "probe-password-123"})
        assert r.status_code == 201, r.text

        sampler.phase = "index_attention"
        result["index_attention"] = _upload_and_wait(client, PAPER, 3600)

        sampler.phase = "ask"
        result["ask_warm"] = _ask_all(client, QUESTIONS[: args.questions])

        with tempfile.TemporaryDirectory() as tmp:
            if not args.skip_big:
                sampler.phase = "index_max_pages"
                big = Path(tmp) / f"synthetic_{args.max_pages}p.pdf"
                _synthetic_pdf(args.max_pages, big)
                result["index_max_pages"] = _upload_and_wait(client, big, 3600)
            sampler.phase = "reject_over_limit"
            over = Path(tmp) / f"synthetic_{args.max_pages + 1}p.pdf"
            _synthetic_pdf(args.max_pages + 1, over)
            result["reject_over_limit"] = _upload_and_wait(client, over, 600)

        result["peak_before_restart"] = _proc_peak()

        sampler.phase = "cold_ask"
        _docker("restart", NAME)
        _wait(client, "/healthz", 600)
        client.post("/auth/login", json={"email": "probe@example.com", "password": "probe-password-123"})
        result["ask_cold"] = _ask_all(client, QUESTIONS[:3])
        result["peak_after_cold_ask"] = _proc_peak()
    except (SystemExit, Exception) as exc:      # record what was measured, then fail
        failure = str(exc) or type(exc).__name__
        result["failed"] = failure
    finally:
        sampler.stop()
        result["docker_stats_peak_mib_by_phase"] = {k: round(v, 1) for k, v in sampler.peaks.items()}
        result["oom_killed"] = _oom_killed()
        result["logs_tail"] = _docker("logs", "--tail", "15", NAME, check=False)
        _docker("rm", "-f", NAME, check=False)
        llm.shutdown()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"probe-{args.label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "logs_tail"}, indent=2))
    return 1 if (result["oom_killed"] or failure) else 0


if __name__ == "__main__":
    sys.exit(main())
