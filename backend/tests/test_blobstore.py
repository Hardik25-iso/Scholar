"""SupabaseBlobStore against a faithful in-memory fake of the Storage REST API.

The fake reproduces the behaviours the client depends on — and the ones that
would bite if it did not handle them:
  * listing is ONE folder level per call, folders have `id: null`, and results
    are paged (so a large workspace needs several calls);
  * a missing object answers 400 (some versions) or 404;
  * deletion takes a list of keys in a JSON body.

Then the library's whole publish/read/delete cycle runs against it, so the
versioned protocol is exercised over the remote backend, not just the local one.
"""
import json
from urllib.parse import unquote

import httpx
import numpy as np
import pytest

from backend import lexical, library
from backend.blobstore import BlobStoreError, SupabaseBlobStore
from backend.models import Chunk
from backend.store import append_to_store

BUCKET = "scholar"


class FakeStorage:
    def __init__(self, missing_status: int = 400, page: int = 1000):
        self.objects: dict[str, bytes] = {}
        self.missing_status = missing_status
        self.page = page
        self.list_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = unquote(request.url.path)
        assert request.headers["authorization"] == "Bearer service-key"
        if request.method == "POST" and path.startswith(f"/storage/v1/object/list/{BUCKET}"):
            return self._list(json.loads(request.content))
        if request.method == "POST" and path.startswith(f"/storage/v1/object/{BUCKET}/"):
            assert request.headers.get("x-upsert") == "true"
            self.objects[path.removeprefix(f"/storage/v1/object/{BUCKET}/")] = request.read()
            return httpx.Response(200, json={"Key": path})
        if request.method == "GET" and path.startswith(f"/storage/v1/object/authenticated/{BUCKET}/"):
            key = path.removeprefix(f"/storage/v1/object/authenticated/{BUCKET}/")
            if key not in self.objects:
                return httpx.Response(self.missing_status, json={"error": "not_found"})
            return httpx.Response(200, content=self.objects[key])
        if request.method == "DELETE" and path == f"/storage/v1/object/{BUCKET}":
            for key in json.loads(request.content)["prefixes"]:
                self.objects.pop(key, None)
            return httpx.Response(200, json=[])
        return httpx.Response(500, text=f"unexpected {request.method} {path}")

    def _list(self, body: dict) -> httpx.Response:
        self.list_calls += 1
        folder = body["prefix"]
        entries: dict[str, dict] = {}
        for key, data in self.objects.items():
            if not key.startswith(folder):
                continue
            rest = key[len(folder):]
            name = rest.split("/")[0]
            if not name.startswith(body.get("search", "")):
                continue
            if "/" in rest:
                entries[name] = {"name": name, "id": None, "metadata": None}
            else:
                entries[name] = {"name": name, "id": "x", "metadata": {"size": len(data)}}
        ordered = [entries[n] for n in sorted(entries)]
        start = body.get("offset", 0)
        return httpx.Response(200, json=ordered[start:start + body["limit"]])


def _store(fake: FakeStorage) -> SupabaseBlobStore:
    client = httpx.Client(base_url="https://proj.supabase.co/storage/v1",
                          headers={"Authorization": "Bearer service-key"},
                          transport=httpx.MockTransport(fake.handler))
    return SupabaseBlobStore("https://proj.supabase.co", "service-key", BUCKET, client=client)


def test_put_get_round_trip_and_missing_keys(tmp_path):
    for status in (400, 404):
        fake = FakeStorage(missing_status=status)
        store = _store(fake)
        src = tmp_path / "a.bin"
        src.write_bytes(b"hello")
        store.put("7/papers/report.pdf", src)
        out = tmp_path / "out" / "a.bin"
        assert store.get_to_file("7/papers/report.pdf", out)
        assert out.read_bytes() == b"hello"
        assert store.get_to_file("7/papers/nope.pdf", tmp_path / "nope") is False


def test_listing_walks_folders_and_pages(tmp_path):
    fake = FakeStorage(page=2)
    store = _store(fake)
    store.LIST_PAGE = 2
    src = tmp_path / "f"
    src.write_bytes(b"12345")
    keys = [f"7/index/000001-aa/{n}" for n in ("index.faiss", "chunks.json", "lexical.db")] + \
           [f"7/index/000002-bb/{n}" for n in ("index.faiss", "chunks.json")] + \
           ["7/papers/report.pdf", "7/papers/report_2.pdf", "8/papers/other.pdf"]
    for key in keys:
        store.put(key, src)

    assert set(store.list_keys("7/")) == {k for k in keys if k.startswith("7/")}
    assert store.list_keys("7/index/000001-aa/") == {
        f"7/index/000001-aa/{n}": 5 for n in ("index.faiss", "chunks.json", "lexical.db")
    }
    # A prefix ending mid-name matches like object storage does — and only that.
    assert set(store.list_keys("7/papers/report.")) == {"7/papers/report.pdf"}
    assert fake.list_calls > 3          # really did page and recurse


def test_delete_prefix_removes_only_that_workspace(tmp_path):
    store = _store(FakeStorage())
    src = tmp_path / "f"
    src.write_bytes(b"x")
    for key in ("7/papers/a.pdf", "7/index/v/chunks.json", "70/papers/b.pdf"):
        store.put(key, src)
    store.delete_prefix("7/")
    assert set(store.list_keys("")) == {"70/papers/b.pdf"}


def test_errors_are_raised_not_swallowed(tmp_path):
    def broken(request):
        return httpx.Response(503, text="storage unavailable")

    client = httpx.Client(base_url="https://x/storage/v1", transport=httpx.MockTransport(broken))
    store = SupabaseBlobStore("https://x", "k", BUCKET, client=client)
    src = tmp_path / "f"
    src.write_bytes(b"x")
    with pytest.raises(BlobStoreError, match="503"):
        store.put("1/papers/a.pdf", src)


def test_missing_configuration_fails_loudly():
    with pytest.raises(BlobStoreError, match="SUPABASE_URL"):
        SupabaseBlobStore("", "", BUCKET)


# ——— the library over the remote backend ———


@pytest.fixture
def remote(client, monkeypatch) -> FakeStorage:
    from backend.config import settings

    fake = FakeStorage(page=3)
    monkeypatch.setattr(settings, "storage_backend", "supabase")
    monkeypatch.setattr(library, "_supabase", _store(fake))
    return fake


def _add(paper_id: str, seed: int):
    chunks = [Chunk(paper_id=paper_id, page=0, chunk_index=i, text=f"{paper_id} text {i}",
                    embed_text=f"{paper_id} text {i}", char_start=0, char_end=5) for i in range(3)]
    v = np.random.default_rng(seed).standard_normal((3, 768)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    def change(work):
        append_to_store(chunks, v, work)
        lexical.index_chunks(chunks, work)
    return change


def test_publish_read_and_delete_over_supabase(remote, tmp_path):
    from sqlmodel import Session

    from backend.db import engine
    from backend.db_models import User, Workspace

    with Session(engine) as session:
        session.add(User(email="r@example.com", hashed_password="x"))
        workspace = Workspace(name="r", is_personal=True)
        session.add(workspace)
        session.commit()
        ws = workspace.id

    library._publish(ws, _add("alpha", 1))
    library._publish(ws, _add("beta", 2))
    versions = {k.split("/")[2] for k in remote.objects if k.startswith(f"{ws}/index/")}
    assert len(versions) == 2                      # current + previous, nothing else

    # A cold start: no local cache at all. Everything must come from the bucket.
    import shutil
    shutil.rmtree(library.CACHE_ROOT, ignore_errors=True)
    library._retrievers.clear()
    retriever = library.get_retriever(ws)
    assert {c.paper_id for c in retriever.chunks} == {"alpha", "beta"}
    assert lexical.search("beta text", retriever.store_dir)

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    library.store_paper(ws, "alpha", ".pdf", src)
    shutil.rmtree(library.CACHE_ROOT, ignore_errors=True)
    fetched = library.stored_path(ws, "alpha")
    assert fetched is not None and fetched.read_bytes() == b"%PDF-1.4 fake"

    library.delete_paper_data(ws, "alpha")
    library.delete_paper_data(ws, "beta")
    assert library.current_index(ws) is None
    assert not [k for k in remote.objects if k.startswith(f"{ws}/")]
