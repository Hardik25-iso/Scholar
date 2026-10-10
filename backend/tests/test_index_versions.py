"""The versioned index protocol: what happens when writes crash or collide.

An index change is published as a complete new version and made current by one
compare-and-swap on the WorkspaceIndex row (see library.py). These tests break
that sequence on purpose, at the points object storage actually fails:

  * an upload dying part-way           -> nothing changes for readers
  * a second writer switching first    -> neither change is lost
  * a newer version going live         -> the previous one is still readable
  * a library from before versioning   -> adopted in place, then replaced

Synthetic vectors stand in for embeddings, so no model is loaded: the protocol
under test does not care what the numbers mean.
"""
import json
import threading

import numpy as np
import pytest
from sqlmodel import Session

from backend import audit, lexical, library
from backend.blobstore import LocalBlobStore
from backend.db import engine
from backend.db_models import User, Workspace, WorkspaceIndex
from backend.models import Chunk
from backend.store import append_to_store, load


@pytest.fixture
def ws(client) -> int:
    """A real workspace row (Postgres enforces the foreign key)."""
    with Session(engine) as session:
        user = User(email="w@example.com", hashed_password="x")
        session.add(user)
        session.commit()
        workspace = Workspace(name="w", is_personal=True)
        session.add(workspace)
        session.commit()
        return workspace.id


def _chunks(paper_id: str, n: int = 3) -> list[Chunk]:
    return [Chunk(paper_id=paper_id, page=0, chunk_index=i, text=f"{paper_id} passage {i}",
                  embed_text=f"{paper_id} passage {i}", char_start=0, char_end=10)
            for i in range(n)]


def _vectors(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, 768)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _adder(paper_id: str, seed: int = 0):
    chunks = _chunks(paper_id)
    vectors = _vectors(len(chunks), seed)

    def add(work):
        append_to_store(chunks, vectors, work)
        lexical.index_chunks(chunks, work)
    return add


def _papers_in(ws: int) -> set[str]:
    _, chunks = load(library.current_index_dir(ws))
    return {c.paper_id for c in chunks}


def _stored_versions(ws: int) -> set[str]:
    keys = LocalBlobStore(library.DATA_ROOT).list_keys(f"{ws}/index/")
    return {k.split("/")[2] for k in keys if k.count("/") >= 3}


def test_a_publish_makes_a_complete_version_current(ws):
    assert library.current_version(ws) is None
    assert library._publish(ws, _adder("alpha")) == 3
    version, fingerprint, n = library.current_version(ws)
    assert n == 3 and version
    # The recorded fingerprint is exactly what audit computes from the files,
    # so answers logged before versioning still compare correctly.
    assert fingerprint == audit.index_fingerprint(library.current_index_dir(ws))[0]
    assert _stored_versions(ws) == {version}


def test_an_upload_that_dies_part_way_changes_nothing(ws, monkeypatch):
    library._publish(ws, _adder("alpha"))
    before = library.current_version(ws)

    real_put = LocalBlobStore.put
    calls = {"n": 0}

    def flaky_put(self, key, source):
        calls["n"] += 1
        if calls["n"] == 2:                      # the second of three files
            raise OSError("connection reset by peer")
        return real_put(self, key, source)

    monkeypatch.setattr(LocalBlobStore, "put", flaky_put)
    with pytest.raises(OSError):
        library._publish(ws, _adder("beta", seed=1))
    monkeypatch.setattr(LocalBlobStore, "put", real_put)

    # Readers see exactly the old library — not a mix, not an empty one.
    assert library.current_version(ws) == before
    library.invalidate(ws)
    assert _papers_in(ws) == {"alpha"}
    # The dead write left an orphan version behind...
    assert len(_stored_versions(ws)) == 2

    # ...which the next successful write sweeps away.
    library._publish(ws, _adder("gamma", seed=2))
    assert _papers_in(ws) == {"alpha", "gamma"}
    current = library.current_version(ws)[0]
    assert _stored_versions(ws) == {current, before[0]}   # current + previous only


def test_a_concurrent_writer_that_switches_first_loses_nothing(ws, monkeypatch):
    """Two processes on SQLite share no lock (an API and an arq worker), so the
    version switch itself must detect the race. Simulated by disabling the
    lock and letting a second writer publish in the middle of the first."""
    library._publish(ws, _adder("base"))

    from contextlib import nullcontext
    monkeypatch.setattr(library, "_workspace_lock", lambda session, workspace_id: nullcontext())

    interfered = {"done": False}
    add_a = _adder("writer_a", seed=3)

    def add_a_while_b_publishes(work):
        if not interfered["done"]:
            interfered["done"] = True
            library._publish(ws, _adder("writer_b", seed=4))   # B wins the switch
        add_a(work)

    library._publish(ws, add_a_while_b_publishes)
    # A's first attempt was built on the version B replaced; its compare-and-swap
    # failed, and it was re-applied on top of B's version.
    assert _papers_in(ws) == {"base", "writer_a", "writer_b"}


def test_parallel_publishes_in_one_process_are_serialised(ws):
    library._publish(ws, _adder("base"))
    threads = [threading.Thread(target=library._publish, args=(ws, _adder(f"t{i}", seed=10 + i)))
               for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _papers_in(ws) == {"base", "t0", "t1", "t2", "t3"}


def test_the_previous_version_stays_readable_after_a_switch(ws):
    """A request that loaded version N may still be running lexical search on
    it when N+1 goes live; its files must not vanish underneath it."""
    library._publish(ws, _adder("alpha"))
    old_version, old_fp, _ = library.current_version(ws)
    old_dir = library.local_index_dir(ws, old_version)

    library._publish(ws, _adder("beta", seed=1))
    assert old_dir.exists()
    assert audit.index_fingerprint(old_dir)[0] == old_fp
    assert lexical.search("alpha passage", old_dir)


def test_retriever_follows_the_current_version(ws):
    library._publish(ws, _adder("alpha"))
    first = library.get_retriever(ws)
    assert library.get_retriever(ws) is first            # cached while current
    library._publish(ws, _adder("beta", seed=1))
    second = library.get_retriever(ws)
    assert second is not first
    assert {c.paper_id for c in second.chunks} == {"alpha", "beta"}


def test_a_pre_versioning_library_is_adopted_then_replaced(ws):
    legacy = library.DATA_ROOT / str(ws) / "index"
    chunks = _chunks("old")
    append_to_store(chunks, _vectors(3, 7), legacy)
    lexical.index_chunks(chunks, legacy)
    expected_fp = audit.index_fingerprint(legacy)[0]

    version, fingerprint, n = library.current_version(ws)
    assert (version, fingerprint, n) == ("", expected_fp, 3)
    assert _papers_in(ws) == {"old"}

    library._publish(ws, _adder("new", seed=8))
    assert _papers_in(ws) == {"old", "new"}
    assert library.current_version(ws)[0] != ""
    # The adopted files are now just "the previous version" and are kept as such.
    assert (legacy / "chunks.json").exists()


def test_deleting_the_last_document_empties_the_library(ws):
    library._publish(ws, _adder("only"))
    library.delete_paper_data(ws, "only")
    assert library.current_version(ws) is None
    assert library.get_retriever(ws) is None
    assert _stored_versions(ws) == set()
    with Session(engine) as session:
        assert session.get(WorkspaceIndex, ws) is None


def test_chunks_json_bytes_define_the_fingerprint(ws):
    """Guard on the audit contract: the fingerprint is sha256 of chunks.json,
    and versioning must not change how chunks.json is written."""
    library._publish(ws, _adder("alpha"))
    raw = (library.current_index_dir(ws) / "chunks.json").read_bytes()
    assert json.loads(raw)[0]["paper_id"] == "alpha"
    # indent=2, unchanged. (store.py writes in text mode, so line endings are
    # the platform's — that predates versioning and is not asserted here.)
    assert raw.replace(b"\r\n", b"\n").startswith(b"[\n  {")
