"""Where library files are kept: a local directory, or a Supabase Storage bucket.

Keys are relative paths shared by both backends:

    {workspace_id}/papers/{paper_id}{suffix}        an uploaded original (write-once)
    {workspace_id}/index/{version}/{file}           one immutable index version

The interface is deliberately tiny — put a file, fetch a file, list, delete —
because that is all object storage reliably offers. In particular there is NO
conditional write: two writers to one key simply race, last one wins. Nothing
here therefore ever overwrites a key that a reader might be using; index
versions get fresh keys, and the database decides which version is current
(see db_models.WorkspaceIndex and library.py).
"""
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import quote

import httpx


class BlobStoreError(RuntimeError):
    """A storage operation failed in a way the caller should not paper over."""


class BlobStore(ABC):
    @abstractmethod
    def put(self, key: str, source: Path) -> None:
        """Store `source` at `key`, replacing anything there."""

    @abstractmethod
    def get_to_file(self, key: str, dest: Path) -> bool:
        """Copy `key` to `dest`. False if no such key; raises on other failures."""

    @abstractmethod
    def list_keys(self, prefix: str) -> dict[str, int]:
        """Every key under `prefix` (recursively), mapped to its size in bytes."""

    @abstractmethod
    def delete(self, keys: list[str]) -> None:
        """Delete these keys. Missing keys are not an error."""

    def delete_prefix(self, prefix: str) -> None:
        keys = list(self.list_keys(prefix))
        if keys:
            self.delete(keys)


class LocalBlobStore(BlobStore):
    """A directory. Development, tests, and any host with a real disk."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise BlobStoreError(f"key escapes the store: {key!r}")
        return path

    def put(self, key: str, source: Path) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy beside the target, then rename: a reader (or a crash) never
        # sees a half-written file under the real name.
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".put-")
        os.close(fd)
        try:
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def get_to_file(self, key: str, dest: Path) -> bool:
        path = self._path(key)
        if not path.is_file():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        return True

    def list_keys(self, prefix: str) -> dict[str, int]:
        base = self._path(prefix) if prefix else self.root.resolve()
        # A prefix may end mid-name ("7/papers/report."), so list its parent
        # directory and filter, the way object storage matches prefixes.
        directory = base if base.is_dir() else base.parent
        if not directory.exists():
            return {}
        root = self.root.resolve()
        found = {}
        for path in directory.rglob("*"):
            if path.is_file() and not path.name.startswith(".put-"):
                key = path.relative_to(root).as_posix()
                if key.startswith(prefix):
                    found[key] = path.stat().st_size
        return found

    def delete(self, keys: list[str]) -> None:
        for key in keys:
            self._path(key).unlink(missing_ok=True)
        # Tidy directories the deletions emptied, so a deleted workspace does
        # not leave a skeleton of empty folders behind.
        for key in keys:
            parent = self._path(key).parent
            while parent != self.root.resolve() and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent


class SupabaseBlobStore(BlobStore):
    """A PRIVATE Supabase Storage bucket, over its REST API.

    Plain httpx rather than supabase-py: four endpoints do not justify a client
    library and its dependency tree in a 512 MB container. The service-role key
    is used because the server does its own authorisation (workspace
    membership) before any of these calls; the bucket itself is never exposed
    to a browser — downloads are streamed through the API.
    """

    LIST_PAGE = 1000

    def __init__(self, url: str, service_key: str, bucket: str,
                 client: httpx.Client | None = None) -> None:
        if not url or not service_key:
            raise BlobStoreError("STORAGE_BACKEND=supabase needs SUPABASE_URL and SUPABASE_SERVICE_KEY")
        self.bucket = bucket
        self.client = client or httpx.Client(
            base_url=f"{url.rstrip('/')}/storage/v1",
            headers={"Authorization": f"Bearer {service_key}", "apikey": service_key},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def _object(self, key: str) -> str:
        return f"/object/{self.bucket}/{quote(key)}"

    @staticmethod
    def _check(response: httpx.Response, action: str) -> None:
        if response.status_code >= 400:
            raise BlobStoreError(f"storage {action} failed: HTTP {response.status_code} {response.text[:200]}")

    def put(self, key: str, source: Path) -> None:
        # Streamed from disk, not read into memory: an upload can be the largest
        # thing the process handles, and the budget is 512 MB for everything.
        with source.open("rb") as body:
            response = self.client.post(
                self._object(key),
                content=body,
                headers={"x-upsert": "true", "Content-Type": "application/octet-stream",
                         "Content-Length": str(source.stat().st_size)},
            )
        self._check(response, f"upload of {key}")

    def get_to_file(self, key: str, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        with self.client.stream("GET", f"/object/authenticated/{self.bucket}/{quote(key)}") as response:
            # Supabase answers a missing object with 400 or 404 depending on
            # the version; both mean "not there".
            if response.status_code in (400, 404):
                return False
            self._check(response, f"download of {key}")
            with partial.open("wb") as out:
                for block in response.iter_bytes(1 << 20):
                    out.write(block)
        os.replace(partial, dest)
        return True

    def list_keys(self, prefix: str) -> dict[str, int]:
        """Storage lists one folder level at a time, so walk the tree."""
        found: dict[str, int] = {}
        folder, _, partial = prefix.rpartition("/")
        pending = [(folder, partial)]
        while pending:
            folder, name_prefix = pending.pop()
            offset = 0
            while True:
                response = self.client.post(f"/object/list/{self.bucket}", json={
                    "prefix": f"{folder}/" if folder else "",
                    "search": name_prefix,
                    "limit": self.LIST_PAGE, "offset": offset,
                })
                self._check(response, f"list of {folder}/")
                entries = response.json()
                for entry in entries:
                    key = f"{folder}/{entry['name']}" if folder else entry["name"]
                    if entry.get("id") is None:          # a folder, not a file
                        pending.append((key, ""))
                    elif key.startswith(prefix):
                        found[key] = int((entry.get("metadata") or {}).get("size", 0))
                if len(entries) < self.LIST_PAGE:
                    break
                offset += self.LIST_PAGE
        return found

    def delete(self, keys: list[str]) -> None:
        for start in range(0, len(keys), 1000):
            response = self.client.request(
                "DELETE", f"/object/{self.bucket}", json={"prefixes": keys[start:start + 1000]}
            )
            self._check(response, "delete")


def from_settings() -> BlobStore:
    from backend import library
    from backend.config import settings

    if settings.storage_backend == "supabase":
        return SupabaseBlobStore(settings.supabase_url, settings.supabase_service_key,
                                 settings.supabase_bucket)
    if settings.storage_backend != "local":
        raise BlobStoreError(f"STORAGE_BACKEND must be 'local' or 'supabase', got {settings.storage_backend!r}")
    return LocalBlobStore(library.DATA_ROOT)
