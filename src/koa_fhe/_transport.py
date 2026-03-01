"""HTTP transport layer for Koa FHE service. stdlib only — no third-party deps."""

from __future__ import annotations

import hashlib
import json
from http.client import HTTPConnection
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class Transport:
    """Low-level HTTP transport for the FHE service."""

    def __init__(self, server_url: str, client_id: str) -> None:
        self._base = server_url.rstrip("/")
        self._client_id = client_id

    def set_client_id(self, client_id: str) -> None:
        """Update client_id (used when adopting server's ID after key dedup)."""
        self._client_id = client_id

    def get(self, path: str) -> dict:
        url = f"{self._base}/{path.lstrip('/')}"
        req = Request(url)
        with urlopen(req) as resp:
            return json.loads(resp.read())

    def post_json(self, path: str, data: dict, headers: dict | None = None) -> dict:
        url = f"{self._base}/{path.lstrip('/')}"
        body = json.dumps(data).encode()
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Client-Id", self._client_id)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        with urlopen(req) as resp:
            return json.loads(resp.read())

    def head_key(self, circuit_name: str, key_hash: str) -> str | None:
        """Check if a key with the given hash exists on the server.

        Returns the existing client_id if found, None otherwise.
        """
        parsed = urlparse(f"{self._base}/keys/{circuit_name}?hash={key_hash}")
        conn = HTTPConnection(parsed.hostname, parsed.port)
        path_with_query = parsed.path
        if parsed.query:
            path_with_query += f"?{parsed.query}"
        conn.request("HEAD", path_with_query)
        resp = conn.getresponse()
        resp.read()  # drain body
        existing_cid = resp.getheader("X-Client-Id") if resp.status == 200 else None
        conn.close()
        return existing_cid

    def post_binary(self, path: str, data_bytes: bytes) -> dict:
        """Stream large binary payloads with 8 MB chunking (macOS sendall limit).

        Sends X-Content-Hash header (SHA-256 truncated to 16 hex) for
        server-side integrity verification.
        """
        content_hash = hashlib.sha256(data_bytes).hexdigest()[:16]
        parsed = urlparse(f"{self._base}/{path.lstrip('/')}")
        conn = HTTPConnection(parsed.hostname, parsed.port)
        conn.putrequest("POST", parsed.path)
        conn.putheader("Content-Type", "application/octet-stream")
        conn.putheader("X-Client-Id", self._client_id)
        conn.putheader("X-Content-Hash", content_hash)
        conn.putheader("Content-Length", str(len(data_bytes)))
        conn.endheaders()
        chunk_size = 8 * 1024 * 1024
        for i in range(0, len(data_bytes), chunk_size):
            conn.send(data_bytes[i : i + chunk_size])
        resp = conn.getresponse()
        resp_body = resp.read()
        conn.close()
        return json.loads(resp_body)
