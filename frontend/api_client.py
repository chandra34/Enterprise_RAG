from pathlib import Path
from typing import Any

import httpx


class APIClientError(RuntimeError):
    pass


class BackendAPIClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
        except httpx.HTTPStatusError as exc:
            raise APIClientError(exc.response.text or str(exc)) from exc
        except httpx.HTTPError as exc:
            raise APIClientError(f"Failed to reach backend: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/health").json()

    def upload_pdf(self, file_path: Path) -> dict[str, Any]:
        with file_path.open("rb") as handle:
            files = {"file": (file_path.name, handle, "application/pdf")}
            return self._request("POST", "/api/v1/upload", files=files).json()

    def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question}
        if top_k is not None:
            payload["top_k"] = top_k
        return self._request("POST", "/api/v1/query", json=payload).json()
