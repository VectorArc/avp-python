"""AVP HTTP/2 transport — client and FastAPI server factory."""

from typing import Any, Awaitable, Callable, Optional

import httpx
import numpy as np

from .codec import decode, encode
from .errors import TransportError
from .types import CONTENT_TYPE, AVP_VERSION_HEADER, AVPMessage, CompressionLevel


class AVPClient:
    """HTTP/2 client for sending AVP binary messages."""

    def __init__(
        self,
        base_url: str,
        agent_id: str = "default",
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.token = token
        self._client = httpx.Client(
            http2=True,
            timeout=timeout,
            base_url=self.base_url,
        )

    def transmit(
        self,
        embedding: np.ndarray,
        model_id: str = "",
        compression: CompressionLevel = CompressionLevel.NONE,
        **encode_kwargs: Any,
    ) -> httpx.Response:
        """Encode and send an embedding to the remote agent.

        Returns the raw httpx.Response (caller can inspect status, body, etc.).
        """
        payload = encode(
            embedding,
            model_id=model_id,
            compression=compression,
            agent_id=self.agent_id,
            **encode_kwargs,
        )

        headers = {
            "Content-Type": CONTENT_TYPE,
            "AVP-Version": AVP_VERSION_HEADER,
            "AVP-Agent-ID": self.agent_id,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = self._client.post("/avp/v1/transmit", content=payload, headers=headers)

        if resp.status_code >= 400:
            raise TransportError(
                f"Transmit failed: {resp.status_code} {resp.text}",
                status_code=resp.status_code,
            )
        return resp

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AVPClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AVPAsyncClient:
    """Async HTTP/2 client for sending AVP binary messages."""

    def __init__(
        self,
        base_url: str,
        agent_id: str = "default",
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.token = token
        self._client = httpx.AsyncClient(
            http2=True,
            timeout=timeout,
            base_url=self.base_url,
        )

    async def transmit(
        self,
        embedding: np.ndarray,
        model_id: str = "",
        compression: CompressionLevel = CompressionLevel.NONE,
        **encode_kwargs: Any,
    ) -> httpx.Response:
        payload = encode(
            embedding,
            model_id=model_id,
            compression=compression,
            agent_id=self.agent_id,
            **encode_kwargs,
        )

        headers = {
            "Content-Type": CONTENT_TYPE,
            "AVP-Version": AVP_VERSION_HEADER,
            "AVP-Agent-ID": self.agent_id,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = await self._client.post("/avp/v1/transmit", content=payload, headers=headers)

        if resp.status_code >= 400:
            raise TransportError(
                f"Transmit failed: {resp.status_code} {resp.text}",
                status_code=resp.status_code,
            )
        return resp

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AVPAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# --- FastAPI server factory ---

AVPHandler = Callable[[AVPMessage], Awaitable[dict]]


def create_app(handler: AVPHandler, **fastapi_kwargs: Any) -> Any:
    """Create a FastAPI app with a POST /avp/v1/transmit endpoint.

    Args:
        handler: Async function (AVPMessage) -> dict that processes incoming messages.
        **fastapi_kwargs: Extra kwargs passed to FastAPI().

    Returns:
        A FastAPI app instance.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="AVP Agent", **fastapi_kwargs)

    @app.post("/avp/v1/transmit")
    async def transmit(request: Request) -> JSONResponse:
        body = await request.body()
        try:
            msg = decode(body)
        except Exception as exc:
            return JSONResponse(
                {"success": False, "message": str(exc)},
                status_code=400,
            )

        result = await handler(msg)
        return JSONResponse({"success": True, **result})

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app
