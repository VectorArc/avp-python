"""AVP HTTP/2 transport — client and FastAPI server factory."""

import json
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx
import numpy as np

from .codec import decode, encode
from .errors import HandshakeError, TransportError
from .fallback import JSONMessage
from .handshake import CompatibilityResolver, HelloMessage
from .types import (
    CONTENT_TYPE,
    AVP_VERSION_HEADER,
    AVPMessage,
    AVPMetadata,
    CommunicationMode,
    CompressionLevel,
    ModelIdentity,
    SessionInfo,
)


class AVPClient:
    """HTTP/2 client for sending AVP binary and JSON messages."""

    def __init__(
        self,
        base_url: str,
        agent_id: str = "default",
        model_identity: Optional[ModelIdentity] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.model_identity = model_identity
        self.token = token
        self._client = httpx.Client(
            http2=True,
            timeout=timeout,
            base_url=self.base_url,
        )
        self._session_info: Optional[SessionInfo] = None

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": CONTENT_TYPE,
            "AVP-Version": AVP_VERSION_HEADER,
            "AVP-Agent-ID": self.agent_id,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def handshake(self) -> SessionInfo:
        """Perform handshake with remote agent to negotiate communication mode.

        Returns:
            SessionInfo with negotiated mode and session_id.
        """
        if self.model_identity is None:
            raise HandshakeError("model_identity required for handshake")

        hello = HelloMessage(
            agent_id=self.agent_id,
            identity=self.model_identity,
        )

        headers = {
            "Content-Type": "application/json",
            "AVP-Version": AVP_VERSION_HEADER,
            "AVP-Agent-ID": self.agent_id,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = self._client.post(
            "/avp/v2/handshake",
            content=json.dumps(hello.to_dict()).encode(),
            headers=headers,
        )

        if resp.status_code >= 400:
            raise HandshakeError(f"Handshake failed: {resp.status_code} {resp.text}")

        data = resp.json()
        remote_identity = ModelIdentity.from_dict(data.get("identity", {}))

        session_info = CompatibilityResolver.resolve(self.model_identity, remote_identity)
        # Use server-provided session_id if available
        if "session_id" in data:
            session_info.session_id = data["session_id"]

        self._session_info = session_info
        return session_info

    def transmit(
        self,
        payload: bytes,
        metadata: AVPMetadata,
        compression: CompressionLevel = CompressionLevel.NONE,
    ) -> httpx.Response:
        """Encode and send a binary payload to the remote agent.

        Returns the raw httpx.Response.
        """
        data = encode(payload, metadata, compression)

        resp = self._client.post(
            "/avp/v2/transmit", content=data, headers=self._headers()
        )

        if resp.status_code >= 400:
            raise TransportError(
                f"Transmit failed: {resp.status_code} {resp.text}",
                status_code=resp.status_code,
            )
        return resp

    def send_text(self, message: JSONMessage) -> httpx.Response:
        """Send a JSON fallback message.

        Args:
            message: JSONMessage to send.

        Returns:
            httpx.Response from the server.
        """
        headers = {
            "Content-Type": "application/json",
            "AVP-Version": AVP_VERSION_HEADER,
            "AVP-Agent-ID": self.agent_id,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = self._client.post(
            "/avp/v2/text",
            content=message.to_json().encode(),
            headers=headers,
        )

        if resp.status_code >= 400:
            raise TransportError(
                f"Text send failed: {resp.status_code} {resp.text}",
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
    """Async HTTP/2 client for sending AVP binary and JSON messages."""

    def __init__(
        self,
        base_url: str,
        agent_id: str = "default",
        model_identity: Optional[ModelIdentity] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.model_identity = model_identity
        self.token = token
        self._client = httpx.AsyncClient(
            http2=True,
            timeout=timeout,
            base_url=self.base_url,
        )

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": CONTENT_TYPE,
            "AVP-Version": AVP_VERSION_HEADER,
            "AVP-Agent-ID": self.agent_id,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def transmit(
        self,
        payload: bytes,
        metadata: AVPMetadata,
        compression: CompressionLevel = CompressionLevel.NONE,
    ) -> httpx.Response:
        data = encode(payload, metadata, compression)

        resp = await self._client.post(
            "/avp/v2/transmit", content=data, headers=self._headers()
        )

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
TextHandler = Callable[[JSONMessage], Awaitable[dict]]
HandshakeHandler = Callable[[HelloMessage], Awaitable[dict]]


def create_app(
    handler: AVPHandler,
    text_handler: Optional[TextHandler] = None,
    handshake_handler: Optional[HandshakeHandler] = None,
    **fastapi_kwargs: Any,
) -> Any:
    """Create a FastAPI app with AVP endpoints.

    Endpoints:
        POST /avp/v2/handshake — Handshake negotiation
        POST /avp/v2/transmit — Binary payload transfer
        POST /avp/v2/text — JSON fallback messages
        GET /health — Health check

    Args:
        handler: Async function (AVPMessage) -> dict for binary messages.
        text_handler: Async function (JSONMessage) -> dict for text messages.
        handshake_handler: Async function (HelloMessage) -> dict for handshake.
        **fastapi_kwargs: Extra kwargs passed to FastAPI().

    Returns:
        A FastAPI app instance.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="AVP Agent", **fastapi_kwargs)

    # --- v0.2.0 endpoints ---

    @app.post("/avp/v2/handshake")
    async def v2_handshake(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            hello = HelloMessage.from_dict(body)
        except Exception as exc:
            return JSONResponse(
                {"success": False, "message": f"Invalid handshake: {exc}"},
                status_code=400,
            )

        if handshake_handler:
            result = await handshake_handler(hello)
            return JSONResponse({"success": True, **result})

        # Default: echo back identity for client-side resolution
        return JSONResponse({
            "success": True,
            "agent_id": hello.agent_id,
            "identity": hello.identity.to_dict() if hello.identity else {},
        })

    @app.post("/avp/v2/transmit")
    async def v2_transmit(request: Request) -> JSONResponse:
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

    @app.post("/avp/v2/text")
    async def v2_text(request: Request) -> JSONResponse:
        if text_handler is None:
            return JSONResponse(
                {"success": False, "message": "Text handler not configured"},
                status_code=501,
            )
        try:
            body = await request.json()
            msg = JSONMessage.from_dict(body)
        except Exception as exc:
            return JSONResponse(
                {"success": False, "message": f"Invalid message: {exc}"},
                status_code=400,
            )

        result = await text_handler(msg)
        return JSONResponse({"success": True, **result})

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app
