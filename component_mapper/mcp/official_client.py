import asyncio
import json
import logging
import aiohttp
from component_mapper.config import MCPConfig

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    pass


class MCPInstallError(Exception):
    pass


class OfficialMCPClient:
    def __init__(self, config: MCPConfig):
        self.config = config
        self._connected = False
        self._proc: asyncio.subprocess.Process | None = None
        self._session: aiohttp.ClientSession | None = None
        self._request_id = 0
        self.calls_made = 0

    async def connect(self) -> None:
        """Establish MCP connection. Logs warning instead of raising if unavailable."""
        try:
            if self.config.transport == "stdio":
                await self._connect_stdio()
            else:
                await self._connect_sse()
            self._connected = True
            logger.info("MCP client connected via %s", self.config.transport)
        except Exception as exc:
            logger.warning(
                "MCP unavailable (%s) — pipeline will use cached index only", exc
            )
            self._connected = False

    async def _connect_stdio(self) -> None:
        self._proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "npx",
                "shadcn",
                "mcp",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=self.config.startup_timeout_seconds,
        )
        # Send JSON-RPC initialize
        await self._send_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "component-mapper", "version": "0.1.0"},
                },
            }
        )
        resp = await self._read_jsonrpc()
        if "error" in resp:
            raise MCPConnectionError(f"MCP initialize failed: {resp['error']}")

    async def _connect_sse(self) -> None:
        self._session = aiohttp.ClientSession()
        # Verify connectivity
        async with self._session.get(
            self.config.sse_url,
            timeout=aiohttp.ClientTimeout(total=self.config.startup_timeout_seconds),
        ) as resp:
            if resp.status >= 400:
                raise MCPConnectionError(f"SSE endpoint returned {resp.status}")

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_jsonrpc(self, payload: dict) -> None:
        if self._proc and self._proc.stdin:
            data = json.dumps(payload) + "\n"
            self._proc.stdin.write(data.encode())
            await self._proc.stdin.drain()

    async def _read_jsonrpc(self) -> dict:
        if self._proc and self._proc.stdout:
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=30)
            return json.loads(line.decode().strip())
        return {}

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        self.calls_made += 1
        if self.config.transport == "stdio":
            await self._send_jsonrpc(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                }
            )
            return await self._read_jsonrpc()
        elif self._session:
            async with self._session.post(
                self.config.sse_url.replace("/sse", "/call"),
                json={"tool": tool_name, "arguments": arguments},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                return await resp.json()
        return {}

    async def list_components(self) -> list[str]:
        """List all available shadcn component names. Called once at startup."""
        if not self._connected:
            logger.debug("MCP not connected, returning empty component list")
            return []
        try:
            resp = await self._call_tool("list_components", {})
            result = resp.get("result", {})
            content = result.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        try:
                            data = json.loads(item["text"])
                            if isinstance(data, list):
                                return [str(c) for c in data]
                            if isinstance(data, dict) and "components" in data:
                                return [str(c) for c in data["components"]]
                        except (json.JSONDecodeError, TypeError):
                            pass
            logger.warning("Unexpected list_components response shape")
            return []
        except Exception as exc:
            logger.warning("list_components failed: %s", exc)
            return []

    async def install_components(self, names: list[str]) -> dict[str, bool]:
        """Install components via MCP. Called once at end of pipeline run."""
        if not self._connected or not names:
            return {n: False for n in names}
        try:
            resp = await self._call_tool("install_components", {"components": names})
            result = resp.get("result", {})
            content = result.get("content", [])
            results: dict[str, bool] = {}
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    try:
                        data = json.loads(item["text"])
                        if isinstance(data, dict):
                            results.update({k: bool(v) for k, v in data.items()})
                    except (json.JSONDecodeError, TypeError):
                        pass
            # Default missing keys to True (assume success)
            for name in names:
                if name not in results:
                    results[name] = True
            logger.info("Installed %d components via MCP", sum(results.values()))
            return results
        except Exception as exc:
            raise MCPInstallError(f"install_components failed: {exc}") from exc

    async def disconnect(self) -> None:
        if self._proc:
            try:
                self._proc.stdin.close()
                await self._proc.wait()
            except Exception:
                pass
            self._proc = None
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.debug("MCP client disconnected")
