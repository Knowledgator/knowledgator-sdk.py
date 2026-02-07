from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .base import BaseAPI, EndpointsConfig

if TYPE_CHECKING:
    from ..configs import SDKConfigs


class OpenModelAPI(BaseAPI):
    METHOD = "POST"
    CONTENT_TYPE = "application/json"

    def __init__(
        self,
        configs: SDKConfigs,
        model_id: str,
        method: Optional[str] = None,
        endpoints_config: Optional[EndpointsConfig] = None,
    ) -> None:
        self.model_id = model_id
        self.method = method
        super().__init__(configs=configs, endpoints_config=endpoints_config)

    def predict(self, body: dict[str, Any], timeout_ms: Optional[int] = None) -> Any:
        """Send a prediction request to the model's root inference endpoint."""
        if self.method:
            return self.call(self.method, body, timeout_ms)

        # The path is /inference/{model_id}/
        # We construct it dynamically.
        path = f"/inference/{self.model_id}/"

        server_url, server_headers = self.configs.get_server_details()
        headers = self.base_headers.copy()
        if server_headers:
            headers.update(server_headers)

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            json_body=body,
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )
        response = self.send(request)
        return response.json()

    async def predict_async(self, body: dict[str, Any], timeout_ms: Optional[int] = None) -> Any:
        """Asynchronously send a prediction request to the model's root inference endpoint."""
        path = f"/inference/{self.model_id}/"

        server_url, server_headers = self.configs.get_server_details()
        headers = self.base_headers.copy()
        if server_headers:
            headers.update(server_headers)

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            json_body=body,
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )
        response = await self.send_async(request)
        return response.json()

    def call(self, method: str, body: dict[str, Any], timeout_ms: Optional[int] = None) -> Any:
        """Send a request to a specific method of the model."""
        path = f"/inference/{self.model_id}/{method}/"

        server_url, server_headers = self.configs.get_server_details()
        headers = self.base_headers.copy()
        if server_headers:
            headers.update(server_headers)

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            json_body=body,
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )
        response = self.send(request)
        return response.json()

    async def call_async(self, method: str, body: dict[str, Any], timeout_ms: Optional[int] = None) -> Any:
        """Asynchronously send a request to a specific method of the model."""
        path = f"/inference/{self.model_id}/{method}/"

        server_url, server_headers = self.configs.get_server_details()
        headers = self.base_headers.copy()
        if server_headers:
            headers.update(server_headers)

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            json_body=body,
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )
        response = await self.send_async(request)
        return response.json()
