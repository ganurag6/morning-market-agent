from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

import httpx


logger = logging.getLogger(__name__)


class MissingAPIKeyError(RuntimeError):
    pass


class PerplexityClient:
    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError(
                "PERPLEXITY_API_KEY is missing. Set it in the environment or .env."
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url or "https://api.perplexity.ai/search"

    def search(
        self,
        *,
        query: str,
        max_results: int = 10,
        country: str = "US",
        max_tokens_per_page: int = 512,
        search_recency_filter: str = "day",
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "max_results": max_results,
            "country": country,
            "max_tokens_per_page": max_tokens_per_page,
            "search_recency_filter": search_recency_filter,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._post_with_retries(self.base_url, payload, headers)
        return response.json()

    def _post_with_retries(
        self, url: str, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                if response.status_code in {429} or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                backoff = 2 ** (attempt - 1)
                logger.warning(
                    "Perplexity request failed (attempt %s/%s). Retrying in %ss.",
                    attempt,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError("Perplexity request failed after retries.") from last_error
