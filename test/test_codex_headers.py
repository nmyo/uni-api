import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import RequestModel
from core.request import get_codex_payload


def test_codex_payload_uses_current_cli_version_headers():
    request = RequestModel(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "say test"}],
        stream=True,
    )
    provider = {
        "provider": "codex",
        "base_url": "https://chatgpt.com/backend-api/codex/responses",
        "model": ["gpt-5.4"],
    }

    _, headers, _ = asyncio.run(get_codex_payload(request, "codex", provider, api_key="access-token"))

    assert headers["Authorization"] == "Bearer access-token"
    assert headers["Version"] == "0.125.0"
    assert headers["User-Agent"] == "codex_cli_rs/0.125.0"
