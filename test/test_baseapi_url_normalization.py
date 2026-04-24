import asyncio
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import ImageGenerationRequest
from core.request import get_dalle_payload
from core.utils import BaseAPI


@pytest.mark.parametrize(
    ("source_url", "expected_image_url"),
    [
        (
            "https://oaix.fugue.pro/v1/responses",
            "https://oaix.fugue.pro/v1/images/generations",
        ),
        (
            "https://oaix.fugue.pro/v1/responses/compact",
            "https://oaix.fugue.pro/v1/images/generations",
        ),
        (
            "https://oaix.fugue.pro/v1/images/generations",
            "https://oaix.fugue.pro/v1/images/generations",
        ),
    ],
)
def test_base_api_normalizes_image_urls(source_url, expected_image_url):
    assert BaseAPI(source_url).image_url == expected_image_url


def test_get_dalle_payload_uses_normalized_images_endpoint_for_responses_provider():
    provider = {
        "provider": "fugue-codex",
        "base_url": "https://oaix.fugue.pro/v1/responses",
        "api": "change-me",
        "engine": "codex",
        "model": ["gpt-image-2"],
    }
    request = ImageGenerationRequest(model="gpt-image-2", prompt="test prompt")

    url, headers, payload = asyncio.run(get_dalle_payload(request, "dalle", provider, api_key="change-me"))

    assert url == "https://oaix.fugue.pro/v1/images/generations"
    assert headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer change-me",
    }
    assert payload == {
        "model": "gpt-image-2",
        "prompt": "test prompt",
    }
