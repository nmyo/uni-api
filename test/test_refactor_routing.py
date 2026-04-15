import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from fastapi import BackgroundTasks
from starlette.responses import Response
from core.models import RequestModel
from routing import build_api_key_models_map


def test_build_api_key_models_map_resolves_nested_api_keys():
    config = {
        "providers": [
            {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1/chat/completions",
                "model": ["gpt-4.1", "gpt-4o-mini"],
            },
            {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com/v1/messages",
                "model": ["claude-sonnet-4-5"],
            },
        ],
        "api_keys": [
            {
                "api": "sk-root",
                "model": ["openai/*"],
            },
            {
                "api": "sk-nested",
                "model": ["sk-root/*", "anthropic/claude-sonnet-4-5"],
            },
        ],
    }

    models_map = build_api_key_models_map(config, ["sk-root", "sk-nested"])

    assert models_map["sk-root"] == ["gpt-4.1", "gpt-4o-mini"]
    assert models_map["sk-nested"] == ["gpt-4.1", "gpt-4o-mini", "claude-sonnet-4-5"]


def test_client_manager_reuses_single_client_under_concurrency(monkeypatch):
    created_clients = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_clients.append(self)

        async def aclose(self):
            return None

    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)

    async def run_test():
        manager = main.ClientManager(pool_size=4)
        await manager.init(
            {
                "headers": {"User-Agent": "test"},
                "http2": True,
                "verify": True,
                "follow_redirects": True,
            }
        )

        async def borrow_client():
            async with manager.get_client("https://example.com/v1/chat/completions") as client:
                await asyncio.sleep(0)
                return client

        clients = await asyncio.gather(*[borrow_client() for _ in range(20)])
        assert len(created_clients) == 1
        assert len({id(client) for client in clients}) == 1
        await manager.close()

    asyncio.run(run_test())


def test_model_request_handler_passes_selected_provider_key(monkeypatch):
    provider_name = "provider-a"

    class DummyCircularList:
        async def is_all_rate_limited(self, model):
            return False

        async def next(self, model):
            return "provider-key-1"

        def get_items_count(self):
            return 1

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-4.1": "gpt-4.1"},
                "base_url": "https://example.com/v1/chat/completions",
                "api": ["provider-key-1"],
                "preferences": {},
            }
        ]

    async def fake_process_request(
        request,
        provider,
        background_tasks,
        endpoint=None,
        role=None,
        timeout_value=0,
        keepalive_interval=None,
        provider_api_key_raw=None,
    ):
        assert provider_api_key_raw == "provider-key-1"
        return Response(content=b"ok", media_type="application/json")

    monkeypatch.setitem(main.provider_api_circular_list, provider_name, DummyCircularList())
    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "process_request", fake_process_request)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-4.1"],
                "preferences": {"AUTO_RETRY": False},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.keepalive_interval = {"global": {"default": 99999}}

    async def run_test():
        handler = main.ModelRequestHandler()
        response = await handler.request_model(
            RequestModel(
                model="gpt-4.1",
                messages=[{"role": "user", "content": "hello"}],
                stream=False,
            ),
            0,
            BackgroundTasks(),
        )
        assert response.status_code == 200

    asyncio.run(run_test())
