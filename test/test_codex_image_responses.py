import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.response import fetch_gpt_response_stream, _responses_output_to_text


class _DummyStreamingResponse:
    def __init__(self, chunks):
        self.status_code = 200
        self._chunks = list(chunks)

    async def aread(self):
        return b""

    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk


class _DummyStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyClient:
    def __init__(self, chunks):
        self._response = _DummyStreamingResponse(chunks)

    def stream(self, method, url, headers=None, content=None, timeout=None):
        _ = (method, url, headers, content, timeout)
        return _DummyStreamContext(self._response)


async def _collect_stream_body(chunks):
    client = _DummyClient(chunks)
    body_parts = []
    async for chunk in fetch_gpt_response_stream(
        client,
        "https://example.com/v1/responses",
        {"Accept": "text/event-stream"},
        {
            "model": "gpt-image-2",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "draw a meme"}],
                }
            ],
            "stream": True,
        },
        30,
    ):
        if isinstance(chunk, bytes):
            body_parts.append(chunk.decode("utf-8"))
        else:
            body_parts.append(chunk)
    return "".join(body_parts)


def test_fetch_gpt_response_stream_translates_image_output_item_done_and_drops_keepalive():
    chunks = [
        ": keepalive\n\n",
        (
            'event: response.created\n'
            'data: {"type":"response.created","response":{"id":"resp_img_done","status":"in_progress","model":"gpt-image-2","created_at":1710000000}}\n\n'
        ),
        (
            'event: keepalive\n'
            'data: {"type":"keepalive","sequence_number":7,"id":"resp_img_done"}\n\n'
        ),
        (
            'event: response.output_item.done\n'
            'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"image_generation_call","id":"ig_done","status":"completed","result":"done-image","output_format":"png"}}\n\n'
        ),
        "data: [DONE]\n\n",
    ]

    body = asyncio.run(_collect_stream_body(chunks))

    assert ": keepalive" in body
    assert '"type":"keepalive"' not in body.replace(" ", "")
    assert "data:image/png;base64,done-image" in body
    assert '"id":"resp_img_done"' in body.replace(" ", "")
    assert '"finish_reason":"stop"' in body.replace(" ", "")
    assert body.rstrip().endswith("data: [DONE]")


def test_responses_output_to_text_includes_image_generation_call():
    content, reasoning_content = _responses_output_to_text(
        {
            "output": [
                {
                    "type": "image_generation_call",
                    "result": "img-b64",
                    "output_format": "jpeg",
                }
            ]
        }
    )

    assert content == "![image](data:image/jpeg;base64,img-b64)"
    assert reasoning_content == ""
