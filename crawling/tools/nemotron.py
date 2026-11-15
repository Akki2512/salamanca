"""Minimal client for the team's OpenAI-compatible Nemotron endpoint.

Reads NEMOTRON_BASE_URL, NEMOTRON_API_KEY and NEMOTRON_MODEL.
"""
import base64
import io
import json
import os
import urllib.request


def ask(prompt, schema, image=None, max_tokens=400):
    """Return the model's reply parsed as JSON matching `schema` (a strict json_schema dict)."""
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": url}})
    body = {
        "model": os.environ["NEMOTRON_MODEL"],
        "temperature": 0,
        "max_tokens": max_tokens,
        # Without this the model spends its token budget on hidden reasoning and returns no content.
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": schema},
        "messages": [{"role": "user", "content": content}],
    }
    req = urllib.request.Request(
        os.environ["NEMOTRON_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["NEMOTRON_API_KEY"],
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(json.load(resp)["choices"][0]["message"]["content"])


def strict_schema(name, properties):
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        },
    }
