"""Deploy the session server to Modal:   modal deploy modal_deploy.py

One container, kept to one on purpose: pairing and request state live in memory, so every wearer, helper and
robot socket must land on the same process. Session logs go to the container's disk and are lost when it
stops; move them to a modal.Volume before relying on them as data.
The Nemotron key and the accepted session name come from .env and are stored as a Modal secret.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# This file is imported again inside the container, where .env does not exist (and must not: it is not uploaded).
env = (dict(line.split("=", 1) for line in (HERE / ".env").read_text().splitlines()
            if "=" in line and not line.lstrip().startswith("#")) if modal.is_local() else {})

app = modal.App("remote-hands")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("fastapi", "httpx", "pillow", "websockets")
         .add_local_python_source("remote_hands")
         .add_local_dir(HERE / "helper", "/root/helper"))
secret = modal.Secret.from_dict({
    "NEMOTRON_API_KEY": env["NEMOTRON_API_KEY"].strip(),
    "NEMOTRON_BASE_URL": env["NEMOTRON_BASE_URL"].strip(),
    "REMOTE_HANDS_SESSIONS": env["REMOTE_HANDS_DEPLOY_SESSION"].strip(),
    "REMOTE_HANDS_LOG_DIR": "/tmp/remote-hands-logs",
} if modal.is_local() else {})


# region: the Nemotron endpoint is in Finland; from Modal's US region calls to it sometimes hung for a minute.
@app.function(image=image, secrets=[secret], max_containers=1, timeout=3600, scaledown_window=1200, region="eu")
@modal.concurrent(max_inputs=200)
@modal.asgi_app()
def web():
    from remote_hands.session_server import app as fastapi_app
    return fastapi_app
