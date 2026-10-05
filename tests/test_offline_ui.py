from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.testclient import TestClient

from ui.offline import LocalAssetsMiddleware


def test_standalone_ui_removes_external_assets_and_keeps_local_bootstrap():
    app = FastAPI()
    app.add_middleware(LocalAssetsMiddleware)

    @app.get("/")
    def index():
        return HTMLResponse('''<html><head>
          <link rel="preconnect" href="https://fonts.googleapis.com" />
          <script src="https://cdnjs.cloudflare.com/iframe-resizer.js" async></script>
          <script>window.config = {"label": "local goals"};</script>
          <script type="module" src="/assets/app.js"></script>
          <link rel="stylesheet" href="/assets/app.css">
        </head><body>Local lab</body></html>''')

    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "cloudflare" not in response.text
    assert "googleapis" not in response.text
    assert 'window.config = {"label": "local goals"}' in response.text
    assert "/assets/app.js" in response.text
    assert "/assets/app.css" in response.text


def test_queue_streams_are_not_filtered_or_buffered_by_html_policy():
    app = FastAPI()
    app.add_middleware(LocalAssetsMiddleware)

    @app.get("/gradio_api/queue/data")
    def stream():
        return StreamingResponse(iter([b"data: training\n\n", b"data: complete\n\n"]), media_type="text/event-stream")

    response = TestClient(app).get("/gradio_api/queue/data")
    assert response.text == "data: training\n\ndata: complete\n\n"
