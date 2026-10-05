from html.parser import HTMLParser


class LocalAssetsParser(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=False)
        self.text = text
        self.offsets = [0]
        for line in text.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.removals = []
        self.script_start = None

    def position(self):
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        external = lambda value: str(value or "").startswith(("http://", "https://", "//"))
        if tag == "script" and external(attrs.get("src")):
            self.script_start = self.position()
        if tag == "link" and external(attrs.get("href")) and attrs.get("rel") in {"preconnect", "dns-prefetch", "stylesheet"}:
            start = self.position()
            self.removals.append((start, start + len(self.get_starttag_text())))

    def handle_endtag(self, tag):
        if tag == "script" and self.script_start is not None:
            self.removals.append((self.script_start, self.position() + len("</script>")))
            self.script_start = None

    def local_html(self):
        self.feed(self.text)
        text = self.text
        for start, end in sorted(self.removals, reverse=True):
            text = text[:start] + text[end:]
        return text.encode("utf-8")


class LocalAssetsMiddleware:
    """Remove optional external assets from Gradio's standalone HTML, not API streams."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/":
            return await self.app(scope, receive, send)
        start = None
        chunks = []

        async def capture(message):
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    body = b"".join(chunks)
                    headers = dict(start["headers"])
                    if b"text/html" in headers.get(b"content-type", b"") and b"content-encoding" not in headers:
                        body = LocalAssetsParser(body.decode("utf-8")).local_html()
                        start["headers"] = [(key, value) for key, value in start["headers"] if key != b"content-length"]
                    await send(start)
                    await send({"type": "http.response.body", "body": body})
            else:
                await send(message)

        await self.app(scope, receive, capture)
