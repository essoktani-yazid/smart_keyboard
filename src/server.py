"""Embedded micro-server for instantaneous, sub-millisecond real-time suggestions."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from src.pipeline import (
    EOS,
    complete_current_word,
    predict_next_word,
    backoff_top_k
)

_GLOBAL_MODELS = None
_SERVER_INSTANCE = None
_SERVER_PORT = None
_SERVER_LOCK = threading.Lock()


def set_models(tri, bi, uni):
    global _GLOBAL_MODELS
    _GLOBAL_MODELS = {"tri": tri, "bi": bi, "uni": uni}


def get_free_port(start_port=8555, max_tries=50):
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 8555


class SuggestionHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        # Silence console log noise for real-time keystrokes
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "ready": _GLOBAL_MODELS is not None}).encode("utf-8"))
            return

        if parsed.path == "/api/suggest":
            t0 = time.perf_counter()
            params = parse_qs(parsed.query)
            text = params.get("text", [""])[0]
            k_comp = int(params.get("k_comp", [4])[0])
            k_next = int(params.get("k_next", [4])[0])

            if not _GLOBAL_MODELS:
                response = {"error": "Models not loaded yet", "completions": [], "next_words": []}
            else:
                tri = _GLOBAL_MODELS["tri"]
                bi = _GLOBAL_MODELS["bi"]
                uni = _GLOBAL_MODELS["uni"]

                # Parse text
                # Check whether user is typing a word (no trailing space) or finished a word (trailing space)
                ends_with_space = len(text) > 0 and text[-1].isspace()
                raw_words = text.lower().replace(".", " . ").split()

                if ends_with_space or not text.strip():
                    # Word completion: nothing or previous word finished
                    current_prefix = ""
                    context_words = raw_words
                    completions = []
                else:
                    # Last token is currently being typed
                    current_prefix = raw_words[-1] if raw_words else ""
                    context_words = raw_words[:-1] if len(raw_words) > 1 else []
                    completions = complete_current_word(tri, bi, uni, context_words, current_prefix, k=k_comp)

                # Next word predictions
                # Context is all completed words (or context_words if typing)
                next_context = raw_words if ends_with_space else context_words
                next_words, order = predict_next_word(tri, bi, uni, next_context, k=k_next)

                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                response = {
                    "text": text,
                    "ends_with_space": ends_with_space,
                    "current_prefix": current_prefix,
                    "context_words": context_words,
                    "completions": completions,
                    "next_words": next_words,
                    "order": order,
                    "vocab_size": uni.V,
                    "latency_ms": latency_ms
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def start_suggestion_server(tri, bi, uni, port=None):
    """Starts the suggestion HTTP server in a background daemon thread."""
    global _SERVER_INSTANCE, _SERVER_PORT
    with _SERVER_LOCK:
        set_models(tri, bi, uni)
        if _SERVER_INSTANCE is not None:
            return _SERVER_PORT

        if port is None:
            port = get_free_port()

        server = HTTPServer(("127.0.0.1", port), SuggestionHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _SERVER_INSTANCE = server
        _SERVER_PORT = port
        return _SERVER_PORT
