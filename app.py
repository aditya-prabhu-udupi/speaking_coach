import os
import tempfile
import threading

from flask import Flask, jsonify, request, send_from_directory

import speech_analyzer as sa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MAX_UPLOAD_MB = 25
VALID_MODELS = {"tiny", "base", "small"}

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_models = {}
_model_lock = threading.Lock()      # protects the model cache
_run_lock = threading.Lock()        # Whisper/LanguageTool are not safe to run in parallel
_grammar = {"tool": None, "tried": False}


def get_model(size):
    with _model_lock:
        if size not in _models:
            _models[size] = sa.load_whisper_model(size)
        return _models[size]


def get_grammar_tool():
    """Start LanguageTool once. Returns False if it can't run (e.g. no Java)."""
    if not _grammar["tried"]:
        _grammar["tried"] = True
        try:
            import language_tool_python
            _grammar["tool"] = language_tool_python.LanguageTool("en-US")
        except Exception:
            _grammar["tool"] = False
    return _grammar["tool"]


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"That file is over {MAX_UPLOAD_MB} MB. Use a shorter clip."), 413


@app.post("/api/analyze")
def analyze():
    f = request.files.get("audio")
    if f is None:
        return jsonify(error="No audio was received. Please record or upload again."), 400

    size = request.form.get("model", "base")
    if size not in VALID_MODELS:
        size = "base"

    # The browser always sends 16 kHz mono WAV, so no ffmpeg is needed.
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        f.save(path)
        with _run_lock:
            model = get_model(size)
            words, segments, duration = sa.transcribe(model, path)
            if not words:
                return jsonify(error="No speech detected. Try again closer to the mic."), 422
            res = sa.analyze(words, segments, duration, 0.8, get_grammar_tool())
        # the front end renders the transcript word by word
        res["words"] = [w["text"] for w in words]
        return jsonify(res)
    except Exception as e:  # noqa: BLE001
        app.logger.exception("analysis failed")
        return jsonify(error=f"Analysis failed: {e}"), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def warm_up():
    """Do the slow one-time setup at startup so the first request is fast."""
    try:
        sa._ensure_nltk()
        sa.split_sentences("Warm up.")
        sa.pos_counts(["warm", "up"])
        get_grammar_tool()
        with _run_lock:
            get_model("small")
        app.logger.info("Warm-up finished.")
    except Exception:  # noqa: BLE001
        app.logger.exception("warm-up failed (will load on first request instead)")


if __name__ == "__main__":
    threading.Thread(target=warm_up, daemon=True).start()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)