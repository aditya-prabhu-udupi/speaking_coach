# Speaking Coach

Pipeline: browser mic/upload -> 16 kHz WAV -> Flask `/api/analyze` -> Whisper (word timestamps) -> metrics -> NLTK habit detection -> LanguageTool -> rule-based feedback -> JSON to the page. Session history is saved in the browser (localStorage).

## Setup

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. NLTK data downloads automatically on first run.
Microphone recording needs `localhost` or HTTPS.

## Files

- `app.py` - Flask server (serves `static/`, provides `POST /api/analyze`)
- `speech_analyzer.py` - all analysis logic
- `static/index.html`, `static/style.css`, `static/app.js` - front end
