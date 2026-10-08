"""
speech_analyzer.py
Core logic for the AI-Based Speaking Habit Analysis and Feedback System.

Pipeline: audio -> Whisper (word timestamps) -> metrics -> NLTK habit detection
          -> LanguageTool grammar check -> rule-based feedback -> JSON history
"""
import json
import os
import re
from collections import Counter
from datetime import datetime

import numpy as np

SAMPLE_RATE = 16000
HISTORY_FILE = "history.json"

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
FILLER_SINGLE = {
    "um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "hm", "mm",
    "like", "basically", "actually", "literally", "honestly", "totally",
}
FILLER_PHRASES = [("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of")]

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so", "to", "of", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "am", "i", "you", "he", "she", "it", "we", "they", "me", "my", "your", "our",
    "this", "that", "these", "those", "there", "here", "have", "has", "had", "do",
    "does", "did", "not", "no", "can", "will", "would", "could", "should", "just",
    "very", "about", "also", "then", "than", "what", "which", "who", "when", "how",
}

# Reference ranges used by the feedback engine
WPM_LOW, WPM_HIGH = 120, 160


# ----------------------------------------------------------------------------
# NLTK helpers 
# ----------------------------------------------------------------------------
def _ensure_nltk():
    try:
        import nltk
        for pkg, path in [
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab"),
            ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
            ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
        return nltk
    except Exception:
        return None


def split_sentences(text):
    nltk = _ensure_nltk()
    if nltk:
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            pass
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def pos_counts(tokens):
    nltk = _ensure_nltk()
    if not nltk or not tokens:
        return {}
    try:
        tags = nltk.pos_tag(tokens)
    except Exception:
        return {}
    groups = Counter()
    for _, tag in tags:
        if tag.startswith("NN"):
            groups["Nouns"] += 1
        elif tag.startswith("VB"):
            groups["Verbs"] += 1
        elif tag.startswith("JJ"):
            groups["Adjectives"] += 1
        elif tag.startswith("RB"):
            groups["Adverbs"] += 1
    return dict(groups)


# ----------------------------------------------------------------------------
# Audio + Whisper
# ----------------------------------------------------------------------------
def record_audio(seconds, sample_rate=SAMPLE_RATE):
    import sounddevice as sd
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def save_wav(audio, path, sample_rate=SAMPLE_RATE):
    from scipy.io import wavfile
    wavfile.write(path, sample_rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))


def load_whisper_model(size="base"):
    import whisper
    return whisper.load_model(size)


def load_wav(path, sample_rate=SAMPLE_RATE):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sample_rate:
        g = gcd(sr, sample_rate)
        data = resample_poly(data, sample_rate // g, sr // g).astype(np.float32)
    return data


def transcribe(model, audio):
    if isinstance(audio, str) and audio.lower().endswith(".wav"):
        audio = load_wav(audio)  # avoids the ffmpeg dependency for WAV files
    # Level-normalise quiet recordings so Whisper can hear soft words.
    if isinstance(audio, np.ndarray) and audio.size:
        peak = float(np.max(np.abs(audio)))
        if 0 < peak < 0.8:
            audio = (audio * (0.9 / peak)).astype(np.float32)
    # Short, natural prompt: a long list of fillers makes Whisper echo it back.
    prompt = "Um, so, uh, I was like, you know, thinking about it."
    result = model.transcribe(audio, word_timestamps=True, initial_prompt=prompt,
                              language="en", fp16=False, temperature=0.0,
                              condition_on_previous_text=False)
    words, segments = [], []
    for seg in result.get("segments", []):
        seg_words = []
        for w in seg.get("words", []):
            text = w["word"].strip()
            norm = re.sub(r"[^a-z']", "", text.lower())
            if not norm:
                continue
            item = {"text": text, "norm": norm,
                    "start": float(w["start"]), "end": float(w["end"])}
            words.append(item)
            seg_words.append(item)
        if seg_words:
            segments.append({"start": seg_words[0]["start"], "end": seg_words[-1]["end"],
                             "n_words": len(seg_words)})
    if isinstance(audio, np.ndarray):
        duration = len(audio) / SAMPLE_RATE
    else:
        duration = words[-1]["end"] if words else 0.0
    return words, segments, duration


# ----------------------------------------------------------------------------
# Habit detection
# ----------------------------------------------------------------------------
def find_fillers(norms):
    """Return list of (start_index, length, phrase) for each filler occurrence."""
    found, i = [], 0
    while i < len(norms):
        matched = False
        for ph in FILLER_PHRASES:
            if tuple(norms[i:i + len(ph)]) == ph:
                found.append((i, len(ph), " ".join(ph)))
                i += len(ph)
                matched = True
                break
        if matched:
            continue
        if norms[i] in FILLER_SINGLE:
            found.append((i, 1, norms[i]))
        i += 1
    return found


def find_consecutive_repeats(norms):
    """Return indices (of the second word) where a word is immediately repeated."""
    return [i for i in range(1, len(norms))
            if norms[i] == norms[i - 1] and norms[i] not in FILLER_SINGLE]


def overused_words(norms, min_count=3):
    content = [w for w in norms if w not in STOPWORDS and w not in FILLER_SINGLE and len(w) > 2]
    return [(w, c) for w, c in Counter(content).most_common() if c >= min_count][:8]


def frequent_phrases(norms, min_count=2):
    """Repeated bigrams/trigrams that are not made only of stopwords."""
    out = []
    for n in (3, 2):
        grams = Counter(tuple(norms[i:i + n]) for i in range(len(norms) - n + 1))
        for g, c in grams.items():
            if c >= min_count and not all(w in STOPWORDS or w in FILLER_SINGLE for w in g):
                out.append((" ".join(g), c))
    out.sort(key=lambda x: (-x[1], -len(x[0])))
    # drop bigrams contained in a reported trigram
    final = []
    for p, c in out:
        if not any(p in q and p != q for q, _ in final):
            final.append((p, c))
    return final[:8]


def find_pauses(words, threshold):
    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= threshold:
            pauses.append({"after_index": i - 1, "gap": round(gap, 2)})
    return pauses


def grammar_check(text, tool=None):
    """Returns list of {"message", "context"}. Needs Java; returns [] if unavailable."""
    if tool is False:
        return None  # caller already knows the checker can't start
    try:
        if tool is None:
            import language_tool_python
            tool = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text)
        issues = []
        for m in matches:
            rule = getattr(m, "ruleId", "") or getattr(m, "rule_id", "")
            if rule in ("UPPERCASE_SENTENCE_START", "COMMA_PARENTHESIS_WHITESPACE",
                        "WHITESPACE_RULE", "MORFOLOGIK_RULE_EN_US"):
                continue  # ASR output formatting noise / names
            issues.append({"message": m.message, "context": m.context})
        return issues
    except Exception:
        return None  # None => checker unavailable


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
def analyze(words, segments, duration, pause_threshold=0.8, grammar_tool=None):
    norms = [w["norm"] for w in words]
    total_words = len(words)
    minutes = duration / 60 if duration > 0 else 1e-9
    wpm = total_words / minutes

    fillers = find_fillers(norms)
    filler_counter = Counter(f[2] for f in fillers)
    repeats = find_consecutive_repeats(norms)
    pauses = find_pauses(words, pause_threshold)

    transcript = " ".join(w["text"] for w in words)
    sentences = split_sentences(transcript)
    unique_ratio = len(set(norms)) / total_words if total_words else 0
    avg_word_len = float(np.mean([len(w) for w in norms])) if norms else 0
    avg_sent_len = total_words / len(sentences) if sentences else 0

    pace_per_segment = []
    for s in segments:
        d = s["end"] - s["start"]
        if d > 0.5:
            pace_per_segment.append(round(s["n_words"] / d * 60, 1))

    grammar = grammar_check(transcript, grammar_tool)

    m = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(duration, 1),
        "total_words": total_words,
        "wpm": round(wpm, 1),
        "filler_count": len(fillers),
        "filler_per_100": round(len(fillers) / total_words * 100, 1) if total_words else 0,
        "filler_breakdown": dict(filler_counter),
        "pause_count": len(pauses),
        "pauses": pauses,
        "repeat_count": len(repeats),
        "overused_words": overused_words(norms),
        "frequent_phrases": frequent_phrases(norms),
        "unique_ratio": round(unique_ratio, 2),
        "avg_word_len": round(avg_word_len, 2),
        "avg_sentence_len": round(avg_sent_len, 1),
        "pos": pos_counts(norms),
        "grammar_issues": grammar,
        "pace_per_segment": pace_per_segment,
        "transcript": transcript,
        # indices used for highlighting
        "_filler_idx": sorted({k for s, n, _ in fillers for k in range(s, s + n)}),
        "_repeat_idx": repeats,
        "_pause_idx": [p["after_index"] for p in pauses],
    }
    m["strengths"], m["improvements"] = generate_feedback(m)
    m["score"] = compute_score(m)
    return m


def compute_score(m):
    score = 100
    if m["wpm"] < WPM_LOW:
        score -= min(20, (WPM_LOW - m["wpm"]) / 2)
    elif m["wpm"] > WPM_HIGH:
        score -= min(20, (m["wpm"] - WPM_HIGH) / 2)
    score -= min(30, m["filler_per_100"] * 4)
    dur_min = max(m["duration_sec"] / 60, 0.1)
    score -= min(20, (m["pause_count"] / dur_min) * 1.5)
    score -= min(10, m["repeat_count"] * 2)
    score -= min(10, len(m["grammar_issues"] or []) * 2)
    return int(max(0, round(score)))


# ----------------------------------------------------------------------------
# Rule-based feedback engine
# ----------------------------------------------------------------------------
def generate_feedback(m):
    good, bad = [], []
    wpm = m["wpm"]
    if wpm > 180:
        bad.append(f"Your speaking speed is very high ({wpm:.0f} WPM). Slow down and breathe between sentences; aim for 120-160 WPM.")
    elif wpm > WPM_HIGH:
        bad.append(f"Your pace is a little fast ({wpm:.0f} WPM). Try slowing down slightly so listeners can follow.")
    elif wpm < 100:
        bad.append(f"Your pace is quite slow ({wpm:.0f} WPM). Practise speaking more fluently to keep listeners engaged.")
    elif wpm < WPM_LOW:
        bad.append(f"Your pace is slightly slow ({wpm:.0f} WPM). A little more energy will help; aim for 120-160 WPM.")
    else:
        good.append(f"Comfortable speaking pace ({wpm:.0f} WPM) within the ideal 120-160 range.")

    f100 = m["filler_per_100"]
    if f100 >= 5:
        top = ", ".join(f'"{k}"' for k, _ in sorted(m["filler_breakdown"].items(), key=lambda x: -x[1])[:3])
        bad.append(f"High filler-word use ({m['filler_count']} fillers, {f100}/100 words), mostly {top}. Replace them with a silent pause.")
    elif f100 >= 2:
        bad.append(f"Moderate filler-word use ({m['filler_count']} fillers). Try pausing silently instead of saying 'um' or 'like'.")
    else:
        good.append("Very few filler words, so your speech sounds confident and clean.")

    dur_min = max(m["duration_sec"] / 60, 0.1)
    ppm = m["pause_count"] / dur_min
    if ppm > 6:
        bad.append(f"Many long pauses ({m['pause_count']}). Plan your key points beforehand to reduce hesitation.")
    elif m["pause_count"] == 0 and m["total_words"] > 40:
        good.append("Smooth delivery with no awkward long pauses.")
    else:
        good.append("Pauses are under control.")

    if m["repeat_count"] >= 2:
        bad.append(f"You repeated words back-to-back {m['repeat_count']} times (e.g. 'I I think'). Finish the thought before speaking.")
    if m["overused_words"]:
        ws = ", ".join(f'"{w}" ({c}x)' for w, c in m["overused_words"][:3])
        bad.append(f"Overused words: {ws}. Use synonyms for variety.")
    if m["frequent_phrases"]:
        ps = ", ".join(f'"{p}" ({c}x)' for p, c in m["frequent_phrases"][:3])
        bad.append(f"Repeated phrases: {ps}.")

    if m["total_words"] >= 30:
        if m["unique_ratio"] >= 0.65:
            good.append(f"Good vocabulary variety (unique-word ratio {m['unique_ratio']}).")
        elif m["unique_ratio"] < 0.5:
            bad.append(f"Limited vocabulary variety (unique-word ratio {m['unique_ratio']}). Try using more varied words.")
    if m["avg_sentence_len"] > 30:
        bad.append("Your sentences are very long. Break them into shorter ones for clarity.")

    g = m["grammar_issues"]
    if g is None:
        pass
    elif len(g) == 0:
        good.append("No grammar mistakes detected.")
    else:
        bad.append(f"{len(g)} possible grammar issue(s) found. Review them below.")

    return good, bad


# ----------------------------------------------------------------------------
# Highlighted transcript (HTML)
# ----------------------------------------------------------------------------
def highlight_html(words, m):
    fill, rep, pau = set(m["_filler_idx"]), set(m["_repeat_idx"]), set(m["_pause_idx"])
    parts = []
    for i, w in enumerate(words):
        t = w["text"]
        if i in fill:
            t = f'<span style="background:#ffd54f;padding:1px 3px;border-radius:3px">{t}</span>'
        elif i in rep:
            t = f'<span style="background:#ef9a9a;padding:1px 3px;border-radius:3px">{t}</span>'
        parts.append(t)
        if i in pau:
            parts.append('<span style="color:#888">[pause]</span>')
    return " ".join(parts)


# ----------------------------------------------------------------------------
# Progress tracking (JSON)
# ----------------------------------------------------------------------------
_SAVE_KEYS = ["timestamp", "duration_sec", "total_words", "wpm", "filler_count",
              "filler_per_100", "pause_count", "repeat_count", "unique_ratio", "score"]


def save_result(m, path=HISTORY_FILE):
    history = load_history(path)
    history.append({k: m[k] for k in _SAVE_KEYS})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_history(path=HISTORY_FILE):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []