import os
import json
import urllib.request
import urllib.error
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
SARVAM_API_KEY = (os.environ.get("SARVAM_API_KEY") or "").strip()

GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]


def build_prompt(question: str, language: str) -> str:
    lang_map = {
        "ta-IN": "Tamil",
        "en-IN": "English",
        "hi-IN": "Hindi",
        "te-IN": "Telugu",
        "ml-IN": "Malayalam",
        "kn-IN": "Kannada",
    }
    lang_label = lang_map.get(language, "Tamil" if str(language).startswith("ta") else "English")
    return (
        f"You are AI Farmer Friend helping Indian farmers with practical crop and general fertilizer advice.\n"
        f"Reply ONLY in {lang_label} using 4 short bullet points.\n"
        f"Never invent fertilizer or pesticide dosages; advise soil testing and consulting local agricultural officers.\n\n"
        f"Farmer question: {question}"
    )


def call_groq_model(prompt: str, model_id: str, max_tokens: int = 400) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in Vercel Environment Variables.")

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "AI-Farmer-Friend/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Groq HTTP {err.code} ({model_id}): {detail}") from err

    choices = body.get("choices") or []
    text = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    if not text:
        raise RuntimeError(f"Empty response from Groq model {model_id}")
    return text


def ask_groq(prompt: str, max_tokens: int = 400):
    last_error = None
    for model_id in GROQ_MODELS:
        try:
            answer = call_groq_model(prompt, model_id, max_tokens=max_tokens)
            return answer, model_id
        except Exception as err:
            last_error = err
            app.logger.warning(f"Groq model {model_id} failed: {err}")
            continue
    raise last_error or RuntimeError("All Groq models failed.")


def demo_answer(question: str, language: str) -> str:
    if str(language).startswith("ta"):
        return (
            "• தற்காலிகமாக AI சேவையில் நெரிசல் உள்ளதால் பொதுவான மாதிரி பதில் காட்டப்படுகிறது.\n"
            "• மண் பரிசோதனை அடிப்படையில் அடித்தள உரம் மற்றும் NPK உரங்களைத் திட்டமிடுங்கள்.\n"
            "• உரம் அல்லது பூச்சிக்கொல்லி அளவுகளை அனுமானிக்காமல் உள்ளூர் வேளாண் அலுவலரிடம் ஆலோசிக்கவும்.\n"
            "• மண்ணின் ஈரப்பதத்திற்கு ஏற்ப சீரான இடைவெளியில் நீர்ப்பாசனம் செய்யவும்."
        )
    return (
        "• Showing general baseline guidance while Groq is temporarily busy or rate-limited.\n"
        "• Base basal manure and NPK applications on a soil test report and crop stage.\n"
        "• Never guess chemical fertilizer or pesticide dosages; consult your local agricultural officer.\n"
        "• Irrigate steadily according to soil moisture and avoid waterlogging."
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test")
def test_route():
    key_info = {
        "groq_key_present": bool(GROQ_API_KEY),
        "prefix": GROQ_API_KEY[:6] if GROQ_API_KEY else None,
        "suffix": GROQ_API_KEY[-4:] if GROQ_API_KEY else None,
    }
    try:
        reply, model_used = ask_groq("Reply with the single word: Working", max_tokens=20)
        return jsonify({
            "ok": True,
            "provider": "groq",
            "model_used": model_used,
            "reply": reply,
            "key_info": key_info,
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "provider": "groq",
            "error": str(e),
            "key_info": key_info,
        }), 200


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    language = data.get("language", "ta-IN")

    if not question:
        return jsonify({"error": "Please enter a farming question."}), 400

    prompt = build_prompt(question, language)

    try:
        answer, model_used = ask_groq(prompt, max_tokens=400)
        return jsonify({
            "answer": answer,
            "provider": "groq",
            "model_used": model_used,
        })
    except Exception as e:
        app.logger.warning(f"Groq fallback triggered ({e}); returning safe baseline answer.")
        return jsonify({
            "answer": demo_answer(question, language),
            "provider": "demo",
            "fallback": True,
            "error": str(e),
        })


@app.route("/speak", methods=["POST"])
def speak():
    try:
        if not SARVAM_API_KEY:
            return jsonify({"error": "Sarvam API key missing"}), 500

        data = request.get_json(silent=True) or {}

        text = (data.get("text") or "").strip()
        language = data.get("language", "ta-IN")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if len(text) > 2500:
            return jsonify({"error": "Text exceeds 2500 characters"}), 400

        if language not in ["ta-IN", "en-IN"]:
            return jsonify({"error": "Unsupported language"}), 400

        payload = {
            "text": text,
            "language_code": language,
            "model": "bulbul:v4-flash",
            "speaker": "simran_en_customer",
            "output_audio_codec": "mp3",
            "speech_sample_rate": 24000
        }

        req = urllib.request.Request(
            "https://api.sarvam.ai/text-to-speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "api-subscription-key": SARVAM_API_KEY
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))

        if not result.get("audios"):
            return jsonify({"error": "No audio returned"}), 502

        return jsonify({
            "audio": result["audios"][0],
            "mime_type": "audio/mpeg"
        })

    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        return jsonify({"error": details}), e.code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
