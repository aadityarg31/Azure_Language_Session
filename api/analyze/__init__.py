import json
import logging
import os
import urllib.error
import urllib.request

import azure.functions as func

# Read these from Azure environment variables / application settings —
# never hard-code the key here.
ENDPOINT = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
KEY = os.environ.get("LANGUAGE_KEY", "")

API_VERSION = "2023-04-01"
TIMEOUT_SECONDS = 20
MAX_CHARS = 5000


def main(req: func.HttpRequest) -> func.HttpResponse:
    endpoint = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
    key = os.environ.get("LANGUAGE_KEY", "")
    if not endpoint or not key :
        return _json_response(
            {"error": "Server is missing LANGUAGE_ENDPOINT / LANGUAGE_KEY environment variables."},
            500,
        )

    try:
        body = req.get_json()
    except ValueError:
        body = {}

    text = (body.get("text") or "").strip()
    if not text:
        return _json_response({"error": 'Request body must include non-empty "text".'}, 400)
    if len(text) > MAX_CHARS:
        return _json_response({"error": f"Text must be {MAX_CHARS} characters or fewer."}, 400)

        # 1. Original Session Tasks (Kept perfectly safe)
    try:
        sentiment_doc = _call_language("SentimentAnalysis", text)["results"]["documents"][0]
        keyphrase_doc = _call_language("KeyPhraseExtraction", text)["results"]["documents"][0]
        entity_doc = _call_language("EntityRecognition", text)["results"]["documents"][0]
    except Exception:
        logging.exception("Original Azure AI Language calls failed")
        return _json_response({"error": "Azure AI Language request failed. Check key/endpoint/quota."}, 502)

        # 2. New Assignment Task: Language Detection
    try:
        lang_doc = _call_language("LanguageDetection", text)["results"]["documents"][0]
        
        # FIX IS HERE: Look inside the 'detectedLanguage' or 'detectedLanguages' array
        detected_languages_list = lang_doc.get("detectedLanguages", [])
        if detected_languages_list:
            detected_language = detected_languages_list[0].get("name", "Unknown")
        else:
            # Fallback check for single object dictionary structure
            detected_language = lang_doc.get("detectedLanguage", {}).get("name", "Unknown")
            
    except Exception:
        logging.exception("Language Detection failed")
        detected_language = "Unavailable"


    # 3. New Assignment Task: PII Entity Redaction
    try:
        pii_doc = _call_language("PiiEntityRecognition", text)["results"]["documents"][0]
        redacted_text = pii_doc.get("redactedText", text)
    except Exception:
        logging.exception("PII Redaction failed")
        redacted_text = text

    # 4. New Assignment Task: Summarization (Isolated so it won't crash your web app)
    try:
        summary_doc = _call_language("ExtractiveSummarization", text)["results"]["documents"][0]
        summary_sentences = [s["text"] for s in summary_doc.get("sentences", [])]
        summary_paragraph = " ".join(summary_sentences)
    except Exception:
        logging.exception("Summarization skipped (Requires Asynchronous Job Endpoint)")
        summary_paragraph = "Summarization requires an Asynchronous Jobs endpoint to execute."

    # 5. Send all data safely to your web page
    result = {
        "sentiment": sentiment_doc["sentiment"],
        "confidenceScores": sentiment_doc["confidenceScores"],
        "keyPhrases": keyphrase_doc.get("keyPhrases", []),
        "entities": [
            {"text": e["text"], "category": e["category"]}
            for e in entity_doc.get("entities", [])
        ],
        "detectedLanguage": detected_language,
        "redactedText": redacted_text,
        "summary": summary_paragraph
    }
    return _json_response(result, 200)


def _call_language(kind: str, text: str) -> dict:
    url = f"{ENDPOINT}/language/:analyze-text?api-version={API_VERSION}"

    payload_parameters = {"modelVersion": "latest"}
    if kind == "PiiEntityRecognition":
        payload_parameters["piiCategories"] = ["All"]
    
    payload = {
        "kind": kind,
        "parameters": payload_parameters,
        "analysisInput": {"documents": [{"id": "1", "text": text}]},
    }

    if kind != "LanguageDetection":
        payload["analysisInput"]["documents"][0]["language"] = "en"
        
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{kind} failed: {exc.code} {detail}") from exc


def _json_response(payload: dict, status: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload),
        status_code=status,
        mimetype="application/json",
    )
