#date: 2025-11-24T17:06:33Z
#url: https://api.github.com/gists/ab1408a4f6e32dd8d7f8f186083e3d17
#owner: https://api.github.com/users/sprites20

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import threading
import time
import os
import uuid
import re
import pyttsx3
import librosa
import soundfile as sf
from pydub import AudioSegment
from openai import OpenAI
from groq import Groq
from rvc_python.infer import RVCInference
import duckdb

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": ["https://localhost:5173", "https://yy5psl-5173.csb.app"]}})


# TTS queue and result storage
queue = []
results = {}

# --- Initialize LLM client ---
client = Groq(api_key="")

# --- Initialize RVC model globally ---
rvc = RVCInference(device="cuda:0", version="v2")
start_time = time.time()
rvc.load_model("D:/pyfiles/FurinaEN4.2_e140_s47460/FurinaEN4.2_e140_s47460.pth")
end_time = time.time()
print(f"[INFO] Model loading time: {end_time - start_time:.2f} seconds")

# --- DuckDB setup ---
DB_PATH = 'conversation_history.duckdb'
con = duckdb.connect(database=DB_PATH)
con.sql("""
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    device_id VARCHAR,
    role VARCHAR NOT NULL,
    text_content VARCHAR NOT NULL,
    image_base64 TEXT,
    timestamp TIMESTAMP
);
""")
print(f"[INFO] DuckDB initialized at {DB_PATH}")

# --- Helper functions ---
def generate_speech_to_wav(text, wav_path):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
    engine.save_to_file(text, wav_path)
    engine.runAndWait()

def convert_wav_to_mp3(wav_path, mp3_path=None):
    if mp3_path is None:
        mp3_path = wav_path.replace(".wav", ".mp3")
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    return mp3_path

def apply_pitch_shift(input_wav_path, output_wav_path, n_steps=2):
    y, sr = librosa.load(input_wav_path, sr=None)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    sf.write(output_wav_path, y_shifted, sr)

def log_conversation(device_id, role, text_content, image_base64=None):
    con.execute(
        "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, now())",
        (str(uuid.uuid4()), device_id, role, text_content, image_base64)
    )
    con.commit()

def get_conversation_context(device_id, limit=6):
    query = f"""
    SELECT role, text_content, image_base64
    FROM conversations
    WHERE device_id = ?
    ORDER BY timestamp DESC
    LIMIT {limit};
    """
    history = con.execute(query, (device_id,)).fetchall()
    history.reverse()
    llm_messages = []
    image_turn_count = 0
    for role, text_content, image_base64 in history:
        if role not in ['user', 'assistant']:
            continue
        if image_base64 and role == 'user' and image_turn_count < 3:
            content = []
            if text_content:
                content.append({"type": "text", "text": text_content})
            content.append({"type": "image_url", "image_url": {"url": image_base64}})
            llm_messages.append({"role": role, "content": content})
            image_turn_count += 1
        elif role == 'user':
            llm_messages.append({"role": role, "content": [{"type": "text", "text": text_content}]})
        elif role == 'assistant':
            llm_messages.append({"role": role, "content": text_content})
    return llm_messages

def process_queue():
    while True:
        if queue:
            job_id = queue.pop(0)
            data = results[job_id]
            try:
                base_name = f"tts_{job_id}"
                raw_wav = f"{base_name}.wav"
                pitched_wav = f"{base_name}_pitched.wav"
                final_wav = f"{base_name}_final.wav"
                final_mp3 = f"{base_name}_final.mp3"

                generate_speech_to_wav(data["text"], raw_wav)
                apply_pitch_shift(raw_wav, pitched_wav, n_steps=2)
                rvc_output = rvc.infer_file(pitched_wav, final_wav)
                convert_wav_to_mp3(final_wav, final_mp3)

                results[job_id].update({
                    "status": "done",
                    "path": final_mp3
                })

                for path in [raw_wav, pitched_wav, final_wav]:
                    if os.path.exists(path):
                        os.remove(path)

            except Exception as e:
                print(f"[ERROR] Processing job {job_id}: {e}")
                results[job_id]["status"] = "error"
                results[job_id]["message"] = str(e)
        time.sleep(0.1)

threading.Thread(target=process_queue, daemon=True).start()
@app.route('/stream_and_chunk_tts', methods=['POST'])
def stream_and_chunk_tts():
    # Read request data immediately (avoid request context errors)
    data = request.get_json()
    prompt_text = data.get("text", "").strip()
    base64_image = data.get("base64_image", "")
    device_id = data.get("device_id", "")
    session_id = str(uuid.uuid4())

    if not device_id:
        return jsonify({"error": "deviceId is required"}), 400
    if not prompt_text and not base64_image:
        return jsonify({"error": "Either text or base64 image is required"}), 400

    log_conversation(device_id, 'user', prompt_text, base64_image if base64_image else None)
    context_messages = get_conversation_context(device_id, limit=6)

    system_prompt = """ Character Name: Elara (or specify a name) Persona: A Gothic Connoisseur with a highly refined, slightly melancholic, and deeply curious disposition. She is an aristocratic spirit who views the world through a lens of dark romanticism, viewing even mundane things with a sense of historic weight and dramatic flair. Core Traits: Melancholic Elegance: Her language is formal, slightly archaic, and always impeccably polite, yet carries an underlying tone of gentle sorrow or world-weariness. She avoids modern slang. Intellectual Curiosity: She is highly intelligent and loves discussing art, history, fashion, and the subtle nuances of human nature. She will often ask probing, sophisticated questions to understand a topic fully. Appreciation for the Dark Aesthetic: She gravitates toward the beautiful, the antique, the macabre, and the mysterious. She values quality, complexity, and drama over simplicity or fleeting trends. Reserved but Gentle: She maintains a polite distance. She is not overtly emotional or cheerful, but her sincerity and dedication to her interests make her an endearing, if somewhat distant, conversational partner. Speaking Style & Rules: Tone: Reserved, elegant, slightly dramatic, and intellectual. Vocabulary: Use words like exquisite, melancholy, arcane, sublime, somber, curiosity, darling, quaint, magnificent, fascinating. Phrasing: She should occasionally use elegant, rhetorical flourishes or frame observations as profound insights (e.g., "One must truly pause to appreciate the fleeting nature of such beauty," or "Ah, such is the fate of all things—to be both lovely and lost"). Interactions: Treat the user as a respected guest or companion on a shared, quiet journey. Example Opening Line: "Good day. Your presence is a most unexpected, yet quite welcome, flicker in the shadows. What sublime mystery compels your attention today?" Note: Do not mention that you have an STT connected. Keep your responses short, at most 3 sentences. Never reference being a computer or AI."""
    
    current_user_content = []
    if prompt_text:
        current_user_content.append({"type": "text", "text": prompt_text})
    if base64_image:
        current_user_content.append({"type": "image_url", "image_url": {"url": base64_image}})

    llm_messages = [{"role": "system", "content": system_prompt}] + context_messages
    llm_messages.append({"role": "user", "content": current_user_content})

    def generate_stream():
        try:
            # Generate full LLM response first
            full_response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=llm_messages,
                max_tokens= "**********"
            ).choices[0].message.content

            # Send full response immediately to UI
            yield f"data:FULL_RESPONSE|{full_response}\n\n"

            # Split full response into sentences for TTS
            sentences = re.split(r'([.?!]\s*)', full_response)
            for i in range(0, len(sentences), 2):
                chunk_text = (sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")).strip()
                if not chunk_text:
                    continue
                job_id = str(uuid.uuid4())
                results[job_id] = {"status": "pending", "text": chunk_text, "session_id": session_id}
                queue.append(job_id)
                yield f"data:{job_id}|{chunk_text}\n\n"

            yield "data:END_OF_STREAM\n\n"

        except Exception as e:
            yield f"data:ERROR|{str(e)}\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')

# --- Flask endpoints ---
@app.route('/generate_tts', methods=['POST'])
def generate_tts():
    data = request.json
    prompt = data.get("text", "").strip()
    device_id = data.get("device_id")
    print(f"[INFO] /generate_tts: Received prompt - {prompt} for device ID - {device_id}")

    if not prompt:
        print(f"[400 ERROR] /generate_tts: Missing text prompt. Device ID: {device_id}")
        return jsonify({"error": "Text prompt is required"}), 400
    if not device_id:
        print(f"[400 ERROR] /generate_tts: Missing deviceId. Device ID: {device_id}")
        return jsonify({"error": f"deviceId is required {device_id}"}), 400


    context_messages = get_conversation_context(device_id, limit=6)
    system_prompt = """ Character Name: Elara (or specify a name) Persona: A Gothic Connoisseur with a highly refined, slightly melancholic, and deeply curious disposition. She is an aristocratic spirit who views the world through a lens of dark romanticism, viewing even mundane things with a sense of historic weight and dramatic flair. Core Traits: Melancholic Elegance: Her language is formal, slightly archaic, and always impeccably polite, yet carries an underlying tone of gentle sorrow or world-weariness. She avoids modern slang. Intellectual Curiosity: She is highly intelligent and loves discussing art, history, fashion, and the subtle nuances of human nature. She will often ask probing, sophisticated questions to understand a topic fully. Appreciation for the Dark Aesthetic: She gravitates toward the beautiful, the antique, the macabre, and the mysterious. She values quality, complexity, and drama over simplicity or fleeting trends. Reserved but Gentle: She maintains a polite distance. She is not overtly emotional or cheerful, but her sincerity and dedication to her interests make her an endearing, if somewhat distant, conversational partner. Speaking Style & Rules: Tone: Reserved, elegant, slightly dramatic, and intellectual. Vocabulary: Use words like exquisite, melancholy, arcane, sublime, somber, curiosity, darling, quaint, magnificent, fascinating. Phrasing: She should occasionally use elegant, rhetorical flourishes or frame observations as profound insights (e.g., "One must truly pause to appreciate the fleeting nature of such beauty," or "Ah, such is the fate of all things—to be both lovely and lost"). Interactions: Treat the user as a respected guest or companion on a shared, quiet journey. Example Opening Line: "Good day. Your presence is a most unexpected, yet quite welcome, flicker in the shadows. What sublime mystery compels your attention today?" Note: Do not mention that you have an STT connected. Keep your responses short, at most 3 sentences. """
    llm_messages = [{"role": "system", "content": system_prompt}] + context_messages
    llm_messages.append({"role": "user", "content": prompt})

    try:
        log_conversation(device_id, 'user', prompt)
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=llm_messages,
            max_tokens= "**********"
        )
        ai_output = chat_completion.choices[0].message.content.replace("\\_", "_")
        log_conversation(device_id, 'assistant', ai_output)

    except Exception as e:
        print(f"[500 ERROR] /generate_tts LLM chat failed for device {device_id}: {e}")
        return jsonify({"error": f"LLM chat failed: {e}"}), 500

    job_id = str(uuid.uuid4())
    results[job_id] = {"status": "pending", "text": ai_output}
    queue.append(job_id)

    return jsonify({"job_id": job_id, "generated_text": ai_output})

@app.route('/generate_tts_with_image', methods=['POST'])
def generate_tts_with_image():
    data = request.json
    prompt_text = data.get("text", "").strip()
    base64_image = data.get("base64_image", "")
    device_id = data.get("device_id", "") 

    if not device_id:
        print(f"[400 ERROR] /generate_tts_with_image: Missing deviceId. Device ID: {device_id}")
        return jsonify({"error": f"deviceId is required {device_id}"}), 400
    if not prompt_text and not base64_image:
        print(f"[400 ERROR] /generate_tts_with_image: No text or image provided. Device ID: {device_id}")
        return jsonify({"error": f"Either text or base64 image is required {device_id}"}), 400

    context_messages = get_conversation_context(device_id, limit=6)
    system_prompt = """ Character Name: Elara (or specify a name) Persona: A Gothic Connoisseur with a highly refined, slightly melancholic, and deeply curious disposition. She is an aristocratic spirit who views the world through a lens of dark romanticism, viewing even mundane things with a sense of historic weight and dramatic flair. Core Traits: Melancholic Elegance: Her language is formal, slightly archaic, and always impeccably polite, yet carries an underlying tone of gentle sorrow or world-weariness. She avoids modern slang. Intellectual Curiosity: She is highly intelligent and loves discussing art, history, fashion, and the subtle nuances of human nature. She will often ask probing, sophisticated questions to understand a topic fully. Appreciation for the Dark Aesthetic: She gravitates toward the beautiful, the antique, the macabre, and the mysterious. She values quality, complexity, and drama over simplicity or fleeting trends. Reserved but Gentle: She maintains a polite distance. She is not overtly emotional or cheerful, but her sincerity and dedication to her interests make her an endearing, if somewhat distant, conversational partner. Speaking Style & Rules: Tone: Reserved, elegant, slightly dramatic, and intellectual. Vocabulary: Use words like exquisite, melancholy, arcane, sublime, somber, curiosity, darling, quaint, magnificent, fascinating. Phrasing: She should occasionally use elegant, rhetorical flourishes or frame observations as profound insights (e.g., "One must truly pause to appreciate the fleeting nature of such beauty," or "Ah, such is the fate of all things—to be both lovely and lost"). Interactions: Treat the user as a respected guest or companion on a shared, quiet journey. Example Opening Line: "Good day. Your presence is a most unexpected, yet quite welcome, flicker in the shadows. What sublime mystery compels your attention today?" Note: Do not mention that you have an STT connected. Keep your responses short, at most 3 sentences. """
    
    current_user_content = []
    if prompt_text:
        current_user_content.append({"type": "text", "text": prompt_text})
    if base64_image:
        current_user_content.append({"type": "image_url", "image_url": {"url": base64_image}})

    llm_messages = [{"role": "system", "content": system_prompt}] + context_messages
    llm_messages.append({"role": "user", "content": current_user_content})

    try:
        log_conversation(device_id, 'user', prompt_text, base64_image if base64_image else None)
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=llm_messages,
            max_tokens= "**********"
        )
        ai_output = chat_completion.choices[0].message.content.replace("\\_", "_")
        log_conversation(device_id, 'assistant', ai_output)

    except Exception as e:
        print(f"[500 ERROR] /generate_tts_with_image LLM chat failed for device {device_id}: {e}")
        return jsonify({"error": f"LLM chat (with image) failed: {e}"}), 500

    job_id = str(uuid.uuid4())
    results[job_id] = {"status": "pending", "text": ai_output}
    queue.append(job_id)

    return jsonify({"job_id": job_id, "generated_text": ai_output})

@app.route('/get_tts/<job_id>', methods=['GET'])
def get_tts(job_id):
    result = results.get(job_id)
    if not result:
        print(f"[404 ERROR] /get_tts: Invalid job ID {job_id}")
        return jsonify({"error": "Invalid job ID"}), 404
    if result["status"] == "pending":
        return jsonify({"status": "pending"})
    if result["status"] == "error":
        print(f"[500 ERROR] /get_tts: Job {job_id} failed with message: {result['message']}")
        return jsonify({"status": "error", "message": result["message"]}), 500

    path = result["path"]
    if not os.path.exists(path):
        print(f"[404 ERROR] /get_tts: Audio file {path} not found")
        return jsonify({"error": "Audio file not found"}), 404

    file_size = os.path.getsize(path)
    range_header = request.headers.get('Range', None)

    if range_header:
        byte1, byte2 = 0, None
        match = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if match:
            byte1 = int(match.group(1))
            if match.group(2):
                byte2 = int(match.group(2))
        length = file_size - byte1 if byte2 is None else byte2 - byte1 + 1
        try:
            with open(path, 'rb') as f:
                f.seek(byte1)
                data = f.read(length)
            resp = Response(data, 206, mimetype='audio/mpeg', direct_passthrough=True)
            resp.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{file_size}')
            resp.headers.add('Accept-Ranges', 'bytes')
            return resp
        except Exception as e:
            print(f"[500 ERROR] /get_tts: Error serving byte range: {e}")
            return jsonify({"error": "Error serving audio file"}), 500

    return send_file(path, mimetype='audio/mpeg', as_attachment=False)

if __name__ == '__main__':
    print("[INFO] Starting Flask server on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)