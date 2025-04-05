import os
import re
import json
from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
from cryptography.fernet import Fernet
import language_tool_python
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Gemini API setup
GOOGLE_API_KEY = "AIzaSyBC7YMzKKrsfpt42PM9ObZtK7R6-QUHrXo"  
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Encryption setup
key = b'w6-1ePqZNLFMxJk6EWZeDnyXSeh8Kunwmu-N6QREkWg='
cipher = Fernet(key)

# File paths
HISTORY_FILE = "email_history.enc"
PREFS_FILE = "user_prefs.json"
FEEDBACK_FILE = "feedback.json"

# Grammar checker
grammar_tool = language_tool_python.LanguageTool('en-US')

# Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedding_cache = {}
suggestion_cache = {}

# Templates
TEMPLATES = {
    "apology": "Write an email apologizing for a delay in {context}.",
    "follow-up": "Write an email about {context}.",
    "request": "Write an email requesting {context}."
}

def get_email_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]
    embedding = embedder.encode(text)
    embedding_cache[text] = embedding
    return embedding

def find_similar_email(prompt, history):
    if not history:
        return None
    prompt_embedding = get_email_embedding(prompt)
    history_embeddings = [get_email_embedding(email) for email in history]
    similarities = [np.dot(prompt_embedding, he) / (np.linalg.norm(prompt_embedding) * np.linalg.norm(he)) for he in history_embeddings]
    max_idx = np.argmax(similarities)
    return history[max_idx] if similarities[max_idx] > 0.7 else None

def load_email_history(cipher):
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "rb") as f:
                content = f.read()
                if not content:
                    return []
                lines = content.split(b"\n---\n")
                decrypted = []
                for line in lines:
                    if line:
                        try:
                            decrypted.append(cipher.decrypt(line).decode('utf-8'))
                            print("Decrypted line:", decrypted[-1][:50])
                        except Exception as e:
                            print(f"Decryption error: {e}")
                return decrypted
        except Exception as e:
            print(f"History file error: {e}")
            return []
    return []

def save_email_to_history(email):
    try:
        history = load_email_history(cipher)
        encrypted_email = cipher.encrypt(email.encode('utf-8'))
        with open(HISTORY_FILE, "ab") as f:
            if history:
                f.write(b"\n---\n")
            f.write(encrypted_email)
    except Exception as e:
        print(f"Save history error: {e}")

def load_user_prefs():
    default_prefs = {
        "default_tone": "professional",
        "default_recipient": "",
        "signature": "Best regards,\n[Your Name]",
        "preferred_phrases": ["Looking forward to your response", "Please let me know"]
    }
    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE, "r") as f:
                prefs = json.load(f)
                default_prefs.update(prefs)
        except:
            pass
    return default_prefs

def save_user_prefs(prefs):
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f)
    except Exception as e:
        print(f"Save prefs error: {e}")

def save_feedback(rating):
    try:
        feedback = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                feedback = json.load(f)
        feedback.append({"rating": rating, "timestamp": datetime.now().isoformat()})
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedback, f)
    except Exception as e:
        print(f"Save feedback error: {e}")

def extract_subject(prompt):
    match = re.search(r"subject:\s*(.+)", prompt, re.IGNORECASE)
    return match.group(1).strip() if match else None

def generate_email(prompt, tone=None, recipient=None, template=None):
    history = load_email_history(cipher)
    prefs = load_user_prefs()
    tone = tone or prefs["default_tone"]
    recipient = recipient or prefs["default_recipient"]
    similar_email = find_similar_email(prompt, history)
    context = f"Similar past email:\n{similar_email}\n\n" if similar_email else ""
    if template and template in TEMPLATES:
        full_prompt = TEMPLATES[template].format(context=prompt)
    else:
        full_prompt = prompt
    full_prompt = f"{context}Write an email in a {tone} tone to {recipient}: {full_prompt}\nInclude: {prefs['preferred_phrases'][0]}"
    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=120)
        )
        raw_text = response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"Error: {str(e)}"
    subject = extract_subject(prompt) or "Untitled Email"
    if "Subject:" in raw_text:
        lines = raw_text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Subject:"):
                subject = line.replace("Subject:", "").strip()
                raw_text = "\n".join(lines[:i] + lines[i+1:]).strip()
                break
    corrected_text = grammar_tool.correct(raw_text)
    email = f"To: {recipient}\nSubject: {subject}\n\nDear {recipient},\n{corrected_text}\n{prefs['signature']}"
    save_email_to_history(email)
    return email

def suggest_reply(email_content):
    prefs = load_user_prefs()
    tone = prefs["default_tone"]
    history = load_email_history(cipher)
    similar_email = find_similar_email(email_content, history)
    context = f"Similar past email:\n{similar_email}\n\n" if similar_email else ""
    full_prompt = f"{context}Suggest a reply to this email in a {tone} tone: {email_content}\nInclude: {prefs['preferred_phrases'][0]}"
    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=100)
        )
        return grammar_tool.correct(response.text)
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"Error: {str(e)}"

def refine_text(text, action):
    actions = {
        "shorten": f"Shorten this text while maintaining its meaning: {text}",
        "improve": f"Improve the clarity and tone of this text: {text}",
        "formalize": f"Rewrite this text in a formal tone: {text}",
        "casual": f"Rewrite this text in a casual tone: {text}"
    }
    if action not in actions:
        return text
    try:
        response = model.generate_content(
            actions[action],
            generation_config=genai.types.GenerationConfig(max_output_tokens=120)
        )
        return grammar_tool.correct(response.text)
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"Error: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def home():
    output = ""
    history = load_email_history(cipher)
    prefs = load_user_prefs()
    if request.method == "POST":
        action = request.form.get("action")
        input_text = request.form.get("input_text", "")
        template = request.form.get("template")
        if action == "compose":
            output = generate_email(input_text, template=template)
        elif action == "reply":
            output = suggest_reply(input_text)
        elif action == "refine":
            refine_action = request.form.get("refine_action", "improve")
            output = refine_text(input_text, refine_action)
        elif action == "clear_history":
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            history = []
            output = "History cleared."
        elif action == "save_prefs":
            prefs["default_tone"] = request.form.get("default_tone", "professional")
            prefs["default_recipient"] = request.form.get("default_recipient", "")
            prefs["signature"] = request.form.get("signature", "Best regards,\n[Your Name]")
            prefs["preferred_phrases"] = [request.form.get("phrase", "Looking forward to your response")]
            save_user_prefs(prefs)
            output = "Preferences saved."
        elif action == "feedback":
            rating = request.form.get("rating")
            if rating:
                save_feedback(int(rating))
                output = "Feedback submitted."
    return render_template("index.html", output=output, history=history, prefs=prefs, templates=TEMPLATES.keys())

@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        text = request.json.get("text", "")
        if text in suggestion_cache:
            return jsonify({"suggestion": suggestion_cache[text]})
        if len(text) < 4:
            return jsonify({"suggestion": ""})
        prefs = load_user_prefs()
        history = load_email_history(cipher)
        similar_email = find_similar_email(text, history)
        context = f"Similar past email:\n{similar_email}\n\n" if similar_email else ""
        full_prompt = f"{context}Suggest a continuation for: {text}\nUse a {prefs['default_tone']} tone and include: {prefs['preferred_phrases'][0]}"
        suggestion = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=50)
        ).text
        suggestion_cache[text] = suggestion
        return jsonify({"suggestion": suggestion})
    except Exception as e:
        print(f"Suggestion error: {e}")
        return jsonify({"suggestion": "Error generating suggestion"})

if __name__ == "__main__":
    app.run(debug=True)