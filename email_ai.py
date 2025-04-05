import os
import re
import json
import time
import logging
import smtplib
from email.mime.text import MIMEText
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
from cryptography.fernet import Fernet
import language_tool_python
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Gemini API setup
GOOGLE_API_KEY = "AIzaSyBC7YMzKKrsfpt42PM9ObZtK7R6-QUHrXo"  # Replace with your API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Email configuration
EMAIL_SENDER = "your-email@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "your-app-password"  # Use App Password for Gmail with 2FA
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

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
    "follow-up": "Write an email following up about {context}.",
    "request": "Write an email requesting {context}."
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email Entry class for history management
class EmailEntry:
    def __init__(self, content, timestamp=None):
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.subject = self._extract_subject()

    def _extract_subject(self):
        match = re.search(r"Subject:\s*(.+)", self.content, re.IGNORECASE)
        return match.group(1).strip() if match else "Untitled"

    def to_dict(self):
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "subject": self.subject
        }

# Utility functions
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
    history_embeddings = [get_email_embedding(email.content) for email in history]
    similarities = [np.dot(prompt_embedding, he) / (np.linalg.norm(prompt_embedding) * np.linalg.norm(he))
                    for he in history_embeddings]
    max_idx = np.argmax(similarities)
    return history[max_idx].content if similarities[max_idx] > 0.75 else None

def load_email_history(cipher):
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "rb") as f:
                content = f.read()
                if not content:
                    return []
                lines = content.split(b"\n---\n")
                history = []
                for line in lines:
                    if line:
                        try:
                            decrypted = cipher.decrypt(line).decode('utf-8')
                            timestamp_match = re.search(r"Timestamp: (.+)\n", decrypted)
                            timestamp = (datetime.fromisoformat(timestamp_match.group(1))
                                        if timestamp_match else datetime.now())
                            content = re.sub(r"Timestamp: .+\n", "", decrypted)
                            history.append(EmailEntry(content, timestamp))
                        except Exception as e:
                            logger.error(f"Decryption error: {e}")
                return sorted(history, key=lambda x: x.timestamp, reverse=True)
        except Exception as e:
            logger.error(f"History file error: {e}")
            return []
    return []

def save_email_to_history(email):
    try:
        history = load_email_history(cipher)
        entry = EmailEntry(email)
        encrypted_email = cipher.encrypt(f"Timestamp: {entry.timestamp.isoformat()}\n{email}".encode('utf-8'))
        with open(HISTORY_FILE, "ab") as f:
            if history:
                f.write(b"\n---\n")
            f.write(encrypted_email)
    except Exception as e:
        logger.error(f"Save history error: {e}")

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
        except Exception as e:
            logger.error(f"Load prefs error: {e}")
    return default_prefs

def save_user_prefs(prefs):
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f)
    except Exception as e:
        logger.error(f"Save prefs error: {e}")

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
        logger.error(f"Save feedback error: {e}")

def extract_subject(prompt):
    match = re.search(r"subject:\s*(.+)", prompt, re.IGNORECASE)
    return match.group(1).strip() if match else None

# Email generation and processing functions
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

    full_prompt = (f"{context}Write an email in a {tone} tone to {recipient}: {full_prompt}\n"
                   f"Include: {prefs['preferred_phrases'][0]}\n"
                   f"Keep it concise and professional unless specified otherwise")

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=150,
                temperature=0.7
            )
        )
        raw_text = response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
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
    email = (f"To: {recipient}\n"
             f"Subject: {subject}\n\n"
             f"Dear {recipient},\n\n"
             f"{corrected_text}\n\n"
             f"{prefs['signature']}")

    save_email_to_history(email)
    return email

def suggest_reply(email_content):
    prefs = load_user_prefs()
    tone = prefs["default_tone"]
    history = load_email_history(cipher)
    similar_email = find_similar_email(email_content, history)
    context = f"Similar past email:\n{similar_email}\n\n" if similar_email else ""

    full_prompt = (f"{context}Suggest a reply to this email in a {tone} tone: {email_content}\n"
                   f"Include: {prefs['preferred_phrases'][0]}\n"
                   f"Keep it concise")

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=120,
                temperature=0.7
            )
        )
        return grammar_tool.correct(response.text.strip())
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"Error: {str(e)}"

def refine_text(text, action):
    actions = {
        "shorten": f"Shorten this text while maintaining its meaning: {text}",
        "improve": f"Improve the clarity and tone of this text: {text}",
        "formalize": f"Rewrite this text in a formal tone: {text}",
        "casual": f"Rewrite this text in a casual tone: {text}",
        "professional": f"Rewrite this text in a professional tone: {text}",
        "friendly": f"Rewrite this text in a friendly tone: {text}"
    }
    if action not in actions:
        return text

    try:
        response = model.generate_content(
            actions[action],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=150,
                temperature=0.7
            )
        )
        return grammar_tool.correct(response.text.strip())
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"Error: {str(e)}"

# Email sending function
def send_email(recipient, subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = recipient

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            logger.info(f"Email sent to {recipient}")
        save_email_to_history(f"To: {recipient}\nSubject: {subject}\n\n{body}")  # Save sent email to history
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return False

# Routes
@app.route("/", methods=["GET", "POST"])
def home():
    output = ""
    history = load_email_history(cipher)
    prefs = load_user_prefs()

    if request.method == "POST":
        action = request.form.get("action")
        input_text = request.form.get("input_text", "").strip()
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
            output = "History cleared successfully."
        elif action == "save_prefs":
            prefs["default_tone"] = request.form.get("default_tone", "professional")
            prefs["default_recipient"] = request.form.get("default_recipient", "")
            prefs["signature"] = request.form.get("signature", "Best regards,\n[Your Name]")
            prefs["preferred_phrases"] = [request.form.get("phrase", "Looking forward to your response")]
            save_user_prefs(prefs)
            output = "Preferences saved successfully."
        elif action == "feedback":
            rating = request.form.get("rating")
            if rating:
                save_feedback(int(rating))
                output = "Feedback submitted successfully."

    return render_template(
        "index.html",
        output=output,
        history=history,
        prefs=prefs,
        templates=TEMPLATES.keys()
    )

@app.route("/suggest", methods=["POST"])
def suggest():
    """
    Generate an email suggestion based on input text when explicitly requested.
    Optimized to reduce API calls using caching and local fallbacks.
    """
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({
                "suggestion": "",
                "status": "error",
                "message": "No input text provided",
                "source": "client"
            }), 400

        if len(text) < 10:
            return jsonify({
                "suggestion": "",
                "status": "warning",
                "message": "Text too short for meaningful suggestion",
                "source": "client"
            }), 200

        if text in suggestion_cache:
            logger.info(f"Returning cached suggestion for: {text[:50]}...")
            return jsonify({
                "suggestion": suggestion_cache[text],
                "status": "success",
                "message": "Retrieved from cache",
                "source": "cache",
                "cached": True
            }), 200

        prefs = load_user_prefs()
        history = load_email_history(cipher)
        similar_email = find_similar_email(text, history)
        context = f"Similar past email:\n{similar_email}\n\n" if similar_email else ""

        full_prompt = (
            f"{context}Generate a concise email suggestion based on: {text}\n"
            f"Use a {prefs['default_tone']} tone and include: {prefs['preferred_phrases'][0]}\n"
            f"Do not continue the text directly, provide a complete suggestion instead"
        )

        current_time = time.time()
        last_suggestion_time = getattr(suggest, 'last_api_call', 0)
        if current_time - last_suggestion_time < 2:
            return jsonify({
                "suggestion": "",
                "status": "warning",
                "message": "Please wait a moment before requesting another suggestion",
                "source": "cooldown"
            }), 429

        logger.info(f"Calling API for suggestion: {text[:50]}...")
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=80,
                temperature=0.7,
                top_p=0.9
            )
        )
        suggestion = grammar_tool.correct(response.text.strip())

        suggestion_cache[text] = suggestion
        suggest.last_api_call = time.time()
        if len(suggestion_cache) > 1000:
            suggestion_cache.pop(next(iter(suggestion_cache)))

        return jsonify({
            "suggestion": suggestion,
            "status": "success",
            "message": "Suggestion generated successfully",
            "source": "api",
            "cached": False
        }), 200

    except ValueError as ve:
        logger.error(f"Value error in suggestion: {str(ve)}")
        return jsonify({
            "suggestion": "",
            "status": "error",
            "message": f"Invalid input: {str(ve)}",
            "source": "client"
        }), 400

    except Exception as e:
        logger.error(f"Suggestion generation failed: {str(e)}", exc_info=True)
        return jsonify({
            "suggestion": "",
            "status": "error",
            "message": "Failed to generate suggestion",
            "source": "server"
        }), 500

@app.route("/send_email", methods=["POST"])
def send_email_route():
    """
    Send an email using SMTP and save it to history.
    """
    try:
        data = request.get_json()
        recipient = data.get("recipient", "").strip()
        subject = data.get("subject", "").strip()
        body = data.get("body", "").strip()

        if not all([recipient, subject, body]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        success = send_email(recipient, subject, body)
        if success:
            return jsonify({"status": "success", "message": "Email sent successfully"}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to send email"}), 500
    except Exception as e:
        logger.error(f"Send email error: {str(e)}")
        return jsonify({"status": "error", "message": "Server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)