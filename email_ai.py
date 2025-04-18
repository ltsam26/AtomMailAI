import os
import re
import json
import time
import logging
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
import google.generativeai as genai
from cryptography.fernet import Fernet
import language_tool_python
from sentence_transformers import SentenceTransformer
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Setup logging at the top
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Gemini API setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Token refresh failed: {e}")
                raise
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=5000, redirect_uri="http://localhost:5000/oauth2callback")
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

# Encryption setup with validation
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY not set in .env")

# Validate and fix the encryption key
try:
    # Strip whitespace and ensure proper padding
    ENCRYPTION_KEY = ENCRYPTION_KEY.strip()
    # Add padding if necessary (base64 strings should be a multiple of 4 in length)
    padding_needed = (4 - len(ENCRYPTION_KEY) % 4) % 4
    ENCRYPTION_KEY += "=" * padding_needed
    # Attempt to create Fernet instance
    cipher = Fernet(ENCRYPTION_KEY)
    logger.info("Encryption key loaded successfully")
except ValueError as e:
    logger.error(f"Invalid ENCRYPTION_KEY: {ENCRYPTION_KEY[:10]}... (length: {len(ENCRYPTION_KEY)}). Error: {e}")
    raise ValueError("ENCRYPTION_KEY must be a valid 32-byte URL-safe base64-encoded string. Check your .env file.")

# File paths
HISTORY_FILE = "email_history.enc"
PREFS_FILE = "user_prefs.json"
FEEDBACK_FILE = "feedback.json"

# Grammar checker and embedding model
grammar_tool = language_tool_python.LanguageTool('en-US')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedding_cache = {}
suggestion_cache = {}

# Templates
TEMPLATES = {
    "apology": "Write an email apologizing for a delay in {context}.",
    "follow-up": "Write an email following up about {context}.",
    "request": "Write an email requesting {context}."
}

# Email Entry class
class EmailEntry:
    def __init__(self, content, timestamp=None):
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.subject = self._extract_subject()

    def _extract_subject(self):
        match = re.search(r"Subject:\s*(.+)", self.content, re.IGNORECASE)
        return match.group(1).strip() if match else "Untitled"

    def to_dict(self):
        return {"content": self.content, "timestamp": self.timestamp.isoformat(), "subject": self.subject}

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
                        decrypted = cipher.decrypt(line).decode('utf-8')
                        timestamp_match = re.search(r"Timestamp: (.+)\n", decrypted)
                        timestamp = datetime.fromisoformat(timestamp_match.group(1)) if timestamp_match else datetime.now()
                        content = re.sub(r"Timestamp: .+\n", "", decrypted)
                        history.append(EmailEntry(content, timestamp))
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
        "preferred_phrases": ["Looking forward to your response"]
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

# Gmail API email functions
def send_email(recipient, subject, body):
    try:
        service = get_gmail_service()
        message = MIMEText(body)
        message['to'] = recipient
        message['subject'] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        body = {'raw': raw}
        service.users().messages().send(userId="me", body=body).execute()
        logger.info(f"Email sent to {recipient}")
        save_email_to_history(f"To: {recipient}\nSubject: {subject}\n\n{body}")
        return True
    except HttpError as e:
        logger.error(f"Gmail API error: {e} - {e.content}")
        raise Exception(f"Gmail API error: {e.status_code} - {e.reason}")
    except Exception as e:
        logger.error(f"Unexpected error in send_email: {e}")
        raise

def check_new_emails():
    try:
        service = get_gmail_service()
        results = service.users().messages().list(userId="me", labelIds=["INBOX"], q="is:unread").execute()
        messages = results.get("messages", [])
        emails = []
        for msg in messages[:5]:
            msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
            headers = msg_data["payload"]["headers"]
            subject = next(h["value"] for h in headers if h["name"] == "Subject")
            from_ = next(h["value"] for h in headers if h["name"] == "From")
            snippet = msg_data.get("snippet", "")
            emails.append({"id": msg["id"], "subject": subject, "from": from_, "snippet": snippet})
        socketio.emit("new_emails", {"emails": emails}, namespace="/email")
        return emails
    except HttpError as e:
        logger.error(f"Gmail API error in check_new_emails: {str(e)}")
        return []

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
            generation_config=genai.types.GenerationConfig(max_output_tokens=150, temperature=0.7)
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
            generation_config=genai.types.GenerationConfig(max_output_tokens=120, temperature=0.7)
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
            generation_config=genai.types.GenerationConfig(max_output_tokens=150, temperature=0.7)
        )
        return grammar_tool.correct(response.text.strip())
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"Error: {str(e)}"

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

    return render_template("index.html", output=output, history=history, prefs=prefs, templates=TEMPLATES.keys())

@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"suggestion": "", "status": "error", "message": "No input text provided"}), 400

        if len(text) < 10:
            return jsonify({"suggestion": "", "status": "warning", "message": "Text too short"}), 200

        if text in suggestion_cache:
            return jsonify({"suggestion": suggestion_cache[text], "status": "success", "message": "From cache"}), 200

        prefs = load_user_prefs()
        history = load_email_history(cipher)
        similar_email = find_similar_email(text, history)
        context = f"Similar past email:\n{similar_email}\n\n" if similar_email else ""

        full_prompt = (f"{context}Generate a concise email suggestion based on: {text}\n"
                       f"Use a {prefs['default_tone']} tone and include: {prefs['preferred_phrases'][0]}")
        
        response = model.generate_content(full_prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=80, temperature=0.7))
        suggestion = grammar_tool.correct(response.text.strip())
        suggestion_cache[text] = suggestion
        return jsonify({"suggestion": suggestion, "status": "success", "message": "Generated"}), 200
    except Exception as e:
        logger.error(f"Suggestion error: {e}")
        return jsonify({"suggestion": "", "status": "error", "message": str(e)}), 500

@app.route("/send_email", methods=["POST"])
def send_email_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400

        recipient = data.get("recipient", "").strip()
        subject = data.get("subject", "").strip()
        body = data.get("body", "").strip()

        if not all([recipient, subject, body]):
            return jsonify({"status": "error", "message": "Missing required fields: recipient, subject, or body"}), 400

        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", recipient):
            return jsonify({"status": "error", "message": "Invalid email address"}), 400

        success = send_email(recipient, subject, body)
        if success:
            return jsonify({"status": "success", "message": "Email sent successfully"}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to send email"}), 500
    except HttpError as e:
        return jsonify({"status": "error", "message": f"Gmail API error: {e.reason}"}), 500
    except Exception as e:
        logger.error(f"Send email error: {str(e)}")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@app.route("/check_emails", methods=["GET"])
def check_emails_route():
    try:
        emails = check_new_emails()
        return jsonify({"status": "success", "emails": emails}), 200
    except Exception as e:
        logger.error(f"Check emails error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/oauth2callback")
def oauth2callback():
    return "Authorization complete. You can close this window."

# Background task to poll emails
def poll_emails():
    while True:
        check_new_emails()
        time.sleep(60)

import threading
threading.Thread(target=poll_emails, daemon=True).start()

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)