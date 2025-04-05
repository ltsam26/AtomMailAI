import google.generativeai as genai
import os
import re
import json
from flask import Flask, request, render_template, jsonify
from cryptography.fernet import Fernet
import language_tool_python
from datetime import datetime

# Configure the API key
GOOGLE_API_KEY = "AIzaSyCxhk2hszcNcSzY-Twsa_A5RiJTkmbi580"  # Replace with your key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Encryption setup
key = b'w6-1ePqZNLFMxJk6EWZeDnyXSeh8Kunwmu-N6QREkWg='  # Replace with your key
cipher = Fernet(key)

# File paths
HISTORY_FILE = "email_history.enc"
PREFS_FILE = "user_prefs.json"
FEEDBACK_FILE = "feedback.json"

# Initialize Flask app
app = Flask(__name__)

# Grammar checker
grammar_tool = language_tool_python.LanguageTool('en-US')

# Templates
TEMPLATES = {
    "apology": "Write an email apologizing for a delay in {context}.",
    "follow-up": "Write a follow-up email about {context}.",
    "request": "Write an email requesting {context}."
}

def save_email_to_history(email):
    try:
        encrypted_email = cipher.encrypt(email.encode('utf-8'))
        with open(HISTORY_FILE, "ab") as f:
            f.write(encrypted_email + b"\n---\n")
    except Exception as e:
        print(f"Error saving email history: {e}")

def load_email_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "rb") as f:
            lines = f.read().split(b"\n---\n")
            decrypted = []
            for line in lines:
                if line:
                    try:
                        decrypted.append(cipher.decrypt(line).decode('utf-8'))
                    except Exception as e:
                        print(f"Decryption error: {e}")
            return decrypted
    return []

def clear_email_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def load_user_prefs():
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f:
            return json.load(f)
    return {"default_tone": "professional", "default_recipient": "Sarah"}

def save_user_prefs(prefs):
    with open(PREFS_FILE, "w") as f:
        json.dump(prefs, f)

def save_feedback(prompt, output, rating):
    feedback = {"prompt": prompt, "output": output, "rating": rating, "timestamp": str(datetime.now())}
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(feedback)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f)

def extract_subject(prompt):
    match = re.search(r"subject\s*['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
    return match.group(1) if match else None

def generate_email(prompt, tone=None, recipient=None, template=None):
    prefs = load_user_prefs()
    tone = tone or prefs["default_tone"]
    recipient = recipient or prefs["default_recipient"]
    history = "\n".join(load_email_history())
    if template and template in TEMPLATES:
        full_prompt = TEMPLATES[template].format(context=prompt)
    else:
        full_prompt = prompt
    full_prompt = f"Based on this past email history:\n{history}\n\nWrite an email in a {tone} tone to {recipient}: {full_prompt}"
    
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=120, temperature=0.7)
    )
    raw_text = response.text

    subject = extract_subject(prompt) or "Untitled Email"
    if "Subject:" in raw_text:
        lines = raw_text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Subject:"):
                subject = line.replace("Subject:", "").strip()
                raw_text = "\n".join(lines[:i] + lines[i+1:]).strip()
                break

    corrected_text = grammar_tool.correct(raw_text)
    email = f"Subject: {subject}\nDear {recipient},\n{corrected_text}\nBest regards,\n[Your Name]"
    save_email_to_history(email)
    return email

def suggest_reply(incoming_email, tone=None):
    prefs = load_user_prefs()
    tone = tone or prefs["default_tone"]
    history = "\n".join(load_email_history())
    full_prompt = f"Based on this past email history:\n{history}\n\nReply to this email in a {tone} tone: {incoming_email}"
    
    subject = extract_subject(incoming_email) or "Your Email"
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=100, temperature=0.7)
    )
    raw_text = response.text

    corrected_text = grammar_tool.correct(raw_text)
    reply = f"Subject: Re: {subject}\nDear Sender,\n{corrected_text}\nRegards,\n[Your Name]"
    save_email_to_history(reply)
    return reply

def refine_text(text, action="shorten"):
    action_prompts = {
        "shorten": "Shorten this text while preserving its core meaning",
        "improve": "Enhance the clarity, grammar, and flow of this text",
        "formalize": "Rewrite this text in a formal, polite tone",
        "casual": "Rewrite this text in a relaxed, casual tone",
        "professional": "Rewrite this text in a professional, business-like tone",
        "technical": "Rewrite this text in a technical, precise tone",
        "friendly": "Rewrite this text in a warm, friendly tone"
    }
    
    actions = action.lower().split(" and ")
    current_text = text
    
    for act in actions:
        act = act.strip()
        if act in action_prompts:
            prompt = f"{action_prompts[act]}:\n\n{current_text}"
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=150, temperature=0.6)
            )
            current_text = grammar_tool.correct(response.text.strip())
        else:
            current_text = f"Error: '{act}' is not a valid refinement action."
            break
    
    return current_text

@app.route("/", methods=["GET", "POST"])
def home():
    output = ""
    history = load_email_history()
    prefs = load_user_prefs()
    
    if request.method == "POST":
        action = request.form["action"]
        input_text = request.form.get("input_text", "")
        
        if action == "compose":
            template = request.form.get("template")
            output = generate_email(input_text, template=template)
        elif action == "reply":
            output = suggest_reply(input_text)
        elif action == "refine":
            refine_action = request.form["refine_action"]
            output = refine_text(input_text, refine_action)
        elif action == "clear_history":
            clear_email_history()
            output = "Email history cleared."
        elif action == "save_prefs":
            prefs["default_tone"] = request.form["default_tone"]
            prefs["default_recipient"] = request.form["default_recipient"]
            save_user_prefs(prefs)
            output = "Preferences saved."
        elif action == "feedback":
            rating = request.form["rating"]
            save_feedback(input_text, output, rating)
            output = "Feedback submitted. Thank you!"
    
    return render_template("index.html", output=output, history=history, prefs=prefs, templates=TEMPLATES.keys())

@app.route("/suggest", methods=["POST"])
def suggest():
    text = request.json.get("text", "")
    if text:
        suggestion = model.generate_content(f"Suggest a continuation for: {text}", max_output_tokens=50).text
        return jsonify({"suggestion": suggestion})
    return jsonify({"suggestion": ""})

if __name__ == "__main__":
    app.run(debug=True)