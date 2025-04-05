import google.generativeai as genai
import os
import re

# Configure the API key 
GOOGLE_API_KEY = "AIzaSyBgc6ZxaiTsnFli9HB9-nYXhnGuBhSrsAc"  
genai.configure(api_key=GOOGLE_API_KEY)


model = genai.GenerativeModel("gemini-1.5-flash")


HISTORY_FILE = "email_history.txt"

def save_email_to_history(email):
    """Save an email to the history file."""
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(email + "\n---\n")

def load_email_history():
    """Load the email history from the file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def extract_subject(prompt):
    """Extract subject from prompt if provided."""
    match = re.search(r"subject\s*['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
    return match.group(1) if match else None

def generate_email(prompt):
    history = load_email_history()
    if history:
        full_prompt = f"Based on this past email history:\n{history}\n\nNow, {prompt}"
    else:
        full_prompt = prompt
    
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=120,
            temperature=0.7,
        )
    )
    raw_text = response.text

    subject = extract_subject(prompt)
    if not subject:
        lines = raw_text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Subject:"):
                subject = line.replace("Subject:", "").strip()
                raw_text = "\n".join(lines[:i] + lines[i+1:]).strip()
                break
        if not subject:
            subject = "Untitled Email"

    email = f"Subject: {subject}\nDear Sarah,\n{raw_text}\nBest regards,\n[Your Name]"
    save_email_to_history(email)
    return email

def suggest_reply(incoming_email):
    history = load_email_history()
    if history:
        full_prompt = f"Based on this past email history:\n{history}\n\nReply to this email in a polite tone: {incoming_email}"
    else:
        full_prompt = f"Reply to this email in a polite tone: {incoming_email}"
    
    subject = extract_subject(incoming_email) or "Your Email"
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=100,
            temperature=0.7,
        )
    )
    raw_text = response.text
    reply = f"Subject: Re: {subject}\nDear Sender,\n{raw_text}\nRegards,\n[Your Name]"
    save_email_to_history(reply)
    return reply

def refine_text(text, action="shorten"):
    
    action_prompts = {
        "shorten": "Shorten this text while preserving its core meaning",
        "improve": "Enhance the clarity, grammar, and flow of this text",
        "formalize": "Rewrite this text in a formal, polite tone",
        "casual": "Rewrite this text in a relaxed, casual tone",
        "professional": "Rewrite this text in a professional, business-like tone"
    }
    
    # Handle combined actions 
    actions = action.lower().split(" and ")
    current_text = text
    
    for act in actions:
        act = act.strip()
        if act in action_prompts:
            prompt = f"{action_prompts[act]}:\n\n{current_text}"
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.6,
                )
            )
            current_text = response.text.strip()
        else:
            print(f"Warning: '{act}' is not a valid refinement action. Skipping.")
    
    return current_text

# Interactive loop
print("Welcome to Atom Mail AI! Type 'quit' to exit.")
while True:
    choice = input("What do you want to do? (compose/reply/refine): ")
    if choice == "quit":
        break
    elif choice == "compose":
        prompt = input("Enter your prompt (e.g., 'Write an email with subject \"Meeting Update\" about...'): ")
        draft = generate_email(prompt)
        print("\nGenerated Draft:")
        print(draft)
    elif choice == "reply":
        incoming = input("Enter the incoming email (e.g., 'Subject \"Project Deadline\"...'): ")
        reply = suggest_reply(incoming)
        print("\nSuggested Reply:")
        print(reply)
    elif choice == "refine":
        text = input("Enter text to refine: ")
        print("Refinement options: shorten / improve / formalize / casual / professional (combine with 'and', e.g., 'shorten and formalize')")
        action = input("Choose refinement action: ")
        refined = refine_text(text, action)
        print("\nRefined Text:")
        print(refined)
    else:
        print("Invalid option. Try 'compose', 'reply', or 'refine'.")