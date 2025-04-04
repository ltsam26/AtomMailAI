from transformers import pipeline

# Load a better model
generator = pipeline("text-generation", model="gpt2-medium")

def generate_email(prompt):
    result = generator(prompt, max_length=120, num_return_sequences=1, temperature=0.7)
    raw_text = result[0]['generated_text']
    email = f"Subject: Apology for Delay\nDear Sarah,\n{raw_text}\nBest regards,\n[Your Name]"
    return email

def suggest_reply(incoming_email):
    prompt = f"Reply to this email in a polite tone: {incoming_email}"
    result = generator(prompt, max_length=100, num_return_sequences=1, temperature=0.7)
    raw_text = result[0]['generated_text']
    reply = f"Subject: Re: Your Email\nDear Sender,\n{raw_text}\nRegards,\n[Your Name]"
    return reply

def refine_text(text, action="shorten"):
    if action == "shorten":
        sentences = text.split(". ")
        short_text = ". ".join(sentences[:2]) + "."
        return short_text
    return text

# Interactive loop
print("Welcome to Atom Mail AI! Type 'quit' to exit.")
while True:
    choice = input("What do you want to do? (compose/reply/refine): ")
    if choice == "quit":
        break
    elif choice == "compose":
        prompt = input("Enter your prompt (e.g., 'Write a formal apology email'): ")
        draft = generate_email(prompt)
        print("\nGenerated Draft:")
        print(draft)
    elif choice == "reply":
        incoming = input("Enter the incoming email: ")
        reply = suggest_reply(incoming)
        print("\nSuggested Reply:")
        print(reply)
    elif choice == "refine":
        text = input("Enter text to refine: ")
        refined = refine_text(text)
        print("\nRefined Text:")
        print(refined)
    else:
        print("Invalid option. Try 'compose', 'reply', or 'refine'.")