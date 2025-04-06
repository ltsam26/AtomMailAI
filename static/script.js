// static/script.js

// Theme toggle functionality
function initializeThemeToggle() {
    const toggleButton = document.getElementById("theme-toggle");
    const icon = toggleButton.querySelector(".material-icons");
    const text = toggleButton.querySelector(".toggle-text");

    toggleButton.addEventListener("click", () => {
        const html = document.documentElement;
        const currentTheme = html.getAttribute("data-theme");
        if (currentTheme === "dark") {
            html.setAttribute("data-theme", "light");
            icon.textContent = "light_mode";
            text.textContent = "Switch to Dark Theme";
        } else {
            html.setAttribute("data-theme", "dark");
            icon.textContent = "dark_mode";
            text.textContent = "Switch to Light Theme";
        }
    });
}

// Toggle refine options
function toggleRefineOptions(show) {
    document.getElementById("refine-options").style.display = show ? "block" : "none";
}

// Throttling constants
const SUGGESTION_COOLDOWN = 2000; // 2 seconds
const SEND_COOLDOWN = 3000; // 3 seconds
let lastSuggestionTime = 0;
let lastSendTime = 0;

// Suggestion functionality
function getSuggestion() {
    const text = document.getElementById("input-text").value;
    const suggestionDiv = document.getElementById("suggestion");
    const loadingDiv = document.getElementById("loading");
    const suggestBtn = document.getElementById("suggest-btn");
    const currentTime = Date.now();

    if (currentTime - lastSuggestionTime < SUGGESTION_COOLDOWN) {
        suggestionDiv.innerHTML = `<span class="error">Please wait ${Math.ceil((SUGGESTION_COOLDOWN - (currentTime - lastSuggestionTime)) / 1000)} seconds</span>`;
        return;
    }

    loadingDiv.style.display = "block";
    suggestionDiv.innerHTML = "";
    suggestBtn.disabled = true;

    fetch("/suggest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    })
    .then(response => response.json())
    .then(data => {
        loadingDiv.style.display = "none";
        suggestBtn.disabled = false;
        lastSuggestionTime = Date.now();

        if (data.status === "success") {
            suggestionDiv.innerHTML = `<strong>Suggestion (${data.source}):</strong> ${data.suggestion}`;
            const lines = data.suggestion.split("\n");
            for (const line of lines) {
                if (line.startsWith("To:")) {
                    document.getElementById("email-recipient").value = line.replace("To:", "").trim();
                } else if (line.startsWith("Subject:")) {
                    document.getElementById("email-subject").value = line.replace("Subject:", "").trim();
                }
            }
        } else {
            suggestionDiv.innerHTML = `<span class="error">${data.message}</span>`;
        }
    })
    .catch(error => {
        loadingDiv.style.display = "none";
        suggestBtn.disabled = false;
        suggestionDiv.innerHTML = `<span class="error">Error: ${error.message}</span>`;
    });
}

// Send Email functionality (combined version)
function sendEmail() {
    const recipientInput = document.getElementById("email-recipient") || document.getElementById("recipient");
    const subjectInput = document.getElementById("email-subject") || document.getElementById("subject");
    const bodyInput = document.getElementById("email-body");
    const output = document.getElementById("output");
    const sendBtn = document.getElementById("send-email-btn");
    const statusDiv = document.getElementById("email-status");
    const currentTime = Date.now();

    if (currentTime - lastSendTime < SEND_COOLDOWN) {
        statusDiv.textContent = `Please wait ${Math.ceil((SEND_COOLDOWN - (currentTime - lastSendTime)) / 1000)} seconds`;
        statusDiv.className = "status-message error";
        return;
    }

    let recipient = recipientInput.value.trim();
    let subject = subjectInput.value.trim();
    let body = bodyInput ? bodyInput.value.trim() : output.textContent.trim();

    // Fallback to parsing output if fields are empty
    if (!recipient || !subject || !body) {
        const lines = output.textContent.split("\n");
        for (const line of lines) {
            if (line.startsWith("To:") && !recipient) {
                recipient = line.replace("To:", "").trim();
            } else if (line.startsWith("Subject:") && !subject) {
                subject = line.replace("Subject:", "").trim();
            } else if (!body && line.trim() && !line.startsWith("To:") && !line.startsWith("Subject:")) {
                body = lines.slice(lines.indexOf(line)).join("\n").trim();
            }
        }
    }

    if (!recipient || !subject || !body) {
        statusDiv.textContent = "Please provide recipient, subject, and body.";
        statusDiv.className = "status-message error";
        return;
    }

    sendBtn.disabled = true;
    statusDiv.textContent = "Sending...";
    statusDiv.className = "status-message";

    fetch("/send_email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recipient, subject, body })
    })
    .then(response => response.json())
    .then(data => {
        sendBtn.disabled = false;
        lastSendTime = Date.now();
        statusDiv.textContent = data.message;
        statusDiv.className = `status-message ${data.status === "success" ? "success" : "error"}`;
        if (data.status === "success") {
            setTimeout(() => { statusDiv.textContent = ""; }, 5000);
            alert("Email sent successfully!");
        } else {
            alert(`Error: ${data.message}`);
        }
    })
    .catch(error => {
        sendBtn.disabled = false;
        statusDiv.textContent = `Error: ${error.message}`;
        statusDiv.className = "status-message error";
        console.error("Send email error:", error);
    });
}

// Copy output functionality
function initializeCopyButton() {
    document.getElementById("copy-output").addEventListener("click", () => {
        const output = document.getElementById("output");
        navigator.clipboard.writeText(output.textContent);
        const btn = document.getElementById("copy-output");
        btn.innerHTML = '<span class="material-icons">check</span> Copied!';
        setTimeout(() => {
            btn.innerHTML = '<span class="material-icons">content_copy</span> Copy';
        }, 2000);
    });
}

// History filter functionality
function filterHistory() {
    const search = document.getElementById("search-input").value.toLowerCase();
    const emails = document.getElementsByClassName("email");
    Array.from(emails).forEach(email => {
        const subject = email.querySelector(".email-subject").textContent.toLowerCase();
        const content = email.querySelector(".email-preview").textContent.toLowerCase();
        email.style.display = (subject.includes(search) || content.includes(search)) ? "block" : "none";
    });
}

// Display emails in inbox
function displayEmails(emails) {
    const emailList = document.getElementById("email-list");
    emailList.innerHTML = "";
    emails.forEach(email => {
        const li = document.createElement("li");
        li.innerHTML = `<strong>${email.subject}</strong> from ${email.from}<br>${email.snippet || email.body || "No preview available"}`;
        emailList.appendChild(li);
    });
}

// Initialize all event listeners
document.addEventListener("DOMContentLoaded", () => {
    initializeThemeToggle();
    initializeCopyButton();

    // Event listeners for buttons and inputs
    document.getElementById("suggest-btn").addEventListener("click", getSuggestion);
    document.getElementById("send-email-btn").addEventListener("click", sendEmail);
    document.getElementById("search-input").addEventListener("keyup", filterHistory);

    // Radio button listeners for refine options
    document.querySelectorAll('input[name="action"]').forEach(radio => {
        radio.addEventListener("change", (e) => {
            toggleRefineOptions(e.target.value === "refine");
        });
    });

    // Auto-toggle refine options based on initial radio state
    const selectedAction = document.querySelector('input[name="action"]:checked');
    if (selectedAction) {
        toggleRefineOptions(selectedAction.value === "refine");
    }

    // Socket.IO for real-time email updates
    const socket = io.connect("http://localhost:5000", { path: "/socket.io", transports: ["websocket"] });
    socket.on("connect", () => {
        console.log("Connected to WebSocket server");
    });
    socket.on("new_emails", (data) => {
        displayEmails(data.emails);
    });

    // Initial email fetch with polling fallback
    function fetchEmails() {
        fetch("/check_emails")
            .then(response => response.json())
            .then(data => {
                if (data.status === "success") {
                    displayEmails(data.emails);
                }
            })
            .catch(error => console.error("Error fetching emails:", error));
    }
    fetchEmails(); // Initial fetch
    setInterval(fetchEmails, 30000); // Poll every 30 seconds as fallback
});