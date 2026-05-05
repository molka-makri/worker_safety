// ━━━ CHATBOT LOGIC ━━━

const chatbotToggle = document.getElementById('chatbot-toggle');
const chatbotWindow = document.getElementById('chatbot-window');
const chatbotClose = document.getElementById('chatbot-close');
const chatbotInput = document.getElementById('chatbot-input');
const chatbotSend = document.getElementById('chatbot-send');
const chatbotMessages = document.getElementById('chatbot-messages');

// Ouvrir/Fermer le chat
chatbotToggle.addEventListener('click', () => {
  chatbotWindow.classList.toggle('hidden');
});

chatbotClose.addEventListener('click', () => {
  chatbotWindow.classList.add('hidden');
});

// Envoyer un message
function addMessage(text, sender) {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('chat-message', sender);
  msgDiv.textContent = text;
  chatbotMessages.appendChild(msgDiv);
  chatbotMessages.scrollTop = chatbotMessages.scrollHeight; // Scroll vers le bas
}

function showTyping() {
  const typingDiv = document.createElement('div');
  typingDiv.classList.add('typing-indicator');
  typingDiv.id = 'typing-indicator';
  typingDiv.innerHTML = '<span></span><span></span><span></span>';
  chatbotMessages.appendChild(typingDiv);
  chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
}

function removeTyping() {
  const typing = document.getElementById('typing-indicator');
  if (typing) typing.remove();
}

async function sendChatMessage() {
  const text = chatbotInput.value.trim();
  if (!text) return;

  // Afficher le message de l'utilisateur
  addMessage(text, 'user');
  chatbotInput.value = '';

  // Afficher l'indicateur de typing
  showTyping();

  try {
    const res = await fetch('/api/chat/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken') // Utilise la fonction existante dans main.js
      },
      body: JSON.stringify({ message: text })
    });

    const data = await res.json();
    
    removeTyping(); // Enlever le loader

    if (data.status === 'success') {
      addMessage(data.reply, 'bot');
    } else {
      // On affiche le VRAI message d'erreur de Django pour pouvoir débugger
      addMessage("ERREUR TECHNIQUE : " + (data.message || 'Erreur inconnue'), 'bot');
      console.error('[SafeBot] Error:', data.message);
    }
  } catch (err) {
    removeTyping();
    addMessage("Erreur de connexion au serveur.", 'bot');
    console.error('[SafeBot] Fetch error:', err);
  }
}

chatbotSend.addEventListener('click', sendChatMessage);
chatbotInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') sendChatMessage();
});