// Chat functionality
let isProcessing = false;

// DOM elements
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');
const buttonText = document.getElementById('buttonText');
const loadingSpinner = document.getElementById('loadingSpinner');

// Event listeners
messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !isProcessing) {
        sendMessage();
    }
});

// Prevent form submission on Enter
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        e.preventDefault();
    }
});

// Auto-resize input based on content
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isProcessing) {
        return;
    }
    
    // Set processing state
    isProcessing = true;
    setButtonLoading(true);
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    try {
        // Send request to backend
        console.log('Sending message:', message);
        
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Response error:', errorText);
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        console.log('Response data:', data);
        
        // Add bot response to chat
        addMessage(data.answer, 'bot');
        
    } catch (error) {
        console.error('Error:', error);
        
        // Add error message to chat
        const errorMessage = `❌ Sorry, I encountered an error: ${error.message}. Please try again.`;
        addMessage(errorMessage, 'bot', true);
    } finally {
        // Reset processing state
        isProcessing = false;
        setButtonLoading(false);
        messageInput.focus();
    }
}

function addMessage(content, sender, isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message${isError ? ' error-message' : ''}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (sender === 'bot') {
        messageContent.innerHTML = `<strong>🤖 Medical Bot:</strong><br><br>${formatMessage(content)}`;
    } else {
        messageContent.innerHTML = `<strong>👤 You:</strong><br><br>${formatMessage(content)}`;
    }
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Add animation
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(20px)';
    
    requestAnimationFrame(() => {
        messageDiv.style.transition = 'all 0.3s ease-out';
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
    });
}

function formatMessage(message) {
    // Convert newlines to <br> tags
    return message
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

function setButtonLoading(loading) {
    if (loading) {
        buttonText.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        sendButton.disabled = true;
        sendButton.style.cursor = 'not-allowed';
    } else {
        buttonText.classList.remove('hidden');
        loadingSpinner.classList.add('hidden');
        sendButton.disabled = false;
        sendButton.style.cursor = 'pointer';
    }
}

// Initialize chat
document.addEventListener('DOMContentLoaded', function() {
    messageInput.focus();
    console.log('Chat interface initialized');
    
    // Add a small delay to ensure the page is fully loaded
    setTimeout(() => {
        console.log('Page fully loaded, ready for interactions');
    }, 100);
});

// Handle window focus
window.addEventListener('focus', function() {
    if (!isProcessing) {
        messageInput.focus();
    }
});

// Error handling for uncaught errors
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
});