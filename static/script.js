// DOM Elements
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const imageInput = document.getElementById('imageInput');
const imageButton = document.getElementById('imageButton');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeImageBtn = document.getElementById('removeImage');
const loadingIndicator = document.getElementById('loadingIndicator');

// State variables
let selectedImage = null;
let isProcessing = false;

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    adjustTextareaHeight();
});

function initializeEventListeners() {
    // Form submission
    chatForm.addEventListener('submit', handleFormSubmit);
    
    // Image button click
    imageButton.addEventListener('click', () => {
        imageInput.click();
    });
    
    // Image input change
    imageInput.addEventListener('change', handleImageSelect);
    
    // Remove image button
    removeImageBtn.addEventListener('click', removeImage);
    
    // Textarea auto-resize
    messageInput.addEventListener('input', adjustTextareaHeight);
    
    // Enter key handling (Ctrl+Enter or Shift+Enter to send)
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && (e.ctrlKey || e.shiftKey)) {
            e.preventDefault();
            if (!isProcessing && messageInput.value.trim()) {
                handleFormSubmit(e);
            }
        }
    });
}

async function handleFormSubmit(e) {
    e.preventDefault();
    
    if (isProcessing) return;
    
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, 'user', selectedImage);
    
    // Clear input and image
    messageInput.value = '';
    const imageToSend = selectedImage;
    removeImage();
    adjustTextareaHeight();
    
    // Show loading
    setProcessing(true);
    
    try {
        let response;
        
        if (imageToSend) {
            // Send to image endpoint if image is present
            response = await sendImageMessage(message, imageToSend);
        } else {
            // Send to text endpoint if no image
            response = await sendTextMessage(message);
        }
        
        // Add bot response to chat
        addMessage(response.answer, 'bot');
        
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, I encountered an error while processing your request. Please try again.', 'bot');
    } finally {
        setProcessing(false);
    }
}

async function sendTextMessage(message) {
    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

async function sendImageMessage(message, imageFile) {
    const formData = new FormData();
    formData.append('message', message);
    formData.append('image', imageFile);
    
    const response = await fetch('/chat/mixed', {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

function handleImageSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
    }
    
    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
        alert('Image file is too large. Please select an image smaller than 10MB.');
        return;
    }
    
    selectedImage = file;
    showImagePreview(file);
}

function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        imageButton.classList.add('has-image');
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    selectedImage = null;
    imagePreview.style.display = 'none';
    imageButton.classList.remove('has-image');
    imageInput.value = '';
}

function addMessage(text, sender, image = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const icon = document.createElement('i');
    icon.className = `fas ${sender === 'user' ? 'fa-user' : 'fa-robot'} message-icon`;
    
    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    
    // Format the text content
    if (typeof text === 'string') {
        // Convert line breaks to paragraphs
        const paragraphs = text.split('\n').filter(p => p.trim());
        if (paragraphs.length > 1) {
            paragraphs.forEach(paragraph => {
                const p = document.createElement('p');
                p.textContent = paragraph.trim();
                messageText.appendChild(p);
            });
        } else {
            const p = document.createElement('p');
            p.textContent = text;
            messageText.appendChild(p);
        }
    }
    
    // Add image if present (for user messages)
    if (image && sender === 'user') {
        const imageElement = document.createElement('img');
        imageElement.className = 'message-image';
        imageElement.src = URL.createObjectURL(image);
        imageElement.alt = 'Uploaded image';
        messageText.appendChild(imageElement);
    }
    
    messageContent.appendChild(icon);
    messageContent.appendChild(messageText);
    messageDiv.appendChild(messageContent);
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    scrollToBottom();
}

function adjustTextareaHeight() {
    messageInput.style.height = 'auto';
    const newHeight = Math.min(messageInput.scrollHeight, 120);
    messageInput.style.height = newHeight + 'px';
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setProcessing(processing) {
    isProcessing = processing;
    sendButton.disabled = processing;
    
    if (processing) {
        loadingIndicator.style.display = 'block';
        sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    } else {
        loadingIndicator.style.display = 'none';
        sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

// Utility function to format bot responses
function formatBotResponse(text) {
    // Handle lists, bold text, etc.
    let formatted = text;
    
    // Convert **bold** to <strong>
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert *italic* to <em>
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert numbered lists
    formatted = formatted.replace(/^\d+\.\s(.*)$/gm, '<li>$1</li>');
    
    // Convert bullet points
    formatted = formatted.replace(/^[-•]\s(.*)$/gm, '<li>$1</li>');
    
    return formatted;
}

// Handle drag and drop for images
chatMessages.addEventListener('dragover', function(e) {
    e.preventDefault();
    e.stopPropagation();
    chatMessages.style.backgroundColor = 'rgba(102, 126, 234, 0.1)';
});

chatMessages.addEventListener('dragleave', function(e) {
    e.preventDefault();
    e.stopPropagation();
    chatMessages.style.backgroundColor = '';
});

chatMessages.addEventListener('drop', function(e) {
    e.preventDefault();
    e.stopPropagation();
    chatMessages.style.backgroundColor = '';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            selectedImage = file;
            showImagePreview(file);
        }
    }
});

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    document.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}