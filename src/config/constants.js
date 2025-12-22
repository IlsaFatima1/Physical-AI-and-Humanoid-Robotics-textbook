// src/config/constants.js
// Constants for the chat interface

const CHAT_CONSTANTS = {
  // API endpoints
  ENDPOINTS: {
    CHAT: '/chat',
    HEALTH: '/health',
    EMBEDDINGS: '/embeddings',
  },

  // UI constants
  UI: {
    MAX_MESSAGE_LENGTH: 2000,
    INPUT_PLACEHOLDER: 'Ask a question about the selected text...',
    LOADING_MESSAGE: 'Thinking...',
    ERROR_MESSAGE: 'Sorry, I encountered an error. Please try again.',
  },

  // Message types
  MESSAGE_TYPES: {
    USER: 'user',
    AI: 'ai',
    SYSTEM: 'system',
  },

  // Storage keys
  STORAGE: {
    CHAT_HISTORY: 'chatHistory',
    SELECTED_TEXT: 'selectedText',
  },
};

export default CHAT_CONSTANTS;