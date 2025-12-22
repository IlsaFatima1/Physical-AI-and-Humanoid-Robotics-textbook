// src/config/apiConfig.js
// Configuration for backend API endpoints

const apiConfig = {
  // Base URL for the RAG agent backend
  baseURL: typeof window !== 'undefined' ?
    (window._env_?.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000') :
    'http://localhost:8000',

  // API endpoints
  endpoints: {
    chat: '/chat',
    health: '/health',
    embeddings: '/embeddings',
  },

  // Default request timeout in milliseconds
  timeout: 30000, // 30 seconds

  // Default headers
  defaultHeaders: {
    'Content-Type': 'application/json',
  },

  // Maximum text length allowed for requests
  maxTextLength: 2000,
};

export default apiConfig;