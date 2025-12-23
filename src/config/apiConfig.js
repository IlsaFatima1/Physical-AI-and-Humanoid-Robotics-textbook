// src/config/apiConfig.js
// Configuration for backend API endpoints

// In Docusaurus, we'll use a global variable that can be set during build time
// This approach allows for environment variable configuration while avoiding process.env issues
const getBaseURL = () => {
  // Check for environment variable in browser context
  if (typeof window !== 'undefined') {
    // This will be replaced during Docusaurus build with actual environment variable values
    // Check for a global variable that might be set by Docusaurus or build process
    if (window.DOCUSAURUS_ENV && window.DOCUSAURUS_ENV.REACT_APP_API_BASE_URL) {
      return window.DOCUSAURUS_ENV.REACT_APP_API_BASE_URL;
    }
    // Fallback to a global variable that might be set in index.html or build
    if (window.APP_CONFIG && window.APP_CONFIG.API_BASE_URL) {
      return window.APP_CONFIG.API_BASE_URL;
    }
  }

  // Default fallback
  return 'http://localhost:8000';
};

const apiConfig = {
  // Base URL for the RAG agent backend
  baseURL: getBaseURL(),

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