// This file will be used to inject environment variables into the client-side
// It will be loaded as a client module in Docusaurus

// Define environment variables - these will be replaced at build time by Docusaurus
// when environment variables are properly configured
const ENV = {
  REACT_APP_API_BASE_URL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
};

// Inject environment variables into window object
if (typeof window !== 'undefined') {
  window.__APP_ENV__ = window.__APP_ENV__ || ENV;
}