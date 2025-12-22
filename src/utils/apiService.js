// src/utils/apiService.js
// API service module for communicating with the RAG agent backend

import apiConfig from '../config/apiConfig';

/**
 * Send a query to the RAG agent backend
 * @param {string} selectedText - The text selected by the user
 * @param {string} userQuestion - The question asked by the user
 * @returns {Promise<Object>} - The response from the backend
 */
export const sendQueryToRAG = async (selectedText, userQuestion) => {
  // Combine selected text and user question to form the message for the chat endpoint
  const message = selectedText ? `${selectedText}\n\nQuestion: ${userQuestion}` : userQuestion;

  // Format request payload for the chat endpoint
  const requestBody = {
    message: message,
    top_k: 5,  // Number of documents to retrieve
    temperature: 0.7,  // Response randomness
    conversation_id: null  // Optional conversation ID
  };

  // Debug: Log the request details
  console.log('DEBUG: sendQueryToRAG - Request details');
  console.log('DEBUG: API Base URL:', apiConfig.baseURL);
  console.log('DEBUG: Request body:', requestBody);
  console.log('DEBUG: Full endpoint URL:', `${apiConfig.baseURL}${apiConfig.endpoints.chat}`);

  try {
    // Add timeout to the fetch request
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.log('DEBUG: Request timeout triggered');
      controller.abort();
    }, apiConfig.timeout);

    console.log('DEBUG: Making fetch request...');
    const response = await fetch(`${apiConfig.baseURL}${apiConfig.endpoints.chat}`, {
      method: 'POST',
      headers: {
        ...apiConfig.defaultHeaders,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    console.log('DEBUG: Response received, status:', response.status);

    if (!response.ok) {
      console.log('DEBUG: Response not OK, status:', response.status, 'statusText:', response.statusText);
      throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
    }

    console.log('DEBUG: Parsing response JSON...');
    const data = await response.json();
    console.log('DEBUG: Response data received:', data);

    // Ensure response has proper structure for citations
    return {
      response: data.answer || data.response || data.message || 'No response received',
      citations: data.sources || data.citations || data.references || [],
      metadata: data.metadata || {}
    };
  } catch (error) {
    console.error('DEBUG: Error communicating with RAG agent:', error);
    console.error('DEBUG: Error name:', error.name);
    console.error('DEBUG: Error message:', error.message);
    console.error('DEBUG: Requested URL was:', `${apiConfig.baseURL}${apiConfig.endpoints.chat}`);

    // Handle different types of errors
    if (error.name === 'AbortError') {
      console.error('DEBUG: Request was aborted due to timeout');
      throw new Error('Request timeout: The server took too long to respond');
    } else if (error.message.includes('HTTP error')) {
      console.error('DEBUG: HTTP error occurred');
      throw error; // Re-throw HTTP errors to be handled by the caller
    } else {
      // Network error or other client-side error
      console.error('DEBUG: Network error or client-side error occurred');
      throw new Error('Network error: Unable to connect to the server');
    }
  }
};

/**
 * Test connection to the backend
 * @returns {Promise<boolean>} - True if connection is successful
 */
export const testConnection = async () => {
  console.log('DEBUG: Testing connection to backend at:', `${apiConfig.baseURL}${apiConfig.endpoints.health}`);
  try {
    const response = await fetch(`${apiConfig.baseURL}${apiConfig.endpoints.health}`);
    console.log('DEBUG: Health check response status:', response.status);
    return response.ok;
  } catch (error) {
    console.error('DEBUG: Connection test failed:', error);
    console.error('DEBUG: Failed URL was:', `${apiConfig.baseURL}${apiConfig.endpoints.health}`);
    return false;
  }
};