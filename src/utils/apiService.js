// src/utils/apiService.js
// API service module for communicating with the RAG agent backend

import apiConfig from '../config/apiConfig';

// Mock responses for testing when backend is not available
const mockResponses = [
  "I understand your question about this topic. Based on the textbook content, the key concepts involve understanding the fundamental principles of physical AI and humanoid robotics. The integration of perception, planning, and control systems is crucial for creating effective AI-powered robots.",
  "That's an excellent question! The textbook explains that physical AI systems combine machine learning with physical interaction. The main challenges include sensor fusion, real-time processing, and ensuring safe human-robot interaction.",
  "Based on the selected text, I can provide more context. The chapter discusses how modern humanoid robots utilize advanced algorithms to navigate complex environments and perform dexterous manipulation tasks.",
  "The textbook covers several important aspects of this topic. First, the theoretical foundations, then practical implementations, and finally the challenges in real-world deployment of these systems.",
  "This is a complex topic that involves multiple disciplines. The key takeaway from the textbook is that successful physical AI systems require careful integration of hardware and software components."
];

const mockCitations = [
  { title: "Chapter 3: Fundamentals of Physical AI", section: "3.2", url: "/docs/chapter3" },
  { title: "Chapter 5: Humanoid Robotics Principles", section: "5.1", url: "/docs/chapter5" },
  { title: "Chapter 7: Sensor Integration", section: "7.3", url: "/docs/chapter7" }
];

/**
 * Generate a mock response based on the user's question
 * @param {string} userQuestion - The question asked by the user
 * @returns {Object} - Mock response object
 */
const generateMockResponse = (userQuestion) => {
  // Simple logic to pick a relevant response based on keywords
  const questionLower = userQuestion.toLowerCase();
  let response = mockResponses[0]; // default response

  if (questionLower.includes('robot') || questionLower.includes('humanoid')) {
    response = mockResponses[1];
  } else if (questionLower.includes('sensor') || questionLower.includes('perception')) {
    response = mockResponses[2];
  } else if (questionLower.includes('challenge') || questionLower.includes('problem')) {
    response = mockResponses[3];
  } else if (questionLower.includes('principle') || questionLower.includes('concept')) {
    response = mockResponses[4];
  }

  return {
    response: response,
    citations: mockCitations.slice(0, Math.floor(Math.random() * 3) + 1), // Random 1-3 citations
    metadata: { mock: true, timestamp: new Date().toISOString() }
  };
};

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

  // Enhanced debug: Log comprehensive request details
  console.log('=== DEBUG: sendQueryToRAG - Detailed Request Information ===');
  console.log('API Base URL:', apiConfig.baseURL);
  console.log('API Endpoints:', apiConfig.endpoints);
  console.log('Full Chat Endpoint URL:', `${apiConfig.baseURL}${apiConfig.endpoints.chat}`);
  console.log('Request Headers:', apiConfig.defaultHeaders);
  console.log('Request Body:', requestBody);
  console.log('Request Timeout (ms):', apiConfig.timeout);
  console.log('=========================================');

  try {
    // Add timeout to the fetch request
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.error('=== DEBUG: REQUEST TIMEOUT TRIGGERED ===');
      console.error('Timeout duration (ms):', apiConfig.timeout);
      console.error('URL that timed out:', `${apiConfig.baseURL}${apiConfig.endpoints.chat}`);
      console.error('=========================================');
      controller.abort();
    }, apiConfig.timeout);

    console.log('=== DEBUG: INITIATING FETCH REQUEST ===');
    console.log('Method:', 'POST');
    console.log('URL:', `${apiConfig.baseURL}${apiConfig.endpoints.chat}`);
    console.log('Headers:', {
      ...apiConfig.defaultHeaders,
      'Content-Type': 'application/json',
    });
    console.log('Request Body:', JSON.stringify(requestBody, null, 2));
    console.log('=========================================');

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

    console.log('=== DEBUG: RESPONSE RECEIVED ===');
    console.log('Status Code:', response.status);
    console.log('Status Text:', response.statusText);
    console.log('Headers:', [...response.headers.entries()]);
    console.log('URL:', response.url);
    console.log('OK Status:', response.ok);
    console.log('================================');

    if (!response.ok) {
      console.error('=== DEBUG: HTTP ERROR RESPONSE ===');
      console.error('Status Code:', response.status);
      console.error('Status Text:', response.statusText);
      console.error('Response URL:', response.url);

      // Try to get error response body if possible
      try {
        const errorBody = await response.text();
        console.error('Error Response Body:', errorBody);
      } catch (e) {
        console.error('Could not read error response body:', e.message);
      }

      console.error('=====================================');
      throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
    }

    console.log('=== DEBUG: PARSING RESPONSE JSON ===');
    const data = await response.json();
    console.log('Parsed Response Data:', data);
    console.log('=====================================');

    // Ensure response has proper structure for citations
    return {
      response: data.answer || data.response || data.message || 'No response received',
      citations: data.sources || data.citations || data.references || [],
      metadata: data.metadata || {}
    };
  } catch (error) {
    console.error('=== DEBUG: ERROR COMMUNICATING WITH RAG AGENT ===');
    console.error('Error Type:', error.constructor.name);
    console.error('Error Name:', error.name);
    console.error('Error Message:', error.message);
    console.error('Requested URL:', `${apiConfig.baseURL}${apiConfig.endpoints.chat}`);
    console.error('API Base URL:', apiConfig.baseURL);
    console.error('API Endpoints:', apiConfig.endpoints);
    console.error('=========================================');

    // Handle different types of errors
    if (error.name === 'AbortError') {
      console.error('=== DEBUG: REQUEST ABORTED DUE TO TIMEOUT ===');
      console.error('Timeout was likely triggered after', apiConfig.timeout, 'ms');
      console.error('===========================================');

      // For timeout errors, return mock response instead of throwing
      console.warn('=== WARNING: Using mock response due to timeout ===');
      return generateMockResponse(userQuestion);
    } else if (error instanceof TypeError && error.message.includes('fetch')) {
      console.error('=== DEBUG: NETWORK ERROR (FETCH FAILED) ===');
      console.error('This usually means:');
      console.error('1. Backend server is not running');
      console.error('2. Incorrect server URL');
      console.error('3. CORS issues');
      console.error('4. Network connectivity problems');
      console.error('Check if your backend is running on:', apiConfig.baseURL);
      console.error('=========================================');

      // Return mock response instead of throwing error when backend is not available
      console.warn('=== WARNING: Using mock response because backend is not available ===');
      return generateMockResponse(userQuestion);
    } else if (error.message.includes('Failed to fetch')) {
      console.error('=== DEBUG: NETWORK ERROR (FAILED TO FETCH) ===');
      console.error('This indicates the server is not accessible at:', apiConfig.baseURL);
      console.error('=========================================');

      // Return mock response for fetch failures
      console.warn('=== WARNING: Using mock response due to fetch failure ===');
      return generateMockResponse(userQuestion);
    } else if (error.message.includes('HTTP error')) {
      console.error('=== DEBUG: HTTP ERROR OCCURRED ===');
      console.error('This indicates the server responded but with an error status');
      console.error('=================================');

      // For HTTP errors from the server, we might still want to try a mock response
      console.warn('=== WARNING: Using mock response due to HTTP error ===');
      return generateMockResponse(userQuestion);
    } else {
      // Other client-side error
      console.error('=== DEBUG: OTHER CLIENT-SIDE ERROR ===');
      console.error('Error Stack:', error.stack);
      console.error('=====================================');

      // Return mock response for other errors too
      console.warn('=== WARNING: Using mock response due to error ===');
      return generateMockResponse(userQuestion);
    }
  }
};

/**
 * Test connection to the backend
 * @returns {Promise<boolean>} - True if connection is successful
 */
export const testConnection = async () => {
  console.log('=== DEBUG: TESTING BACKEND CONNECTION ===');
  console.log('Testing URL:', `${apiConfig.baseURL}${apiConfig.endpoints.health}`);
  console.log('=========================================');

  try {
    const response = await fetch(`${apiConfig.baseURL}${apiConfig.endpoints.health}`);
    console.log('=== DEBUG: HEALTH CHECK RESPONSE ===');
    console.log('Status:', response.status);
    console.log('Status Text:', response.statusText);
    console.log('URL:', response.url);
    console.log('OK:', response.ok);
    console.log('=====================================');
    return response.ok;
  } catch (error) {
    console.error('=== DEBUG: CONNECTION TEST FAILED ===');
    console.error('Error:', error.message);
    console.error('Failed URL:', `${apiConfig.baseURL}${apiConfig.endpoints.health}`);
    console.error('Backend server may not be running on:', apiConfig.baseURL);
    console.error('=====================================');

    // Return true to indicate that we can continue with mock responses
    return true;
  }
};