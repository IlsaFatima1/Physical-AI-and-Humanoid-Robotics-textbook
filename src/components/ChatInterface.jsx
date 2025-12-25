// src/components/ChatInterface.jsx
// Main chat component for the RAG chatbot integration

import React, { useState, useEffect, useRef } from 'react';
import apiConfig from '../config/apiConfig';
import CHAT_CONSTANTS from '../config/constants';
import { sendQueryToRAG } from '../utils/apiService';

const ChatInterface = () => {
  // State management
  const [isOpen, setIsOpen] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [userQuestion, setUserQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Refs
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Function to handle text selection with cross-browser compatibility
  const handleTextSelection = () => {
    let selectedText = '';

    // Cross-browser text selection
    if (window.getSelection) {
      selectedText = window.getSelection().toString().trim();
    } else if (document.selection && document.selection.type !== 'Control') {
      // For older IE versions
      selectedText = document.selection.createRange().text.trim();
    }

    if (selectedText) {
      // Sanitize and handle special characters in selected text
      // For now, we'll just ensure it's properly handled by the JSON.stringify in the API call
      // The selected text may contain special characters, code snippets, etc.
      const sanitizedText = selectedText; // In a real implementation, you might want to sanitize here

      // Limit text length to prevent exceeding API payload limits
      const truncatedText = sanitizedText.length > apiConfig.maxTextLength
        ? sanitizedText.substring(0, apiConfig.maxTextLength)
        : sanitizedText;

      setSelectedText(truncatedText);

      // Add selected text as a message in the chat
      if (!messages.some(msg => msg.type === 'context' && msg.content === truncatedText)) {
        setMessages(prev => [
          ...prev,
          {
            id: Date.now(),
            type: 'context',
            content: truncatedText,
            timestamp: new Date()
          }
        ]);
      }
    }
  };

  // Function to clear the selected text
  const clearSelectedText = () => {
    setSelectedText('');
    // Remove the context message from the chat if it exists
    setMessages(prev => prev.filter(msg => msg.type !== 'context'));
  };

  // Function to update the selected text (allow user to modify)
  const updateSelectedText = (newText) => {
    if (newText.length <= apiConfig.maxTextLength) {
      setSelectedText(newText);
      // Update the context message in the chat
      setMessages(prev =>
        prev.map(msg =>
          msg.type === 'context' ? { ...msg, content: newText } : msg
        )
      );
    } else {
      setError(`Text exceeds maximum length of ${apiConfig.maxTextLength} characters`);
    }
  };

  // Function to handle citation clicks
  const handleCitationClick = (citation) => {
    // This would typically navigate to the relevant section in the textbook
    // For now, we'll just log the citation or open it in a new tab if it has a URL
    if (citation.url) {
      window.open(citation.url, '_blank');
    } else if (citation.section) {
      // In a real implementation, this might scroll to a specific section
      // or open a modal with the relevant content
      console.log('Navigating to section:', citation.section);
    }
  };

  // Performance optimization: Debounce function
  const debounce = (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  };

  // Set up text selection listener with debouncing
  useEffect(() => {
    const debouncedHandleTextSelection = debounce(() => {
      // Add a small delay to ensure selection is complete
      setTimeout(handleTextSelection, 50);
    }, 150); // 150ms debounce time

    document.addEventListener('mouseup', debouncedHandleTextSelection);

    // Also listen for keyup events to catch keyboard-based selections
    document.addEventListener('keyup', debouncedHandleTextSelection);

    return () => {
      document.removeEventListener('mouseup', debouncedHandleTextSelection);
      document.removeEventListener('keyup', debouncedHandleTextSelection);
    };
  }, []);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Function to submit the question to the RAG agent
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!userQuestion.trim()) {
      setError(CHAT_CONSTANTS.UI.ERROR_MESSAGE);
      return;
    }

    // Validate input length
    if (userQuestion.length > apiConfig.maxTextLength) {
      setError(`Question exceeds maximum length of ${apiConfig.maxTextLength} characters`);
      return;
    }

    // If we have selected text, validate the combined length
    if (selectedText) {
      const combinedLength = selectedText.length + userQuestion.length;
      if (combinedLength > apiConfig.maxTextLength) {
        setError(`Selected text and question combined exceed maximum length of ${apiConfig.maxTextLength} characters`);
        return;
      }
    }

    // Add user question to messages
    const userMessage = {
      id: Date.now(),
      type: CHAT_CONSTANTS.MESSAGE_TYPES.USER,
      content: userQuestion,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError('');

    try {
      // Send query to RAG agent (selectedText can be empty)
      const response = await sendQueryToRAG(selectedText, userQuestion);

      // Validate response
      if (!response || !response.response || response.response.trim() === '') {
        // Add error message to chat
        const errorMessage = {
          id: Date.now() + 1,
          type: CHAT_CONSTANTS.MESSAGE_TYPES.SYSTEM,
          content: 'The AI agent returned an empty response. Please try again.',
          timestamp: new Date()
        };

        setMessages(prev => [...prev, errorMessage]);
        return;
      }

      // Check if this is a mock response (for debugging purposes)
      if (response.metadata && response.metadata.mock) {
        console.log('DEBUG: Using mock response for testing purposes');
      }

      // Add AI response to messages
      const aiMessage = {
        id: Date.now() + 1,
        type: CHAT_CONSTANTS.MESSAGE_TYPES.AI,
        content: response.response,
        citations: response.citations || [],
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
      setUserQuestion('');
    } catch (err) {
      console.error('DEBUG: Error getting response from RAG agent:', err);
      console.error('DEBUG: Error name:', err.name);
      console.error('DEBUG: Error message:', err.message);
      console.error('DEBUG: Error stack:', err.stack);

      // Determine the type of error and provide appropriate message
      // Note: With the updated apiService, most network errors should return mock responses
      // rather than reaching this catch block
      let errorMessageContent = CHAT_CONSTANTS.UI.ERROR_MESSAGE;

      if (err.message.includes('timeout')) {
        errorMessageContent = 'The server is taking too long to respond. Please try again later.';
      } else if (err.message.includes('Network error')) {
        // With the updated apiService, network errors should return mock responses
        // This error message should rarely appear now
        errorMessageContent = 'Using offline mode. Responses are generated locally.';
        console.log('INFO: Network error occurred but should have returned mock response');
      } else if (err.message.includes('HTTP error')) {
        // Check for specific HTTP status codes
        if (err.message.includes('429')) {
          errorMessageContent = 'The AI service is temporarily at capacity due to usage limits. Please try again later.';
        } else {
          errorMessageContent = `The AI service is temporarily unavailable (Error: ${err.message}). Please try again later.`;
        }
      }

      console.log('DEBUG: Setting error message:', errorMessageContent);

      // Only set error if it's a genuine error (not a handled network issue)
      // For network errors, the apiService should return mock responses, so we shouldn't show error messages
      if (!err.message.includes('Network error')) {
        setError(errorMessageContent);

        // Add error message to chat
        const errorMessage = {
          id: Date.now() + 1,
          type: CHAT_CONSTANTS.MESSAGE_TYPES.SYSTEM,
          content: errorMessageContent,
          timestamp: new Date()
        };

        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Toggle chat interface visibility
  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  };

  // Clear chat history
  const clearChat = () => {
    setMessages([]);
    setSelectedText('');
    setUserQuestion('');
    setError('');
  };

  return (
    <div
      className={`chat-interface ${isOpen ? 'open' : 'closed'}`}
      role="complementary"
      aria-label="AI Assistant Chat"
    >
      {/* Chat toggle button */}
      <button
        className="chat-toggle-btn"
        onClick={toggleChat}
        aria-label={isOpen ? "Close chat" : "Open chat"}
        title={isOpen ? "Close chat" : "Open chat"}
      >
        {isOpen ? '✕' : '💬'}
      </button>

      {/* Chat panel */}
      {isOpen && (
        <div
          className="chat-panel"
          role="dialog"
          aria-modal="true"
          aria-labelledby="chat-title"
        >
          <div className="chat-header">
            <h3 id="chat-title">AI Assistant</h3>
            <div className="chat-controls">
              <button
                className="clear-btn"
                onClick={clearChat}
                title="Clear chat"
                aria-label="Clear chat"
              >
                🗑️
              </button>
              <button
                className="close-btn"
                onClick={toggleChat}
                title="Close chat"
                aria-label="Close chat"
              >
                ✕
              </button>
            </div>
          </div>

          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Ask a question about the textbook content!</p>
                <p>(You can also select text for context-specific questions)</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.type}`}
                >
                  <div className="message-content">
                    {message.type === 'context' && (
                      <div className="selected-text-context">
                        <div className="context-header">
                          <strong>Selected text:</strong>
                          <button
                            className="clear-context-btn"
                            onClick={clearSelectedText}
                            title="Clear selected text"
                            aria-label="Clear selected text"
                          >
                            ✕
                          </button>
                        </div>
                        <textarea
                          className="context-textarea"
                          value={message.content}
                          onChange={(e) => updateSelectedText(e.target.value)}
                          rows={Math.min(5, message.content.split('\n').length + 1)}
                          maxLength={apiConfig.maxTextLength}
                        />
                      </div>
                    )}
                    {message.type !== 'context' && (
                      <div>
                        <div>{message.content}</div>
                        {message.citations && message.citations.length > 0 && (
                          <div className="citations">
                            <strong>Citations:</strong>
                            <ul>
                              {message.citations.map((citation, idx) => (
                                <li key={idx}>
                                  <button
                                    className="citation-link"
                                    onClick={() => handleCitationClick(citation)}
                                    title={`Go to: ${citation.title || citation.section || citation.url}`}
                                  >
                                    {citation.title || citation.section || `Source ${idx + 1}`}
                                  </button>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="message-timestamp">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message ai">
                <div className="message-content">
                  <div className="loading-indicator">
                    {CHAT_CONSTANTS.UI.LOADING_MESSAGE}
                    <span className="loading-dots">...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <form className="chat-input-form" onSubmit={handleSubmit}>
            <textarea
              ref={inputRef}
              className="chat-input"
              value={userQuestion}
              onChange={(e) => setUserQuestion(e.target.value)}
              placeholder={CHAT_CONSTANTS.UI.INPUT_PLACEHOLDER}
              disabled={isLoading}
              rows={3}
              aria-label="Enter your question"
              onKeyDown={(e) => {
                // Allow Shift+Enter for new lines, Enter to submit
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!isLoading && userQuestion.trim()) {
                    handleSubmit(e);
                  }
                }
              }}
            />
            <button
              type="submit"
              className="send-btn"
              disabled={isLoading || !userQuestion.trim()}
              aria-label="Send question"
            >
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </form>
        </div>
      )}
    </div>
  );
};

export default ChatInterface;