# Chat Interface Component

The ChatInterface component provides an AI-powered chat assistant that integrates with the textbook content, allowing users to ask questions about selected text.

## Features

- **Text Selection**: Automatically captures selected text on the page
- **AI Integration**: Connects to a RAG (Retrieval-Augmented Generation) agent backend
- **Citations**: Displays source citations for AI responses
- **Responsive Design**: Works on desktop and mobile devices
- **Accessibility**: Full keyboard navigation and screen reader support
- **Error Handling**: Graceful degradation when backend is unavailable

## Usage

The component can be integrated into any page and will automatically detect text selections.

## API Integration

The component communicates with the backend via the `/chat` endpoint, sending the selected text and user question as context for the AI response.

## Configuration

The component uses the following configuration files:
- `src/config/apiConfig.js` - API endpoint configuration
- `src/config/constants.js` - UI and behavior constants

## Accessibility

- Full keyboard navigation support
- Screen reader optimized markup
- Proper ARIA attributes
- Focus management

## Browser Compatibility

- Modern browsers supporting ES6+
- Text selection works across browsers
- Graceful degradation for older browsers