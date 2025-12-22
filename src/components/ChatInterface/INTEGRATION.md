# Docusaurus Integration Guide

This guide explains how to integrate the ChatInterface component with a Docusaurus-based textbook site.

## Installation

1. Place the ChatInterface component files in your Docusaurus project:
   - `src/components/ChatInterface.jsx`
   - `src/css/chat-component.css`
   - `src/utils/apiService.js`
   - `src/config/apiConfig.js`
   - `src/config/constants.js`

## Docusaurus Configuration

### 1. Add CSS to Docusaurus

Add the chat component CSS to your Docusaurus configuration in `docusaurus.config.js`:

```js
module.exports = {
  // ... other config
  stylesheets: [
    // ... other stylesheets
    '/css/chat-component.css', // Add this line
  ],
};
```

### 2. Import Component in Layout

To make the chat available on all pages, you can add it to your Docusaurus layout. Create or modify `src/theme/Layout/index.js`:

```jsx
import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatInterface from '@site/src/components/ChatInterface';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
        <ChatInterface />
      </OriginalLayout>
    </>
  );
}
```

### 3. Environment Variables

Set up environment variables in your `.env` file:

```
REACT_APP_API_BASE_URL=http://localhost:8000
```

## Customization

### Styling

The chat component uses CSS variables that can be customized to match your Docusaurus theme:

- Colors can be overridden in your Docusaurus CSS
- Sizes and spacing can be adjusted in `chat-component.css`

### API Configuration

Update `src/config/apiConfig.js` to match your backend endpoint configuration.

## Usage

Once integrated, the chat interface will appear as a floating button on all textbook pages. Users can:
1. Select text on any page
2. Click the chat icon to open the interface
3. Ask questions about the selected text
4. View AI responses with citations