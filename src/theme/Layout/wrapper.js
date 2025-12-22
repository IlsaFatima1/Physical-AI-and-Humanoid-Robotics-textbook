import React from 'react';
import ChatInterface from '@site/src/components/ChatInterface';

export default function LayoutWrapper({children}) {
  return (
    <>
      {children}
      <ChatInterface />
    </>
  );
}