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