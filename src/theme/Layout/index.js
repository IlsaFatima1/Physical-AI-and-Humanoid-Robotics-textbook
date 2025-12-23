import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatInterface from '@site/src/components/ChatInterface';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function Layout(props) {
  const context = useDocusaurusContext();
  const { REACT_APP_API_BASE_URL } = context.siteConfig.customFields || {};

  // Inject environment variables into window object for access by other components
  React.useEffect(() => {
    if (typeof window !== 'undefined') {
      window.DOCUSAURUS_ENV = window.DOCUSAURUS_ENV || {};
      window.DOCUSAURUS_ENV.REACT_APP_API_BASE_URL = REACT_APP_API_BASE_URL || 'http://localhost:8000';
    }
  }, [REACT_APP_API_BASE_URL]);

  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
        <ChatInterface />
      </OriginalLayout>
    </>
  );
}