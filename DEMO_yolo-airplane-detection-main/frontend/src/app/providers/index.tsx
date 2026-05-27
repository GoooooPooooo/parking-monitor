import React from 'react';
import { App } from 'antd';
import { ThemeProvider } from './ThemeProvider';
import { QueryProvider } from './QueryProvider';

interface AppProvidersProps {
  children: React.ReactNode;
}

export const AppProviders: React.FC<AppProvidersProps> = ({ children }) => {
  return (
    <ThemeProvider>
      <QueryProvider>
        <App>{children}</App>
      </QueryProvider>
    </ThemeProvider>
  );
};
