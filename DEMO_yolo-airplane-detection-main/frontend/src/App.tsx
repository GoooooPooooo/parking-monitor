import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { AppProviders } from './app/providers';
import { AppRouter } from './app/routers/AppRouter';
import { NavbarWidget } from './widgets/navbar';
import './app/styles/global.css';

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <AppProviders>
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
          <NavbarWidget />
          <main style={{ flex: 1 }}>
            <AppRouter />
          </main>
          <footer style={{ textAlign: 'center', padding: '16px', color: '#999' }}>
            © 2026 YOLO Airplane Detection
          </footer>
        </div>
      </AppProviders>
    </BrowserRouter>
  );
};

export default App;
