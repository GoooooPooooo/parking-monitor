import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { DetectionPage } from '../../pages/detection';
import { HistoryPage } from '../../pages/history';
import { ROUTES } from '../../shared/config/routes';

export const AppRouter: React.FC = () => {
  return (
    <Routes>
      <Route path={ROUTES.DETECTION} element={<DetectionPage />} />
      <Route path={ROUTES.HISTORY} element={<HistoryPage />} />
    </Routes>
  );
};
