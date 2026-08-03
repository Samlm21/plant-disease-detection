import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { AppLayout } from '@/components/layout/AppLayout';

// Using standard named imports instead of React.lazy
import { Home } from '@/pages/Home';
import { Predict } from '@/pages/Predict';
import { History } from '@/pages/History';
import { About } from '@/pages/About';
import { NotFound } from '@/pages/NotFound';

export const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<AppLayout />}>
        <Route index element={<Home />} />
        <Route path="predict" element={<Predict />} />
        <Route path="history" element={<History />} />
        <Route path="about" element={<About />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
};

export default AppRoutes;