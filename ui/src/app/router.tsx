import React from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import DashboardLayout from './layouts/DashboardLayout';
import LivePage from '../pages/LivePage';
import ArchitecturePage from '../pages/ArchitecturePage';
import AnalyticsPage from '../pages/AnalyticsPage';

export const router = createBrowserRouter([
    {
        path: '/',
        element: <DashboardLayout />,
        children: [
            {
                index: true,
                element: <Navigate to="/live" replace />,
            },
            {
                path: 'live',
                element: <LivePage />,
            },
            {
                path: 'analytics',
                element: <AnalyticsPage />,
            },
            {
                path: 'architecture',
                element: <ArchitecturePage />,
            },
        ],
    },
]);
