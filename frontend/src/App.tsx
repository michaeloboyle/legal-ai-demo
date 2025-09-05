/**
 * Government Plain Language Compliance System - Main App
 * Agent: Frontend_MaterialUI
 * GitHub Issue: #8
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';

// Components
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import DocumentEditor from './pages/DocumentEditor';
import ComplianceMetrics from './pages/ComplianceMetrics';
import AgencyDashboard from './pages/AgencyDashboard';
import Settings from './pages/Settings';

// Government compliance theme
const governmentTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1565c0', // Government blue
      light: '#5e92f3',
      dark: '#003c8f',
    },
    secondary: {
      main: '#e65100', // Government orange for accents
      light: '#ff833a',
      dark: '#ac1900',
    },
    background: {
      default: '#fafafa',
      paper: '#ffffff',
    },
    text: {
      primary: '#212121',
      secondary: '#757575',
    },
    error: {
      main: '#d32f2f',
    },
    warning: {
      main: '#f57c00',
    },
    success: {
      main: '#388e3c',
    },
    info: {
      main: '#1976d2',
    },
  },
  typography: {
    fontFamily: [
      'Source Sans Pro',
      'Helvetica Neue',
      'Arial',
      'sans-serif'
    ].join(','),
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 500,
      lineHeight: 1.4,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    button: {
      textTransform: 'none', // Government style: no all caps
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 4, // Clean, minimal rounded corners
  },
  components: {
    // Government accessibility compliance
    MuiButton: {
      styleOverrides: {
        root: {
          minHeight: '44px', // 508 compliance - minimum touch target
          fontSize: '1rem',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            minHeight: '44px', // 508 compliance
          },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          minHeight: '44px', // 508 compliance
          fontSize: '1rem',
        },
      },
    },
  },
});

const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = React.useState(true);

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <ThemeProvider theme={governmentTheme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
          <Header onMenuClick={handleSidebarToggle} />
          
          <Sidebar 
            open={sidebarOpen} 
            onToggle={handleSidebarToggle}
          />
          
          <Box
            component=\"main\"
            sx={{
              flexGrow: 1,
              p: 3,
              mt: 8, // Account for header height
              ml: sidebarOpen ? '240px' : '0px',
              transition: (theme) => 
                theme.transitions.create(['margin'], {
                  easing: theme.transitions.easing.easeOut,
                  duration: theme.transitions.duration.enteringScreen,
                }),
            }}
          >
            <Routes>
              <Route path=\"/\" element={<Dashboard />} />
              <Route path=\"/editor\" element={<DocumentEditor />} />
              <Route path=\"/metrics\" element={<ComplianceMetrics />} />
              <Route path=\"/agency\" element={<AgencyDashboard />} />
              <Route path=\"/settings\" element={<Settings />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App;