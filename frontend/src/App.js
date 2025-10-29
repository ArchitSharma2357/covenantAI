import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import HeroSection from './components/HeroSection';
import UploadBox from './components/UploadBox';
import AnalyzeButton from './components/AnalyzeButton';
import LandingPage from './components/LandingPage';
import Footer from './components/Footer';
import ReportPage from './components/ReportPage';
import AuthPage from './components/AuthPage';
import HistoryPage from './components/HistoryPage';
import ChatPage from './components/ChatPage';
import AnalyticsPage from './components/AnalyticsPage';
import ComparePage from './components/ComparePage';
import ProfilePage from './components/ProfilePage';
import './App.css';
import { Plus } from 'lucide-react';

function AppContent() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [textInput, setTextInput] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
  const [matrixMode, setMatrixMode] = useState(false);
  const [showFAB, setShowFAB] = useState(true);

  useEffect(() => {
    // Check if user is authenticated on app load
    const token = localStorage.getItem('token');
    console.log('🔐 App.js loaded - Token in localStorage:', token ? 'YES' : 'NO');
    if (token) {
      console.log('🔐 Setting up axios default header with token');
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      checkAuthStatus();
    } else {
      console.log('🔐 No token found - User is GUEST');
    }
    // apply theme to html element for Tailwind dark mode
    try {
      if (theme === 'dark') {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    } catch (e) {
      // ignore in SSR or restrictive environments
    }

  // matrix theme has been removed; keep particles toggle only
  const onToggleParticles = () => window.dispatchEvent(new CustomEvent('toggleParticlesInternal'));
    window.addEventListener('toggleParticles', onToggleParticles);
  const onToggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');
  window.addEventListener('toggleTheme', onToggleTheme);
  // unified theme toggle event (new) for consistent handling across components
  window.addEventListener('themeToggle', onToggleTheme);
    // Listen for profile updates from ProfilePage
    const onProfileUpdated = (e) => {
      if (e?.detail) setUser(e.detail);
    };
    window.addEventListener('profileUpdated', onProfileUpdated);

    // Settings events
    const onExport = async () => {
      try {
        const token = localStorage.getItem('token');
        const headers = token ? { Authorization: `Bearer ${token}` } : {};
        const resp = await axios.get('http://localhost:8000/api/auth/export', { headers, responseType: 'blob' });
        const url = window.URL.createObjectURL(new Blob([resp.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `covenantai-export.zip`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Export started' }));
      } catch (err) {
        console.warn('Export failed', err);
        window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Export failed' }));
      }
    };

    const onDeleteAccount = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'You must be logged in to delete account.' }));
          return;
        }
        await axios.delete('http://localhost:8000/api/auth/me', { headers: { Authorization: `Bearer ${token}` } });
        handleLogout();
        window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Account deleted' }));
      } catch (err) {
        console.warn('Delete account failed', err);
        window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Account deletion failed' }));
      }
    };

    const onManageSubscription = () => {
      // Navigate to profile subscriptions section
      try {
        navigate('/profile');
        // allow ProfilePage to scroll to subscriptions area if it exists
        setTimeout(() => window.location.hash = '#subscriptions', 150);
      } catch (e) {
        console.warn('Failed to open subscriptions', e);
      }
    };

    window.addEventListener('exportData', onExport);
    window.addEventListener('deleteAccount', onDeleteAccount);
    window.addEventListener('manageSubscription', onManageSubscription);
  // allow guest users to toggle theme via event
  const onGuestToggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');
  window.addEventListener('guestToggleTheme', onGuestToggleTheme);

    return () => {
  window.removeEventListener('toggleParticles', onToggleParticles);
      window.removeEventListener('toggleTheme', onToggleTheme);
      window.removeEventListener('profileUpdated', onProfileUpdated);
  window.removeEventListener('exportData', onExport);
  window.removeEventListener('deleteAccount', onDeleteAccount);
  window.removeEventListener('manageSubscription', onManageSubscription);
      window.removeEventListener('guestToggleTheme', onGuestToggleTheme);
    };
  }, []);

  useEffect(() => {
    // keep matrix mode global and notify listeners
    // matrix removed
  }, [matrixMode]);

  useEffect(() => {
    // persist and apply theme globally
    localStorage.setItem('theme', theme);
    try {
      if (theme === 'dark') {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    } catch (e) {}
    // notify other components (Header listens for this to update icons)
    try {
      window.dispatchEvent(new CustomEvent('themeChanged', { detail: theme }));
    } catch (e) {
      // ignore in older browsers
    }
  }, [theme]);

  const checkAuthStatus = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:8000/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      console.log('✅ Auth check SUCCESS - User:', response.data.name || response.data.email);
      setUser(response.data);
      setIsAuthenticated(true);
    } catch (error) {
      // Only clear token on explicit 401 Unauthorized. Other errors (network, 5xx)
      // may be transient and shouldn't log the user out when they click Profile.
      const status = error?.response?.status;
      console.log('❌ Auth check FAILED - Status:', status, 'Message:', error?.message);
      if (status === 401) {
        localStorage.removeItem('token');
        setIsAuthenticated(false);
        setUser(null);
      } else {
        // Keep current auth state but inform the user; do not silently log out.
        console.warn('Auth check failed but not removing token, status:', status, error?.message);
        window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Could not verify session. Please try again.' }));
      }
    }
  };

  const handleFileSelect = (file) => {
    setSelectedFile(file);
  };

  const handleTextInput = (text) => {
    setTextInput(text);
  };

  const handleAnalyze = async () => {
    console.log('🚀 handleAnalyze START');
    setLoading(true);
    const formData = new FormData();
    let documentTextContent = '';
    
    if (selectedFile) {
      const text = await selectedFile.text();
      documentTextContent = text;
      formData.append('rawText', text);
    } else if (textInput) {
      documentTextContent = textInput;
      formData.append('rawText', textInput);
    } else {
      alert('Please select a file or enter text to analyze.');
      setLoading(false);
      return;
    }

    try {
      // First upload the document
      console.log('📤 Uploading document...');
      const uploadResponse = await axios.post('http://localhost:8000/api/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('📤 Upload success, document_id:', uploadResponse.data.document_id);
      
      // Then analyze it
      const documentId = uploadResponse.data.document_id;
      console.log('🔍 Analyzing document:', documentId);
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `http://localhost:8000/api/documents/${documentId}/analyze`,
        {},
        token ? {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        } : {}
      );
      console.log('🔍 Analysis success, response keys:', Object.keys(response.data));
      setAnalysisResult(response.data);
      console.log('📊 setAnalysisResult called');

      console.log('📊 Analysis complete. isAuthenticated:', isAuthenticated);
      console.log('📊 Token available:', !!token);

      // Save to history - check token directly from localStorage to avoid state timing issues
      console.log('💾💾💾 BEFORE TOKEN CHECK - token is:', token ? 'PRESENT' : 'MISSING');
      
      if (token) {
        console.log('💾 Token found! Attempting to save to history with axios');
        console.log('💾 Axios Authorization header:', axios.defaults.headers.common['Authorization']);
        console.log('💾 Request payload:', {
          documentText: documentTextContent.substring(0, 50) + '...',
          analysisResult: Object.keys(response.data),
          filename: selectedFile ? selectedFile.name : 'Pasted Text',
        });
        // Use axios (with defaults already set) to POST to history
        try {
          console.log('💾 About to send POST request to /api/history');
          const historyRes = await axios.post('http://localhost:8000/api/history', {
            documentText: documentTextContent,
            analysisResult: response.data,
            filename: selectedFile ? selectedFile.name : 'Pasted Text',
          }, {
            timeout: 5000  // 5 second timeout
          });
          console.log('✅ Document saved to history:', historyRes.data);
          console.log('✅ History save response status:', historyRes.status);
          // Give it a moment to complete
          await new Promise(resolve => setTimeout(resolve, 500));
        } catch (historyError) {
          console.error('❌ Failed to save to history:');
          console.error('❌ Error message:', historyError.message);
          console.error('❌ Error response:', historyError.response?.data);
          console.error('❌ Error status:', historyError.response?.status);
          console.error('❌ Full error:', historyError);
        }
      } else {
        console.log('👤 User is guest - saving to localStorage');
        // Persist a lightweight guest history entry to localStorage so guest history page can show it
        try {
          const guestHistory = JSON.parse(localStorage.getItem('guest_history_v1') || '[]');
          const id = 'g-' + Date.now();
          const entry = {
            id,
            filename: selectedFile ? selectedFile.name : 'Pasted Text',
            upload_time: new Date().toISOString(),
            expires_at: new Date(Date.now() + 10 * 60 * 1000).toISOString(),
            analysis_result: response.data,
          };
          guestHistory.unshift(entry);
          // keep only last 20
          localStorage.setItem('guest_history_v1', JSON.stringify(guestHistory.slice(0, 20)));
          console.log('✅ Guest entry saved to localStorage');
        } catch (err) {
          console.warn('Failed to persist guest history locally', err);
        }
      }

      console.log('📊 About to navigate to /report');
      // Navigate to report page after all history saving is done
      navigate('/report');
    } catch (error) {
      console.error('❌ CAUGHT ERROR in handleAnalyze:', error);
      console.error('❌ Error message:', error.message);
      console.error('❌ Error stack:', error.stack);
      alert('Error analyzing document. Please try again.');
    } finally {
      console.log('🏁 handleAnalyze finally block');
      setLoading(false);
    }
  };

  const handleLogin = async (email, password) => {
    try {
      console.log('🔐 Attempting login for:', email);
      const response = await axios.post('http://localhost:8000/api/auth/login', {
        email,
        password,
      });
      // If server didn't return an access token (e.g., email verification required),
      // don't store an undefined token or mark user as authenticated.
      const access_token = response?.data?.access_token;
      console.log('🔐 Login response received. Token:', access_token ? 'YES' : 'NO');
      if (!access_token) {
        const msg = response?.data?.message || 'Login failed: no access token returned';
        // Provide the server message (like verification required) to the caller
        throw new Error(msg);
      }

      console.log('💾 Saving token to localStorage');
      localStorage.setItem('token', access_token);
      console.log('💾 Token saved. Verify:', localStorage.getItem('token') ? 'YES' : 'NO');
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      setIsAuthenticated(true);
      console.log('✅ isAuthenticated set to true');
      await checkAuthStatus();
    } catch (error) {
      console.error('❌ Login failed:', error);
      const msg = error?.response?.data?.detail || error?.message || 'Login failed';
      throw new Error(msg);
    }
  };

  const handleSignup = async (email, password, name) => {
    try {
      await axios.post('http://localhost:8000/api/auth/signup', {
        email,
        password,
        name,
      });
      await handleLogin(email, password);
    } catch (error) {
      const msg = error?.response?.data?.detail || error?.message || 'Signup failed';
      throw new Error(msg);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    delete axios.defaults.headers.common['Authorization'];
    setIsAuthenticated(false);
    setUser(null);
  };

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar isCollapsed={sidebarCollapsed} setIsCollapsed={setSidebarCollapsed} isAuthenticated={isAuthenticated} />
  <div className={`flex-1 transition-all duration-300 pt-16 ${sidebarCollapsed ? 'ml-16' : 'ml-64'}`}>
        <Header
          isAuthenticated={isAuthenticated}
          user={user}
          onLogout={handleLogout}
        />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<LandingPage setAnalysisResult={setAnalysisResult} isAuthenticated={isAuthenticated} />} />
            <Route path="/report" element={<ReportPage analysisResult={analysisResult} />} />
            <Route path="/auth" element={
              isAuthenticated ? <Navigate to="/" /> : <AuthPage onLogin={handleLogin} onSignup={handleSignup} />
            } />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/history/guest" element={<HistoryPage type="guest" />} />
            <Route path="/history/user" element={<HistoryPage type="user" />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/compare" element={<ComparePage />} />
          </Routes>
        </main>
        {/* Floating Action Button */}
        {showFAB && (
          <button
            onClick={() => window.dispatchEvent(new CustomEvent('openUpload'))}
            className="fixed right-8 bottom-8 z-[130] bg-black text-white w-14 h-14 rounded-full shadow-2xl flex items-center justify-center hover:scale-105 transition-transform"
            title="Upload / Add Document"
          >
            <Plus className="w-6 h-6" />
          </button>
        )}
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
