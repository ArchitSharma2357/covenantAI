import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Bell, User, Settings, Upload, LogOut, Key, CreditCard, Cloud, Moon, Sun, Download, Trash2 } from 'lucide-react';

const Header = ({ isAuthenticated, user, onLogout }) => {
  const [showSettings, setShowSettings] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [showNotifications, setShowNotifications] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
  const settingsRef = useRef(null);

  // Listen for FAB openUpload event to trigger the hidden file input
  useEffect(() => {
    const handler = () => {
      const el = document.getElementById('file-upload-header');
      if (el) el.click();
    };
    window.addEventListener('openUpload', handler);
    return () => window.removeEventListener('openUpload', handler);
  }, []);

  // reflect theme changes from global App
  useEffect(() => {
    const handler = () => setTheme(localStorage.getItem('theme') || 'light');
    window.addEventListener('storage', handler);
    window.addEventListener('themeChanged', handler);
    // also listen for unified toggle event so icon flips immediately
    const toggleHandler = () => setTheme(localStorage.getItem('theme') || 'light');
    window.addEventListener('themeToggle', toggleHandler);
    return () => {
      window.removeEventListener('storage', handler);
      window.removeEventListener('themeChanged', handler);
      window.removeEventListener('themeToggle', toggleHandler);
    };
  }, []);

  useEffect(() => {
    const handleNotify = (e) => {
      setNotifications(prev => [{ message: e.detail, time: new Date().toLocaleTimeString() }, ...prev].slice(0, 10));
    };
    window.addEventListener('notifyEvent', handleNotify);
    return () => window.removeEventListener('notifyEvent', handleNotify);
  }, []);

  const navigate = useNavigate();
  const [canGoBack, setCanGoBack] = useState(false);

  useEffect(() => {
    // quick heuristic: enable back button when history length > 1
    try {
      setCanGoBack(window.history.length > 1);
    } catch (e) {
      setCanGoBack(false);
    }
  }, []);

  const goToProfile = () => {
    if (isAuthenticated) {
      navigate('/profile');
    } else {
      navigate('/auth');
    }
  };

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (settingsRef.current && !settingsRef.current.contains(e.target)) {
        setShowSettings(false);
      }
    };
    const handleEsc = (e) => {
      if (e.key === 'Escape') setShowSettings(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEsc);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEsc);
    };
  }, []);

  return (
  <header className="bg-card/90 backdrop-blur-xl border-b border-border/50 h-16 flex items-center justify-between px-6 shadow-sm fixed top-0 left-0 right-0 z-[110]">
      <div className="flex items-center gap-4">
        {canGoBack && (
          <button
            title="Back"
            onClick={() => navigate(-1)}
            className="p-2 hover:bg-muted/80 rounded-xl transition-all duration-200 mr-2"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
        )}
        {/* Theme toggle available for guests too */}
        <button
          className="p-2 hover:bg-muted/80 rounded-xl transition-all duration-200 hover:scale-105 relative mr-2"
          onClick={() => {
            // dispatch a single unified event; keep older events for compatibility
            if (isAuthenticated) {
              window.dispatchEvent(new CustomEvent('toggleTheme'));
              window.dispatchEvent(new CustomEvent('themeToggle'));
            } else {
              window.dispatchEvent(new CustomEvent('guestToggleTheme'));
              window.dispatchEvent(new CustomEvent('themeToggle'));
            }
          }}
          title="Toggle theme"
        >
          {theme === 'dark' ? <Sun className="w-5 h-5 text-muted-foreground" /> : <Moon className="w-5 h-5 text-muted-foreground" />}
        </button>
        {/* Search removed to simplify header UX per request */}
      </div>
      <div className="flex items-center gap-4">
        <button
          className="p-2 hover:bg-muted/80 rounded-xl transition-all duration-200 hover:scale-105 relative"
          onClick={() => document.getElementById('file-upload-header').click()}
          title="Upload document for analysis"
        >
          <Upload className="w-5 h-5 text-muted-foreground" />
          <input
            id="file-upload-header"
            type="file"
            className="hidden"
            onChange={e => {
              if (e.target.files && e.target.files[0]) {
                // Trigger analysis with selected file
                window.dispatchEvent(new CustomEvent('analyzeFile', { detail: e.target.files[0] }));
              }
            }}
          />
        </button>
        <button
          className="p-2 hover:bg-muted/80 rounded-xl transition-all duration-200 hover:scale-105 relative"
          onClick={() => setShowNotifications(v => !v)}
          title="Notifications"
        >
          <Bell className="w-5 h-5 text-muted-foreground" />
          {notifications.length > 0 && <span className="absolute -top-1 -right-1 w-3 h-3 bg-destructive rounded-full shadow-sm"></span>}
          {showNotifications && (
            <div className="absolute right-0 mt-2 w-64 bg-white border border-gray-200 rounded-xl shadow-lg z-50">
              <div className="p-2 text-xs font-semibold border-b">Notifications</div>
              <ul className="max-h-60 overflow-y-auto">
                {notifications.length === 0 ? (
                  <li className="p-2 text-xs text-gray-400">No notifications yet.</li>
                ) : notifications.map((n, i) => (
                  <li key={i} className="p-2 border-b last:border-b-0 text-xs text-gray-700">
                    <span className="font-bold">{n.time}</span>: {n.message}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </button>
        {isAuthenticated ? (
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center shadow-lg cursor-pointer"
                onClick={goToProfile}
                title="Profile"
              >
                <User className="w-4 h-4 text-primary-foreground" />
              </div>
              <span className="text-sm font-medium text-foreground">{user?.name || 'User'}</span>
            </div>
            <div className="relative" ref={settingsRef}>
              <button
                className="p-2 hover:bg-muted/80 rounded-xl transition-all duration-200 hover:scale-105 relative"
                onClick={() => setShowSettings(v => !v)}
                title="Settings"
              >
                <Settings className="w-5 h-5 text-muted-foreground" />
              </button>
              {showSettings && (
                <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-gray-900 rounded-xl shadow-2xl z-50 border border-border/50 overflow-hidden">
                  <div className="px-4 py-3 border-b flex items-center gap-3">
                    {user?.avatar ? (
                      <img src={user.avatar} alt="avatar" className="w-9 h-9 rounded-full object-cover" />
                    ) : (
                      <div className="w-9 h-9 bg-muted/20 rounded-full flex items-center justify-center">
                        <User className="w-4 h-4 text-muted-foreground" />
                      </div>
                    )}
                    <div>
                      <div className="text-sm font-semibold text-foreground">{user?.name || 'Your Account'}</div>
                      <div className="text-xs text-gray-500">{user?.email || ''}</div>
                    </div>
                  </div>
                  <ul className="py-1">
                    {/* Change Password option removed as requested */}
                    {/* Connect OAuth removed per UX request */}
                    {/* Manage Subscription removed per UX request */}
                    <li>
                      <button className="flex items-center gap-3 w-full px-4 py-2 hover:bg-muted/60 text-sm" onClick={() => {
                        if (isAuthenticated) {
                          window.dispatchEvent(new CustomEvent('toggleTheme'))
                        } else {
                          // allow guests to toggle theme locally
                          window.dispatchEvent(new CustomEvent('guestToggleTheme'))
                        }
                      }}>
                        {theme === 'dark' ? <Sun className="w-4 h-4 text-muted-foreground" /> : <Moon className="w-4 h-4 text-muted-foreground" />} {theme === 'dark' ? 'Light Theme' : 'Dark Theme'}
                      </button>
                    </li>
                    <li>
                      <button className="flex items-center gap-3 w-full px-4 py-2 hover:bg-muted/60 text-sm" onClick={() => window.dispatchEvent(new CustomEvent('exportData'))}>
                        <Download className="w-4 h-4 text-muted-foreground" /> Export Data
                      </button>
                    </li>
                    <li>
                      <button className="flex items-center gap-3 w-full px-4 py-2 hover:bg-muted/60 text-sm text-destructive" onClick={() => {
                        if (confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
                          window.dispatchEvent(new CustomEvent('deleteAccount'));
                        }
                      }}>
                        <Trash2 className="w-4 h-4" /> Delete Account
                      </button>
                    </li>
                  </ul>
                  <div className="px-4 py-3 border-t flex items-center gap-2">
                    <button className="flex-1 text-left text-sm px-3 py-2 rounded-md hover:bg-muted/60" onClick={onLogout}>
                      <div className="flex items-center gap-2"><LogOut className="w-4 h-4" /> Logout</div>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div
            className="w-8 h-8 bg-muted/50 rounded-full flex items-center justify-center cursor-pointer"
            onClick={() => window.location.href = '/auth'}
            title="Login / Profile"
          >
            <User className="w-4 h-4 text-muted-foreground" />
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
