import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight, Home, History, MessageSquare, BarChart3, GitCompare, FileText, ChevronDown, ChevronUp, Users, User } from 'lucide-react';

const Sidebar = ({ isCollapsed, setIsCollapsed, isAuthenticated }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [activeSection, setActiveSection] = useState('main');
  const [historyExpanded, setHistoryExpanded] = useState(false);
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);

  const mainNavItems = [
    { path: '/', icon: Home, label: 'Home' },
    { path: '/chat', icon: MessageSquare, label: 'AI Chat' },
    { path: '/analytics', icon: BarChart3, label: 'Analytics' },
    { path: '/compare', icon: GitCompare, label: 'Compare' },
  ];

  const historyItems = [
    { path: '/history/guest', icon: Users, label: 'Guest History', type: 'guest' },
    { path: '/history/user', icon: User, label: 'User History', type: 'user', requiresAuth: true },
  ];

  const handleNavClick = (path) => {
    if (path === '/chat') {
      navigate('/chat');
    } else {
      navigate(path);
    }
  };

  const isActive = (path) => location.pathname === path;

  return (
    // place sidebar below the fixed header (h-16) so header doesn't overlap
    <div className={`fixed left-0 top-16 h-[calc(100%-4rem)] bg-card/80 backdrop-blur-xl border-r border-border/50 transition-all duration-300 z-50 shadow-2xl ${isCollapsed ? 'w-16' : 'w-64'}`}>
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        {!isCollapsed && (
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-primary rounded-xl flex items-center justify-center shadow-lg">
              <FileText className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-semibold text-foreground">CovenantAI</span>
          </div>
        )}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-2 hover:bg-muted/80 rounded-xl transition-all duration-200 hover:scale-105"
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      <nav className="p-4">
        <div className="space-y-2">
          {mainNavItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.path}
                onClick={() => handleNavClick(item.path)}
                className={`flex items-center gap-3 px-3 h-11 rounded-xl transition-all duration-200 w-full text-left group relative ${
                  isActive(item.path)
                    ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/25'
                    : 'hover:bg-muted/80 text-muted-foreground hover:text-foreground hover:shadow-md'
                }`}
              >
                <Icon className={`w-5 h-5 flex-shrink-0 transition-transform duration-200 ${isActive(item.path) ? 'scale-110' : 'group-hover:scale-105'}`} />
                {!isCollapsed && <span className="font-medium">{item.label}</span>}
                {isActive(item.path) && (
                  <div className="absolute inset-0 bg-primary/20 rounded-xl blur-xl -z-10"></div>
                )}
              </button>
            );
          })}

          {/* History Section */}
          <div className="pt-2">
            <button
              onClick={() => setHistoryExpanded(!historyExpanded)}
              className={`flex items-center gap-3 px-3 h-11 rounded-xl transition-all duration-200 w-full text-left group relative ${
                location.pathname.startsWith('/history')
                  ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/25'
                  : 'hover:bg-muted/80 text-muted-foreground hover:text-foreground hover:shadow-md'
              }`}
            >
              <History className={`w-5 h-5 flex-shrink-0 transition-transform duration-200 ${location.pathname.startsWith('/history') ? 'scale-110' : 'group-hover:scale-105'}`} />
              {!isCollapsed && (
                <>
                  <span className="font-medium">History</span>
                  {historyExpanded ? <ChevronUp className="w-4 h-4 ml-auto" /> : <ChevronDown className="w-4 h-4 ml-auto" />}
                </>
              )}
              {location.pathname.startsWith('/history') && (
                <div className="absolute inset-0 bg-primary/20 rounded-xl blur-xl -z-10"></div>
              )}
            </button>

            {/* Sub-buttons for history */}
            {historyExpanded && !isCollapsed && (
              <div className="ml-4 mt-2 space-y-1">
                {historyItems.map((item) => {
                  const isVisible = !item.requiresAuth || isAuthenticated;
                  if (!isVisible) return null;
                  const Icon = item.icon;
                  return (
                    <button
                      key={item.path}
                      onClick={() => handleNavClick(item.path)}
                      className={`flex items-center gap-3 px-3 h-9 rounded-lg transition-all duration-200 w-full text-left group relative text-sm ${
                        isActive(item.path)
                          ? 'bg-primary/20 text-primary shadow-md'
                          : 'hover:bg-muted/60 text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      <Icon className={`w-4 h-4 flex-shrink-0 transition-transform duration-200 ${isActive(item.path) ? 'scale-110' : 'group-hover:scale-105'}`} />
                      <span className="font-medium">{item.label}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </nav>
    </div>
  );
};

export default Sidebar;
