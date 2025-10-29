import React, { useEffect, useState } from 'react';
import axios from 'axios';

const ProfilePage = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [avatarData, setAvatarData] = useState(null);
  const [subscription, setSubscription] = useState(null);

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          setError('Not authenticated');
          setLoading(false);
          return;
        }
        const resp = await axios.get('http://localhost:8000/api/auth/me', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setUser(resp.data);
        setName(resp.data.name || '');
        setDescription(resp.data.description || '');
        setAvatarData(resp.data.avatar || null);
        // fetch subscription status if available
        try {
          const subResp = await axios.get('http://localhost:8000/api/auth/subscription', { headers: { Authorization: `Bearer ${localStorage.getItem('token')}` } });
          setSubscription(subResp.data || null);
        } catch (err) {
          // no subscription endpoint or not subscribed
          setSubscription(null);
        }
      } catch (err) {
        setError('Failed to load profile');
      } finally {
        setLoading(false);
      }
    };
    fetchUser();
  }, []);

  const handleAvatarChange = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setAvatarData(reader.result);
    reader.readAsDataURL(file);
  };

  const handleSave = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;
      const resp = await axios.patch('http://localhost:8000/api/auth/me', { name, description, avatar: avatarData }, { headers: { Authorization: `Bearer ${token}` } });
      setUser(resp.data);
      setEditing(false);
    } catch (err) {
      console.error('Failed to save profile', err);
      alert('Failed to save profile');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    window.location.href = '/';
  };

  if (loading) return <div className="p-6">Loading profile...</div>;
  if (error) return (
    <div className="p-6">
      <p className="text-destructive font-semibold">{error}</p>
      <a href="/auth" className="text-primary hover:underline">Sign in</a>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Subtle background pattern */}
      <div className="absolute inset-0 opacity-[0.02] dark:opacity-[0.05]">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, rgba(0,0,0,.15) 1px, transparent 0)`,
          backgroundSize: '20px 20px'
        }} />
      </div>

      <div className="relative z-10 max-w-2xl mx-auto p-6 min-h-screen">
        <div className="pt-12">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2" style={{
              fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif'
            }}>
              Profile
            </h1>
            <p className="text-slate-600 dark:text-slate-300 font-light" style={{
              fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
            }}>
              Manage your account settings and preferences.
            </p>
          </div>

          <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-3xl border border-slate-200/50 dark:border-slate-700/50 shadow-sm p-8">
            <div className="flex items-center space-x-6 mb-8">
              <div className="w-24 h-24 bg-slate-900 dark:bg-slate-100 dark:text-slate-900 rounded-2xl overflow-hidden flex items-center justify-center text-3xl font-bold text-white shadow-lg">
                {avatarData ? (
                  <img src={avatarData} alt="avatar" className="w-full h-full object-cover" />
                ) : (
                  (user.name || 'U')[0]
                )}
              </div>
              <div className="flex-1">
                {!editing ? (
                  <>
                    <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-1" style={{
                      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif'
                    }}>
                      {user.name}
                    </h2>
                    <p className="text-slate-600 dark:text-slate-300 font-light mb-2" style={{
                      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
                    }}>
                      {user.email}
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-400 font-light">ID: {user.id}</p>
                  </>
                ) : (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300" style={{
                        fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
                      }}>
                        Name
                      </label>
                      <input
                        value={name}
                        onChange={e => setName(e.target.value)}
                        className="w-full px-4 py-3 border border-slate-200 dark:border-slate-600 rounded-xl bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-900/20 dark:focus:ring-slate-100/20 focus:border-slate-900/30 dark:focus:border-slate-100/30 transition-all"
                        style={{
                          fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
                        }}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300" style={{
                        fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
                      }}>
                        Description
                      </label>
                      <textarea
                        value={description}
                        onChange={e => setDescription(e.target.value)}
                        className="w-full px-4 py-3 border border-slate-200 dark:border-slate-600 rounded-xl bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-900/20 dark:focus:ring-slate-100/20 focus:border-slate-900/30 dark:focus:border-slate-100/30 transition-all resize-none"
                        rows={4}
                        style={{
                          fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
                        }}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300" style={{
                        fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
                      }}>
                        Avatar
                      </label>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={e => handleAvatarChange(e.target.files[0])}
                        className="w-full px-4 py-3 border border-slate-200 dark:border-slate-600 rounded-xl bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:text-white file:bg-slate-900 dark:file:bg-slate-100 dark:file:text-slate-900 dark:file:text-white hover:file:bg-slate-800 dark:hover:file:bg-slate-200 transition-all"
                        style={{
                          fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
                        }}
                      />
                    </div>
                    <div className="flex space-x-3">
                      <button
                        onClick={handleSave}
                        className="px-6 py-3 bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white rounded-xl font-medium hover:bg-slate-800 dark:hover:bg-slate-200 transition-all"
                        style={{
                          fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif'
                        }}
                      >
                        Save
                      </button>
                      <button
                        onClick={() => setEditing(false)}
                        className="px-6 py-3 border border-slate-200 dark:border-slate-600 rounded-xl font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-all"
                        style={{
                          fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif'
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {!editing && (
              <div className="flex space-x-3 mb-8">
                <button
                  onClick={() => setEditing(true)}
                  className="px-6 py-3 bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white rounded-xl font-medium hover:bg-slate-800 dark:hover:bg-slate-200 transition-all"
                  style={{
                    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif'
                  }}
                >
                  Edit Profile
                </button>
                <button
                  onClick={handleLogout}
                  className="px-6 py-3 border border-slate-200 dark:border-slate-600 rounded-xl font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-all"
                  style={{
                    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif'
                  }}
                >
                  Logout
                </button>
              </div>
            )}

            <div className="space-y-6">
              <div className="bg-slate-50/80 dark:bg-slate-800/50 rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-3" style={{
                  fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif'
                }}>
                  About
                </h3>
                <p className="text-slate-600 dark:text-slate-300 font-light leading-relaxed" style={{
                  fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
                }}>
                  {user.description || 'This is your profile. You can add a description, upload a profile photo, or connect accounts from the Settings page.'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;
