import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { FileText, Download, Eye, Calendar, AlertTriangle, CheckCircle, XCircle, Trash2, Clock } from 'lucide-react';

const HistoryPage = ({ type = 'user' }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [history, setHistory] = useState([]);
  const [fullHistory, setFullHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [countdowns, setCountdowns] = useState({});
  const [deletingId, setDeletingId] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    fetchHistory();
  }, [type]);

  useEffect(() => {
    if (type === 'guest' && history.length > 0) {
      const interval = setInterval(() => {
        setCountdowns(prev => {
          const newCountdowns = {};
          const now = new Date().getTime();
          history.forEach(item => {
            const expiresAt = new Date(item.expires_at).getTime();
            const timeLeft = Math.max(0, expiresAt - now);
            newCountdowns[item.id] = timeLeft;
          });
          return newCountdowns;
        });

        // Remove expired items
        const now = new Date().getTime();
        const expiredIds = history.filter(item => new Date(item.expires_at).getTime() <= now).map(item => item.id);
        if (expiredIds.length > 0) {
          setHistory(prev => {
            const filtered = prev.filter(item => !expiredIds.includes(item.id));
            setFullHistory(filtered);
            // Update localStorage
            localStorage.setItem('guest_history_v1', JSON.stringify(filtered));
            return filtered;
          });
        }
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [history, type]);

  const fetchHistory = async () => {
    try {
      if (type === 'user') {
        const token = localStorage.getItem('token');
        if (!token) {
          setHistory([]);
          setLoading(false);
          return;
        }

          const response = await axios.get('http://localhost:8000/api/history', {
            headers: { Authorization: `Bearer ${token}` }
          });
          // Defensive: ensure analysis_result is parsed if string
          const parsedHistory = response.data.map(item => {
            let analysis_result = item.analysis_result;
            if (typeof analysis_result === 'string') {
              try {
                analysis_result = JSON.parse(analysis_result);
              } catch {
                analysis_result = {};
              }
            }
            return { ...item, analysis_result };
          });
          setHistory(parsedHistory);
          setFullHistory(parsedHistory);
      } else if (type === 'guest') {
        try {
          const response = await axios.get('http://localhost:8000/api/history/guest');
          if (Array.isArray(response.data) && response.data.length > 0) {
            // Filter out expired entries
            const now = new Date().getTime();
            const filteredResponse = response.data.filter(item => new Date(item.expires_at).getTime() > now);
            setHistory(filteredResponse);
            setFullHistory(filteredResponse);
          } else {
            // Fallback to localStorage-stored guest history persisted during analysis
            let stored = [];
            try {
              stored = JSON.parse(localStorage.getItem('guest_history_v1') || '[]');
              if (!Array.isArray(stored)) stored = [];
            } catch (e) {
              console.warn('Malformed guest_history_v1 in localStorage; resetting', e);
              stored = [];
            }
            // Filter out expired entries safely
            const now = new Date().getTime();
            const filteredStored = stored.filter(item => {
              try {
                return item && item.expires_at && new Date(item.expires_at).getTime() > now;
              } catch (e) {
                return false;
              }
            });
            setHistory(filteredStored);
            setFullHistory(filteredStored);
            // Update localStorage with filtered data
            try { localStorage.setItem('guest_history_v1', JSON.stringify(filteredStored)); } catch (e) { console.warn('Failed to update guest_history_v1', e); }
          }
        } catch (err) {
          console.warn('Guest history endpoint failed, falling back to local storage', err);
          let stored = [];
          try {
            stored = JSON.parse(localStorage.getItem('guest_history_v1') || '[]');
            if (!Array.isArray(stored)) stored = [];
          } catch (e) {
            console.warn('Malformed guest_history_v1 in localStorage during fallback', e);
            stored = [];
          }
          const now = new Date().getTime();
          const filteredStored = stored.filter(item => {
            try { return item && item.expires_at && new Date(item.expires_at).getTime() > now; } catch (e) { return false; }
          });
          setHistory(filteredStored);
          setFullHistory(filteredStored);
          try { localStorage.setItem('guest_history_v1', JSON.stringify(filteredStored)); } catch (e) { console.warn('Failed to update guest_history_v1 after fallback', e); }
        }
      }
    } catch (error) {
      console.error('Error fetching history:', error);
      if (error.response?.status === 401) {
        localStorage.removeItem('token');
        setHistory([]);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!searchQuery.trim()) {
      setHistory(fullHistory);
      return;
    }
    const q = searchQuery.toLowerCase().trim();
    const filtered = fullHistory.filter(item => {
      const filename = (item.filename || '').toLowerCase();
      const summary = (item.analysis_result?.summary || '').toLowerCase();
      return filename.includes(q) || summary.includes(q);
    });
    setHistory(filtered);
  }, [searchQuery, fullHistory]);

  const notify = (message) => {
    window.dispatchEvent(new CustomEvent('notifyEvent', { detail: message }));
  };

  const handleExportPDF = async (documentId, isGuest = false) => {
    // Robust download helper: accepts Blob or raw data
    const triggerDownload = (payload, filename, contentType) => {
      try {
        let blob;
        if (payload instanceof Blob) blob = payload;
        else blob = new Blob([payload], { type: contentType || 'application/octet-stream' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', filename);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      } catch (e) {
        console.error('triggerDownload failed', e);
      }
    };

    const getExtFromContentType = (ct) => {
      if (!ct) return '.pdf';
      ct = ct.toLowerCase();
      if (ct.includes('zip')) return '.zip';
      if (ct.includes('pdf')) return '.pdf';
      if (ct.includes('text')) return '.txt';
      return '.bin';
    };

    try {
      const historyItem = history.find(h => h.id === documentId) || {};
      const analysis = historyItem.analysis_result || null;
      const baseName = (historyItem.filename && historyItem.filename.length > 0) ? historyItem.filename.replace(/\s+/g, '_') : `legal-analysis-${documentId}`;

      // 1) Try bundled export (ZIP containing PDF + full AI text)
      if (analysis && analysis.raw_text && analysis.raw_text.length > 0) {
        try {
          const bundleResp = await axios.post('http://localhost:8000/api/export/pdf?bundle=true', analysis, {
            responseType: 'blob',
            headers: isGuest ? {} : { Authorization: `Bearer ${localStorage.getItem('token')}` }
          });
          if (bundleResp.status === 200) {
            const ct = bundleResp.headers && bundleResp.headers['content-type'];
            const ext = getExtFromContentType(ct) || '.zip';
            const filename = `${baseName}${ext}`;
            triggerDownload(bundleResp.data, filename, ct);
            notify('Downloaded bundled report for ' + baseName);
            return;
          }
        } catch (err) {
          console.warn('Bundle download failed, falling back to PDF-only or raw text', err);
        }
      }

      // 2) Try PDF-from-analysis (POST)
      if (analysis && Object.keys(analysis).length > 0) {
        try {
          const pdfResp = await axios.post('http://localhost:8000/api/export/pdf', analysis, {
            responseType: 'blob',
            headers: isGuest ? {} : { Authorization: `Bearer ${localStorage.getItem('token')}` }
          });
          if (pdfResp.status === 200) {
            const ct = pdfResp.headers && pdfResp.headers['content-type'];
            const ext = getExtFromContentType(ct) || '.pdf';
            const filename = `${baseName}${ext}`;
            triggerDownload(pdfResp.data, filename, ct);
            notify('Downloaded report for ' + baseName);
            return;
          }
        } catch (err) {
          console.warn('PDF-from-analysis failed, will try id-based export', err);
        }
      }

      // 3) Fallback: server-side export by document id
      try {
        const token = localStorage.getItem('token');
        const endpoint = isGuest
          ? `http://localhost:8000/api/documents/${documentId}/export-pdf?guest=true`
          : `http://localhost:8000/api/documents/${documentId}/export-pdf`;
        const headers = isGuest ? {} : { Authorization: `Bearer ${token}` };
        const response = await axios.get(endpoint, {
          responseType: 'blob',
          headers,
        });
        if (response.status === 200) {
          const ct = response.headers && response.headers['content-type'];
          const ext = getExtFromContentType(ct) || '.pdf';
          const filename = `${baseName}${ext}`;
          triggerDownload(response.data, filename, ct);
          notify('Downloaded report for ' + baseName);
          return;
        }
      } catch (err) {
        console.warn('ID-based export failed', err);
      }

      // 4) Final fallback: if we have raw_text, download as UTF-8 .txt (include BOM for Windows apps)
      if (analysis && analysis.raw_text) {
        try {
          const text = '\uFEFF' + analysis.raw_text; // prepend BOM for better compatibility
          const filename = `${baseName}.txt`;
          triggerDownload(text, filename, 'text/plain;charset=utf-8');
          notify('Downloaded raw analysis text for ' + baseName);
          return;
        } catch (err) {
          console.warn('Raw text download failed', err);
        }
      }

      // If nothing worked
      window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Failed to download report for ' + baseName }));
      alert('Download failed. Please try again.');
    } catch (error) {
      window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Failed to download report' }));
      alert('Download failed. Please try again.');
      console.error('Error exporting PDF:', error);
    }
  };

  const handleDelete = async (documentId, isGuest = false) => {
    setDeletingId(documentId);
    try {
      // If this is a guest-local entry (created client-side) its id is prefixed with 'g-'
      // In that case we only need to remove it from localStorage and state.
      if (isGuest && String(documentId).startsWith('g-')) {
        const stored = JSON.parse(localStorage.getItem('guest_history_v1') || '[]');
        const filtered = stored.filter(item => item.id !== documentId);
        localStorage.setItem('guest_history_v1', JSON.stringify(filtered));
        setHistory(prev => prev.filter(item => item.id !== documentId));
        setFullHistory(prev => prev.filter(item => item.id !== documentId));
        notify('Deleted guest document #' + documentId);
      } else {
        const endpoint = isGuest
          ? `http://localhost:8000/api/documents/${documentId}?guest=true`
          : `http://localhost:8000/api/documents/${documentId}`;
        const headers = isGuest ? {} : { Authorization: `Bearer ${localStorage.getItem('token')}` };
        await axios.delete(endpoint, { headers });
        setHistory(prev => prev.filter(item => item.id !== documentId));
        setFullHistory(prev => prev.filter(item => item.id !== documentId));
        notify('Deleted document #' + documentId);
      }
    } catch (error) {
      notify('Failed to delete document #' + documentId);
      console.error('Error deleting document:', error);
    } finally {
      setDeletingId(null);
    }
  };

  const formatTimeLeft = (timeLeft) => {
    const minutes = Math.floor(timeLeft / 60000);
    const seconds = Math.floor((timeLeft % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const getRiskColor = (score) => {
    if (score > 6) return 'text-destructive';
    if (score > 4) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getRiskIcon = (score) => {
    if (score > 6) return <XCircle className="w-5 h-5" />;
    if (score > 4) return <AlertTriangle className="w-5 h-5" />;
    return <CheckCircle className="w-5 h-5" />;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background px-8 pb-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-semibold mb-8 text-foreground">
            Document History
          </h1>
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
          </div>
        </div>
      </div>
    );
  }

  const pageTitle = type === 'guest' ? 'Guest History' : 'User History';

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto p-4">
        <div className="bg-card rounded-lg border border-border shadow-sm h-[80vh] flex flex-col">
          <div className="p-6 border-b border-border">
            <div>
              <h1 className="text-3xl font-semibold text-foreground">{pageTitle}</h1>
              <p className="text-muted-foreground text-sm mt-1">View and manage your analyzed documents.</p>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-6">
        <h1 className="text-3xl font-semibold mb-8 text-foreground sr-only">
          {pageTitle}
        </h1>
        <div className="mb-6">
          <input
            type="text"
            placeholder="Search documents by filename or summary..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2 border border-border rounded-lg bg-background text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
        {type === 'guest' && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <p className="text-yellow-800 text-sm">
              <AlertTriangle className="w-4 h-4 inline mr-2" />
              Guest uploads are temporary and visible to everyone for 10 minutes.
            </p>
          </div>
        )}
        {history.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24">
            <FileText className="w-20 h-20 mb-6 text-muted-foreground/40" />
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">No Documents Found</h2>
            <p className="text-base text-muted-foreground mb-2">
              {type === 'guest' ? 'No guest uploads available.' : (localStorage.getItem('token') ? 'No documents analyzed yet.' : 'Please log in to view your document history.')}
            </p>
            <p className="text-sm text-muted-foreground/70 mb-4">
              {type === 'guest' ? 'Guest uploads appear here temporarily.' : (localStorage.getItem('token') ? 'Start by uploading and analyzing a legal document.' : 'Log in to access your personal document history.')}
            </p>
            <button
              className="bg-primary text-primary-foreground px-6 py-3 rounded-lg shadow font-semibold text-base hover:bg-primary/90 transition-colors"
              onClick={() => window.location.href = '/'}
            >
              Go to Upload
            </button>
          </div>
        ) : (
          <div className="grid gap-6">
            {history.map((item) => {
              const analysis = item.analysis_result || {};
              const riskScore = analysis.riskScore || {};
              return (
                <div key={item.id} className="bg-card rounded-lg p-6 border border-border shadow-sm hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-start space-x-3">
                      <FileText className="w-6 h-6 text-primary mt-1" />
                      <div>
                        <h3 className="text-xl font-medium text-foreground mb-1">{type === 'guest' ? item.filename : `Analysis #${item.id}`}</h3>
                        <div className="flex items-center text-muted-foreground text-sm">
                          <Calendar className="w-4 h-4 mr-1" />
                          {new Date(type === 'guest' ? item.upload_time : item.created_at).toLocaleDateString()} at {new Date(type === 'guest' ? item.upload_time : item.created_at).toLocaleTimeString()}
                          {type === 'guest' && (
                            <span className="ml-4 flex items-center">
                              <Clock className="w-4 h-4 mr-1" />
                              Expires in: {formatTimeLeft(countdowns[item.id] || 0)}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      {type === 'user' ? (
                        <>
                          <button
                            onClick={() => {
                              setSelectedDocument(item);
                              notify('Opened analysis for document #' + item.id);
                              navigate('/report', { state: { analysisResult: analysis } });
                            }}
                            className="bg-secondary hover:bg-secondary/80 text-secondary-foreground px-4 py-2 rounded-md transition-colors flex items-center space-x-2"
                          >
                            <Eye className="w-4 h-4" />
                            <span>View Report</span>
                          </button>
                          <button
                            onClick={() => handleExportPDF(item.id, type === 'guest')}
                            className="bg-primary hover:bg-primary/90 text-primary-foreground px-4 py-2 rounded-md transition-colors flex items-center space-x-2"
                          >
                            <Download className="w-4 h-4" />
                            <span>Download</span>
                            {analysis && analysis.raw_text ? (
                              <span title="Bundle: includes PDF + full raw AI text (.txt)" aria-label="Bundle includes PDF and raw text" className="ml-2 inline-block text-xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded">bundle</span>
                            ) : null}
                          </button>
                          <button
                            onClick={() => handleDelete(item.id, type === 'guest')}
                            className={`bg-destructive hover:bg-destructive/90 text-destructive-foreground px-4 py-2 rounded-md transition-colors flex items-center space-x-2 ${deletingId === item.id ? 'opacity-50 cursor-not-allowed' : ''}`}
                            disabled={deletingId === item.id}
                          >
                            {deletingId === item.id ? (
                              <span className="animate-spin mr-2 w-4 h-4 border-b-2 border-white rounded-full"></span>
                            ) : (
                              <Trash2 className="w-4 h-4" />
                            )}
                            <span>Delete</span>
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            onClick={() => {
                              setSelectedDocument(item);
                              notify('Opened analysis for document #' + item.id);
                              navigate('/report', { state: { analysisResult: analysis } });
                            }}
                            className="bg-secondary hover:bg-secondary/80 text-secondary-foreground px-4 py-2 rounded-md transition-colors flex items-center space-x-2"
                          >
                            <Eye className="w-4 h-4" />
                            <span>View Report</span>
                          </button>
                          <button
                            onClick={() => handleExportPDF(item.id, true)}
                            className="bg-primary hover:bg-primary/90 text-primary-foreground px-4 py-2 rounded-md transition-colors flex items-center space-x-2"
                          >
                            <Download className="w-4 h-4" />
                            <span>Download</span>
                            {analysis && analysis.raw_text ? (
                              <span title="Bundle: includes PDF + full raw AI text (.txt)" aria-label="Bundle includes PDF and raw text" className="ml-2 inline-block text-xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded">bundle</span>
                            ) : null}
                          </button>
                          <button
                            onClick={() => handleDelete(item.id, true)}
                            className={`bg-destructive hover:bg-destructive/90 text-destructive-foreground px-4 py-2 rounded-md transition-colors flex items-center space-x-2 ${deletingId === item.id ? 'opacity-50 cursor-not-allowed' : ''}`}
                            disabled={deletingId === item.id}
                          >
                            {deletingId === item.id ? (
                              <span className="animate-spin mr-2 w-4 h-4 border-b-2 border-white rounded-full"></span>
                            ) : (
                              <Trash2 className="w-4 h-4" />
                            )}
                            <span>Delete</span>
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-muted/50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        {getRiskIcon(riskScore.score)}
                        <h4 className="font-medium text-foreground">Risk Score</h4>
                      </div>
                      <p className={`text-2xl font-bold ${getRiskColor(riskScore.score)}`}>{riskScore.score !== undefined ? riskScore.score : '--'}/10</p>
                      <p className="text-sm text-muted-foreground">{riskScore.label || 'No label'}</p>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <CheckCircle className="w-5 h-5 text-green-600" />
                        <h4 className="font-medium text-foreground">Strengths</h4>
                      </div>
                      <p className="text-2xl font-bold text-foreground">{Array.isArray(analysis.analysis?.strengths) ? analysis.analysis.strengths.length : 0}</p>
                      <p className="text-sm text-muted-foreground">identified</p>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <AlertTriangle className="w-5 h-5 text-destructive" />
                        <h4 className="font-medium text-foreground">Critical Flags</h4>
                      </div>
                      <p className="text-2xl font-bold text-foreground">{Array.isArray(analysis.criticalFlags) ? analysis.criticalFlags.length : 0}</p>
                      <p className="text-sm text-muted-foreground">issues found</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {selectedDocument && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <div className="bg-card rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-border">
              <div className="p-6 border-b border-border">
                <div className="flex justify-between items-center">
                  <h2 className="text-2xl font-semibold text-foreground">Analysis Details</h2>
                  <button
                    onClick={() => setSelectedDocument(null)}
                    className="text-muted-foreground hover:text-foreground text-2xl transition-colors"
                  >
                    ×
                  </button>
                </div>
              </div>
              <div className="p-6">
                {selectedDocument.analysis_result ? (
                  <div className="space-y-4">
                    <div>
                      <span className="font-bold">Summary:</span>
                      <p className="text-sm text-foreground bg-muted p-2 rounded-md">{selectedDocument.analysis_result?.summary}</p>
                    </div>
                    <div>
                      <span className="font-bold">Risk Score:</span>
                      <p className="text-sm text-foreground bg-muted p-2 rounded-md">{selectedDocument.analysis_result?.riskScore?.score} / {selectedDocument.analysis_result?.riskScore?.max} ({selectedDocument.analysis_result?.riskScore?.label})</p>
                    </div>
                    <div>
                      <span className="font-bold">Strengths:</span>
                      <ul className="list-disc ml-6 text-sm text-foreground">
                        {selectedDocument.analysis_result?.analysis?.strengths?.map((s, i) => <li key={i}>{s.text}</li>)}
                      </ul>
                    </div>
                    <div>
                      <span className="font-bold">Weaknesses:</span>
                      <ul className="list-disc ml-6 text-sm text-foreground">
                        {selectedDocument.analysis_result?.analysis?.weaknesses?.map((w, i) => <li key={i}>{w.text}</li>)}
                      </ul>
                    </div>
                    <div>
                      <span className="font-bold">Opportunities:</span>
                      <ul className="list-disc ml-6 text-sm text-foreground">
                        {selectedDocument.analysis_result?.analysis?.opportunities?.map((o, i) => <li key={i}>{o.text}</li>)}
                      </ul>
                    </div>
                    <div>
                      <span className="font-bold">Threats:</span>
                      <ul className="list-disc ml-6 text-sm text-foreground">
                        {selectedDocument.analysis_result?.analysis?.threats?.map((t, i) => <li key={i}>{t.text}</li>)}
                      </ul>
                    </div>
                    <div>
                      <span className="font-bold">Critical Flags:</span>
                      <ul className="list-disc ml-6 text-sm text-foreground">
                        {selectedDocument.analysis_result?.criticalFlags?.map((flag, i) => <li key={i}>{flag.title}: {flag.explanation}</li>)}
                      </ul>
                    </div>
                    <div>
                      <span className="font-bold">Negotiation Points:</span>
                      <ul className="list-disc ml-6 text-sm text-foreground">
                        {selectedDocument.analysis_result?.negotiationPoints?.map((point, i) => <li key={i}>{point.title}: {point.example}</li>)}
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-muted-foreground p-8">No analysis result available for this document.</div>
                )}
              </div>
            </div>
          </div>
        )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default HistoryPage;
