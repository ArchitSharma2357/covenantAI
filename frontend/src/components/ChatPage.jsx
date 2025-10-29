import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, Bot, User as UserIcon, FileText, Upload, Trash2 } from 'lucide-react';

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatMode, setChatMode] = useState('general'); // 'general' or 'document'
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [expandedMessages, setExpandedMessages] = useState({});
  const messagesEndRef = useRef(null);
  const loadingTimeoutRef = useRef(null);

  // Load chat history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
      try {
        const history = JSON.parse(savedHistory);
        if (history.messages && history.messages.length > 0) {
          setMessages(history.messages);
          setChatMode(history.chatMode || 'general');
          if (history.selectedDocument) {
            setSelectedDocument(history.selectedDocument);
          }
        }
      } catch (error) {
        console.error('Error loading chat history:', error);
      }
    }
  }, []);

  // Save chat history to localStorage whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      const history = {
        messages,
        chatMode,
        selectedDocument,
        timestamp: Date.now()
      };
      localStorage.setItem('chatHistory', JSON.stringify(history));
    }
  }, [messages, chatMode, selectedDocument]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const inputRef = useRef(null);

  // Fetch user's documents on component mount
  useEffect(() => {
    const fetchDocuments = async () => {
      const token = localStorage.getItem('token');
      setLoadingDocuments(true);
      try {
        if (token) {
          const response = await axios.get('http://localhost:8000/api/documents', {
            headers: { Authorization: `Bearer ${token}` }
          });
          setDocuments(response.data || []);
          // Don't auto-select first document unless we're in document mode
          if (chatMode === 'document' && response.data && response.data.length > 0) {
            setSelectedDocument(response.data[0]);
          }
        } else {
          // Try guest-visible documents from history endpoint (only analysis, not uploaded docs)
          try {
            const resp = await axios.get('http://localhost:8000/api/history/guest');
            // history returns records; map to a lightweight doc shape
            const docs = (resp.data || []).map(h => ({
              id: h.id || h._id || h.document_id || h.file_id || h.filename || (h.upload_id || ''),
              filename: h.filename || h.original_filename || (h.summary && h.summary.slice ? (h.summary.slice(0, 30) + '...') : 'Document'),
              analysis: h.analysis_result || h.analysis || (h.summary ? { raw_text: h.summary } : null)
            }));
            setDocuments(docs);
            if (docs.length > 0) setSelectedDocument(docs[0]);
          } catch (guestErr) {
            // ignore guest fetch errors
          }
        }
      } catch (error) {
        console.error('Error fetching documents:', error);
      } finally {
        setLoadingDocuments(false);
      }
    };

    fetchDocuments();
  }, []);

  // Auto-select first document when switching to document mode if none selected
  useEffect(() => {
    if (chatMode === 'document' && !selectedDocument && documents.length > 0) {
      setSelectedDocument(documents[0]);
    }
  }, [chatMode, documents, selectedDocument]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    // Check if document mode is selected but no document is chosen
    if (chatMode === 'document' && !selectedDocument) {
      const errorMessage = { role: 'assistant', content: 'Please select a document first or upload a new document to analyze.' };
      setMessages(prev => [...prev, errorMessage]);
      return;
    }

    // If in document mode, prefix the question with the document name for clarity
    let processedQuestion = inputMessage;
    if (chatMode === 'document' && selectedDocument) {
      processedQuestion = `Regarding the document "${selectedDocument.filename}", ${inputMessage}`;
    }

    // Always attempt backend analysis for selected document; let backend handle fallback/empty cases

    const userMessage = { role: 'user', content: processedQuestion };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);
    // start a safety timeout to avoid indefinite "Thinking..."
    if (loadingTimeoutRef.current) clearTimeout(loadingTimeoutRef.current);
    loadingTimeoutRef.current = setTimeout(() => {
      if (loading) {
        setLoading(false);
        setMessages(prev => [...prev, { role: 'assistant', content: 'The AI is taking longer than expected. Please try again or retry your question.' }]);
      }
    }, 60000); // 60s

    try {
      const token = localStorage.getItem('token');
      const headers = token ? { Authorization: `Bearer ${token}` } : {};

      let response;
      if (chatMode === 'document' && selectedDocument) {
        // Log the selected document and its content for debugging
        console.debug('Selected document:', selectedDocument);
        console.debug('Document analysis:', selectedDocument.analysis);

        // For authenticated users with server-stored documents, try server-side ask first
        if (token && selectedDocument.id && !selectedDocument.id.toString().startsWith('inline_')) {
          try {
            response = await axios.post(`http://localhost:8000/api/documents/ask`, {
              document_id: selectedDocument.id,
              question: inputMessage,
              session_id: `chat_${selectedDocument.id}_${Date.now()}`
            }, {
              headers: headers
            });
            const answer = (response.data && response.data.answer) ? response.data.answer : '';
            const asksForDocument = /provide.*document|please provide.*document|please paste|please upload/i.test(answer);
            if (asksForDocument) {
              setMessages(prev => [...prev, { role: 'assistant', content: 'I could not access the document text on the server. Please upload the document again, or paste the text here so I can summarize it.' }]);
            } else {
              setMessages(prev => [...prev, { role: 'assistant', content: answer }]);
            }
          } catch (err) {
            // fallback to inline using provided analysis
            const respInline = await axios.post('http://localhost:8000/api/documents/ask/inline', {
              question: inputMessage,
              analysis: selectedDocument.analysis,
              document_id: selectedDocument.id
            }, { headers });
            const inlineAnswer = respInline?.data?.answer || '';
            const inlineAsksForDoc = /provide.*document|please provide.*document|please paste|please upload/i.test(inlineAnswer);
            if (inlineAsksForDoc) {
              setMessages(prev => [...prev, { role: 'assistant', content: 'I could not extract text from the selected document. Please upload a document that contains text, or paste the text here.' }]);
            } else {
              setMessages(prev => [...prev, { role: 'assistant', content: inlineAnswer }]);
            }
          }
        } else {
          // Inline ask for guest documents or newly uploaded inline docs (no token or inline document)
          const docText = selectedDocument.analysis?.raw_text ||
                        selectedDocument.document_text ||
                        selectedDocument.summary ||
                        (selectedDocument.analysis?.summary) || '';

          console.debug('Document text length:', docText.length);
          console.debug('Sending document_id:', selectedDocument.id);
          const respInline = await axios.post('http://localhost:8000/api/documents/ask/inline', {
            document_text: docText,
            document_id: selectedDocument.id,
            question: processedQuestion
          }, { headers });
          const inlineAnswer = respInline?.data?.answer || '';
          const inlineAsksForDoc = /provide.*document|please provide.*document|please paste|please upload/i.test(inlineAnswer);
          if (inlineAsksForDoc) {
            setMessages(prev => [...prev, { role: 'assistant', content: 'I could not extract text from the selected document. Please upload a document that contains text, or paste the text here.' }]);
          } else {
            setMessages(prev => [...prev, { role: 'assistant', content: inlineAnswer }]);
          }
        }
      } else {
        // Use general chat endpoint
        response = await axios.post('http://localhost:8000/api/chat', {
          message: inputMessage,
          context: messages.slice(-5), // Send last 5 messages for context
          mode: chatMode // Send chat mode to backend
        }, {
          headers: headers
        });
        const aiMessage = { role: 'assistant', content: response.data.response };
        setMessages(prev => [...prev, aiMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      if (error.response?.status === 401) {
        localStorage.removeItem('token');
        const errorMessage = { role: 'assistant', content: 'Your session has expired. Please log in again.' };
        setMessages(prev => [...prev, errorMessage]);
      } else {
        const errorMessage = { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' };
        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current);
        loadingTimeoutRef.current = null;
      }
      setLoading(false);
    }
  };

  // Upload & Chat: Upload file then ask a question about it immediately
  const fileInputRef = React.useRef(null);

  const handleUploadAndAsk = async (file) => {
    if (!file) return;
    setLoading(true);
    if (loadingTimeoutRef.current) clearTimeout(loadingTimeoutRef.current);
    loadingTimeoutRef.current = setTimeout(() => {
      if (loading) {
        setLoading(false);
        setMessages(prev => [...prev, { role: 'assistant', content: 'Processing the uploaded document is taking longer than expected. Please try again in a moment.' }]);
      }
    }, 60000);
    try {
      const token = localStorage.getItem('token');
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      const formData = new FormData();
      formData.append('file', file);

      const uploadResp = await axios.post('http://localhost:8000/api/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data', ...(headers || {}) }
      });

      // The upload endpoint returns a document identifier; trigger analysis immediately (synchronous) so chat can ask about it now
      const created = uploadResp.data || {};
      const documentId = created.document_id || created.id || created._id || `inline_${Date.now()}`;

      // Try to fetch analysis immediately (analyze endpoint performs extraction and analysis)
      let analysisResult = created.analysis_result || created.analysis || null;
      let documentText = created.text || created.document_text || null;
      try {
        const analyzeResp = await axios.post(`http://localhost:8000/api/documents/${documentId}/analyze`);
        analysisResult = analyzeResp.data || analysisResult;
      } catch (anErr) {
        // If analysis endpoint fails, continue with whatever data upload returned (likely minimal)
        console.debug('Immediate analyze after upload failed or is pending:', anErr?.message || anErr);
      }

      // Prefer raw text from analysis if available
      if (!documentText && analysisResult) {
        documentText = analysisResult.raw_text || analysisResult.document_text || null;
      }

      const lightweight = {
        id: documentId,
        filename: created.filename || file.name,
        analysis: analysisResult || (created.summary ? { raw_text: created.summary } : null),
        document_text: documentText || null
      };

      // Add to local documents list and select
      setDocuments(prev => [lightweight, ...prev]);
      setSelectedDocument(lightweight);

      // If user has a typed question, run inline ask against the analysis/text
      if (inputMessage && inputMessage.trim()) {
        try {
          const respInline = await axios.post('http://localhost:8000/api/documents/ask/inline', {
            question: inputMessage,
            analysis: lightweight.analysis,
            document_text: lightweight.document_text
          }, { headers });

          setMessages(prev => [...prev, { role: 'assistant', content: respInline.data.answer }]);
          setInputMessage('');
        } catch (askErr) {
          console.error('Inline ask after upload failed', askErr);
          setMessages(prev => [...prev, { role: 'assistant', content: 'Uploaded but could not process your question immediately. Please try asking again in a moment.' }]);
        }
      } else {
        // Set chat mode to document and show welcome message
        setChatMode('document');
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `${file.name} is ready for analysis. You can now ask questions about this document. For example:\n- Explain the key legal terms\n- What are the main obligations?\n- Identify potential risks\n- Summarize the document`
        }]);
      }
    } catch (err) {
      console.error('Upload & ask failed', err);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Upload failed. Please try again.' }]);
    } finally {
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current);
        loadingTimeoutRef.current = null;
      }
      setLoading(false);
    }
  };

  const clearChatHistory = () => {
    setMessages([]);
    setChatMode('general');
    setSelectedDocument(null);
    localStorage.removeItem('chatHistory');
  };

  const handleDeleteDocument = async (docId) => {
    try {
      const token = localStorage.getItem('token');
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      await axios.delete(`http://localhost:8000/api/documents/${docId}`, { headers });
    } catch (error) {
      console.error('Error deleting document:', error);
      // Continue to remove from local state even if API call fails
    }
    setDocuments(prev => prev.filter(doc => doc.id !== docId));
    if (selectedDocument?.id === docId) {
      setSelectedDocument(null);
      // Clear localStorage to prevent deleted document from reappearing
      const savedHistory = localStorage.getItem('chatHistory');
      if (savedHistory) {
        try {
          const history = JSON.parse(savedHistory);
          if (history.selectedDocument?.id === docId) {
            history.selectedDocument = null;
            localStorage.setItem('chatHistory', JSON.stringify(history));
          }
        } catch (e) {
          // Silently fail if localStorage handling fails
        }
      }
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto p-4">
        <div className="bg-card rounded-lg border border-border shadow-sm h-[80vh] flex flex-col">
          {/* Header */}
          <div className="p-6 border-b border-border">
            <div className="flex justify-between items-start">
              <div>
                <h1 className="text-2xl font-semibold text-foreground">
                  AI Legal Assistant
                </h1>
                <p className="text-muted-foreground text-sm mt-1">
                  Ask questions about legal documents, contracts, or legal matters
                </p>
                {chatMode === 'document' && selectedDocument && (
                  <div className="mt-2 text-xs text-muted-foreground flex items-center gap-3">
                    <span>About: <strong className="text-foreground">{selectedDocument.filename}</strong></span>
                    <button
                      onClick={() => setInputMessage('Give me a concise plain-English summary of the selected document.')}
                      className="text-xs text-primary hover:underline"
                    >Use sample question</button>
                  </div>
                )}
              </div>
              {messages.length > 0 && (
                <button
                  onClick={clearChatHistory}
                  className="flex items-center space-x-2 px-3 py-2 text-sm text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded-md transition-colors"
                  title="Clear chat history"
                >
                  <Trash2 className="w-4 h-4" />
                  <span className="hidden sm:inline">Clear</span>
                </button>
              )}
            </div>

            {/* Chat Mode Selector */}
            <div className="mt-4 space-y-4">
              <div className="flex items-center space-x-4">
                <span className="text-sm font-medium text-foreground">Chat Mode:</span>
                <div className="flex rounded-md border border-border">
                  <button
                    onClick={() => setChatMode('general')}
                    className={`px-4 py-2 text-sm font-medium rounded-l-md transition-colors ${
                      chatMode === 'general'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-background text-muted-foreground hover:bg-muted'
                    }`}
                  >
                    General Legal
                  </button>
                  <button
                    onClick={() => setChatMode('document')}
                    className={`px-4 py-2 text-sm font-medium rounded-r-md transition-colors ${
                      chatMode === 'document'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-background text-muted-foreground hover:bg-muted'
                    }`}
                  >
                    About My Document
                  </button>
                </div>
              </div>

              {/* Document Selector - Only show in document mode */}
              {chatMode === 'document' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-foreground">Document:</span>
                    <div>
                      <button onClick={() => fileInputRef.current?.click()} className="text-sm text-primary hover:underline">Upload document</button>
                    </div>
                  </div>

                  {loadingDocuments ? (
                    <div className="flex items-center space-x-2 text-muted-foreground">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                      <span className="text-sm">Loading documents...</span>
                    </div>
                  ) : documents.length === 0 ? (
                    <div className="flex items-center space-x-3 bg-muted/50 rounded-md px-4 py-2">
                      <Upload className="w-4 h-4 text-muted-foreground" />
                      <div className="text-sm text-muted-foreground">
                        <span className="font-medium">No documents uploaded yet.</span>
                        <div>Upload a document to start chatting about it.</div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex gap-3 overflow-x-auto py-2">
                        {documents.map((doc) => (
                          <div
                            key={doc.id}
                            onClick={() => { console.debug('Document selected (click):', doc); setSelectedDocument(doc); }}
                            className={`relative min-w-[200px] max-w-[220px] p-3 rounded-lg border cursor-pointer transition-shadow flex flex-col justify-between h-full ${selectedDocument?.id === doc.id ? 'border-primary shadow-lg bg-primary/5' : 'border-border bg-card'}`}
                          >
                            {selectedDocument?.id === doc.id && (
                              <div className="absolute -top-2 -right-2 bg-primary text-primary-foreground text-xs px-2 py-1 rounded-md shadow">Selected</div>
                            )}
                          <div className="flex items-start justify-between mb-2">
                            <div className="font-medium text-foreground truncate">{doc.filename}</div>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteDocument(doc.id);
                              }}
                              className="text-xs text-destructive hover:text-destructive/80 p-1"
                              title="Remove document"
                            >
                              <Trash2 className="w-3 h-3" />
                            </button>
                          </div>
                          <div className="text-xs text-muted-foreground mb-3 break-words">{(doc.analysis?.summary || doc.document_text || '').slice(0, 120)}</div>
                          <div className="flex items-center gap-2">
                            <button onClick={() => { console.debug('Document selected (button):', doc); setSelectedDocument(doc); }} className="text-sm px-2 py-1 rounded bg-secondary text-secondary-foreground">Select</button>
                            <button onClick={() => {
                              console.debug('Document selected via Ask now:', doc);
                              setChatMode('document');
                              setSelectedDocument(doc);
                              // focus the input textarea so user can type their question
                              setTimeout(() => { try { inputRef.current?.focus(); } catch (e) {} }, 50);
                            }} className="text-sm px-2 py-1 rounded bg-muted/20">Ask now</button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Hidden file input for upload & chat */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        handleUploadAndAsk(e.target.files[0]);
                        e.target.value = null;
                      }
                    }}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-card" style={{ minHeight: '52vh', maxHeight: '72vh' }}>
            {messages.length === 0 && (
              <div className="text-center text-muted-foreground py-12">
                <Bot className="w-16 h-16 mx-auto mb-4 text-muted-foreground/50" />
                <h3 className="text-xl font-medium mb-2">Welcome to Legal AI Chat</h3>
                <p className="text-sm max-w-md mx-auto mb-4">
                  Ask me questions about legal documents, contracts, compliance, or any legal matters.
                </p>
                {/* removed redundant upload prompt for document chat mode per UX request */}
              </div>
            )}

            {messages.map((message, index) => {
              const isAssistant = message.role === 'assistant';
              const isUser = message.role === 'user';
              const longMessage = message.content && message.content.length > 400;
              const isExpanded = !!expandedMessages[index];
              const bubbleMaxWidth = chatMode === 'document' && isAssistant ? 'max-w-[92%]' : 'max-w-[70%]';

              return (
                <div key={index} className={`flex items-start space-x-3 ${isUser ? 'justify-end space-x-reverse' : 'justify-start'}`}>
                  {isAssistant && (
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <Bot className="w-4 h-4 text-primary" />
                    </div>
                  )}

          <div className={`${bubbleMaxWidth} rounded-lg p-4 ${isUser ? 'bg-primary text-primary-foreground ml-auto' : 'bg-muted text-foreground'}`}>
            <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
                      {isAssistant && chatMode === 'document' && longMessage && !isExpanded
                        ? `${message.content.slice(0, 400)}...`
                        : message.content}
                    </p>

                    {isAssistant && chatMode === 'document' && longMessage && (
                      <button
                        onClick={() => setExpandedMessages(prev => ({ ...prev, [index]: !prev[index] }))}
                        className="mt-2 text-xs text-primary hover:underline"
                        aria-expanded={isExpanded}
                      >
                        {isExpanded ? 'Show less' : 'Show more'}
                      </button>
                    )}
                  </div>

                  {isUser && (
                    <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                      <UserIcon className="w-4 h-4 text-primary-foreground" />
                    </div>
                  )}
                </div>
              );
            })}

            {loading && (
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-primary" />
                </div>
                <div className="bg-muted rounded-lg p-4 max-w-[70%]">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                    <span className="text-muted-foreground text-sm">Thinking...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-6 border-t border-border">
            <form onSubmit={handleSendMessage} className="flex space-x-3">
              <div className="flex-1 relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder={chatMode === 'document'
                    ? (selectedDocument ? `Ask a question about \"${selectedDocument.filename}\"...` : 'Ask a question about the selected document...')
                    : 'Ask a legal question...'}
                  className="w-full border border-input rounded-md px-4 py-3 pr-12 bg-background text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
                  disabled={loading}
                />
              </div>
              <button
                type="submit"
                disabled={loading || !inputMessage.trim()}
                className="bg-primary hover:bg-primary/90 disabled:bg-muted text-primary-foreground disabled:text-muted-foreground px-4 py-3 rounded-md font-medium transition-colors disabled:cursor-not-allowed flex items-center space-x-2"
              >
                <Send className="w-4 h-4" />
                <span className="hidden sm:inline">Send</span>
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
