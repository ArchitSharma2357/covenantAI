import React, { useState, useEffect } from "react";
import axios from "axios";
import Sidebar from "../components/Sidebar";
import DocumentChat from "../components/DocumentChat";
import GeneralChat from "../components/GeneralChat";

const ChatPage = () => {
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [token, setToken] = useState(localStorage.getItem("token") || null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // ✅ Unified fetchDocuments function with fixes applied
  const fetchDocuments = async () => {
    try {
      if (token) {
        // Authenticated user → Fetch using Authorization header
        const response = await axios.get("http://localhost:8000/api/documents", {
          headers: { Authorization: `Bearer ${token}` },
        });

        // ✅ FIXED: Removed malformed spread, properly mapping analysis
        const mappedDocs = (response.data || []).map((doc) => ({
          ...doc,
          analysis: {
            raw_text: doc.raw_text || doc.content,
            document_text: doc.document_text || doc.content,
            summary: doc.summary,
          },
        }));

        setDocuments(mappedDocs);
        if (mappedDocs.length > 0) setSelectedDocument(mappedDocs[0]);
      } else {
        // ✅ FIXED: Guest users can now fetch documents too
        try {
          const response = await axios.get("http://localhost:8000/api/documents");

          const mappedDocs = (response.data || []).map((doc) => ({
            ...doc,
            analysis: {
              raw_text: doc.raw_text || doc.content,
              document_text: doc.document_text || doc.content,
              summary: doc.summary,
            },
          }));

          setDocuments(mappedDocs);
          if (mappedDocs.length > 0) setSelectedDocument(mappedDocs[0]);
        } catch (guestErr) {
          // ✅ Fallback for guest users
          try {
            const resp = await axios.get("http://localhost:8000/api/history/guest");
            const docs = (resp.data || []).map((h) => ({
              id:
                h.id ||
                h._id ||
                h.document_id ||
                h.file_id ||
                h.filename ||
                h.upload_id ||
                "",
              filename:
                h.filename ||
                h.original_filename ||
                (h.summary && h.summary.slice
                  ? h.summary.slice(0, 30) + "..."
                  : "Document"),
              analysis:
                h.analysis_result ||
                h.analysis ||
                (h.summary ? { raw_text: h.summary } : null),
            }));
            setDocuments(docs);
            if (docs.length > 0) setSelectedDocument(docs[0]);
          } catch (historyErr) {
            console.error("Guest document fallback failed:", historyErr);
          }
        }
      }
    } catch (err) {
      console.error("Document fetch error:", err);
    }
  };

  // ✅ Re-fetch documents when token changes or first load
  useEffect(() => {
    fetchDocuments();
  }, [token]);

  const handleDocumentSelect = (doc) => {
    setSelectedDocument(doc);
  };

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white">
      {/* Sidebar */}
      <Sidebar
        documents={documents}
        onDocumentSelect={handleDocumentSelect}
        selectedDocument={selectedDocument}
        isSidebarOpen={isSidebarOpen}
        setIsSidebarOpen={setIsSidebarOpen}
      />

      {/* Main Chat Window */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {selectedDocument ? (
          <DocumentChat
            selectedDocument={selectedDocument}
            token={token}
            refreshDocuments={fetchDocuments}
          />
        ) : (
          <GeneralChat token={token} />
        )}
      </div>
    </div>
  );
};

export default ChatPage;
