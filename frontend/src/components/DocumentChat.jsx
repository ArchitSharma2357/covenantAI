import React from 'react';

const DocumentChat = ({ selectedDocument, token, refreshDocuments }) => {
  if (!selectedDocument) return <div className="p-6">No document selected.</div>;

  const title = selectedDocument.filename || selectedDocument.id || 'Document';

  return (
    <div className="flex-1 p-6 overflow-auto">
      <h2 className="text-2xl font-semibold mb-4">{title}</h2>
      <div className="bg-gray-900 rounded-lg p-4 text-sm whitespace-pre-wrap">
        {selectedDocument.document_text || (selectedDocument.analysis && selectedDocument.analysis.raw_text) || 'No document text available.'}
      </div>
      <div className="mt-4 text-xs text-muted-foreground">This is a lightweight DocumentChat placeholder. For full chat features, ensure the original DocumentChat component is restored.</div>
    </div>
  );
};

export default DocumentChat;
