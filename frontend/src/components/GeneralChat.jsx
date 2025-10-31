import React from 'react';

const GeneralChat = ({ token }) => {
  return (
    <div className="flex-1 p-6 overflow-auto">
      <h2 className="text-2xl font-semibold mb-4">General Chat</h2>
      <div className="bg-gray-900 rounded-lg p-4 text-sm">
        <p>This is a lightweight GeneralChat placeholder. Full chat features are not included in this placeholder.</p>
        <p className="mt-2 text-xs text-muted-foreground">Token present: {token ? 'yes' : 'no'}</p>
      </div>
    </div>
  );
};

export default GeneralChat;
