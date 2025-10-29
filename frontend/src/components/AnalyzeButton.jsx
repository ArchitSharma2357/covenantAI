import React from 'react';
import { Play, BarChart3 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const AnalyzeButton = ({ onAnalyze, loading }) => {
  const navigate = useNavigate();

  const handleAnalyze = async () => {
    // Just call onAnalyze - let App.js handle navigation after history is saved
    await onAnalyze();
  };

  return (
    <div className="text-center">
      <button
        onClick={handleAnalyze}
        disabled={loading}
        className={`inline-flex items-center gap-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 ${
          loading
            ? 'bg-muted text-muted-foreground cursor-not-allowed'
            : 'bg-primary text-primary-foreground hover:bg-primary/90'
        }`}
      >
        {loading ? (
          <>
            <div className="w-5 h-5 border-2 border-muted-foreground border-t-transparent rounded-full animate-spin"></div>
            Analyzing Document...
          </>
        ) : (
          <>
            <BarChart3 className="w-5 h-5" />
            Analyze Document
          </>
        )}
      </button>
      <p className="mt-4 text-sm text-muted-foreground">
        Your document will be processed securely with AI technology
      </p>
    </div>
  );
};

export default AnalyzeButton;
