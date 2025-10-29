import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Download, FileText, TrendingUp, AlertTriangle, CheckCircle, XCircle, Plus, Shield, Target, Lightbulb, Zap, Star, Award, Clock, Users } from 'lucide-react';

const ReportPage = ({ analysisResult }) => {
  const location = useLocation();
  const [currentAnalysisResult, setCurrentAnalysisResult] = useState(analysisResult);
  const [activeSection, setActiveSection] = useState('summary');
  const sections = [
    { key: 'summary', label: 'Summary' },
    { key: 'risk', label: 'Risk Assessment' },
    { key: 'swot', label: 'SWOT Analysis' },
    { key: 'flags', label: 'Critical Flags' },
    { key: 'negotiation', label: 'Negotiation Plan' }
  ];

  useEffect(() => {
    if (location.state?.analysisResult) {
      setCurrentAnalysisResult(location.state.analysisResult);
    } else {
      setCurrentAnalysisResult(analysisResult);
    }
  }, [location.state, analysisResult]);

  const reportData = currentAnalysisResult;
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownloadPDF = async () => {
    setIsDownloading(true);
    try {
      const response = await fetch('http://localhost:8000/api/export/pdf', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(reportData),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'legal-analysis-report.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert('Failed to download PDF. Please try again.');
      }
    } catch (error) {
      console.error('Error downloading PDF:', error);
      alert('Error downloading PDF. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  const renderSection = () => {
    if (!reportData) return null;
    switch (activeSection) {
      case 'summary':
        return (
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-2">Document Summary</h2>
            <p className="text-sm text-gray-700 bg-gray-50 rounded-lg p-3 border border-gray-200">{reportData.summary || 'No summary available.'}</p>
          </div>
        );
      case 'risk':
        if (!reportData.riskScore) {
          return <div className="p-4 text-xs text-gray-500">No risk score available.</div>;
        }
        return (
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-2">Risk Assessment Score</h2>
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="text-red-400" size={18} />
              <span className="text-xs font-bold">{reportData.riskScore.score} / {reportData.riskScore.max}</span>
              <span className="text-xs text-gray-500">{reportData.riskScore.label}</span>
            </div>
            <p className="text-xs text-gray-700 bg-gray-50 rounded-lg p-3 border border-gray-200">{reportData.riskScore.description || 'No description available.'}</p>
          </div>
        );
      case 'swot':
        if (!reportData.analysis) {
          return <div className="p-4 text-xs text-gray-500">No SWOT analysis available.</div>;
        }
        return (
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-2">SWOT Analysis</h2>
            <div className="space-y-2">
              <div>
                <span className="font-bold text-green-600">Strengths:</span>
                <ul className="ml-2 text-xs">
                  {(reportData.analysis.strengths || []).map((s, i) => <li key={i}>{s.text}</li>)}
                </ul>
              </div>
              <div>
                <span className="font-bold text-yellow-600">Weaknesses:</span>
                <ul className="ml-2 text-xs">
                  {(reportData.analysis.weaknesses || []).map((w, i) => <li key={i}>{w.text}</li>)}
                </ul>
              </div>
              <div>
                <span className="font-bold text-blue-600">Opportunities:</span>
                <ul className="ml-2 text-xs">
                  {(reportData.analysis.opportunities || []).map((o, i) => <li key={i}>{o.text}</li>)}
                </ul>
              </div>
              <div>
                <span className="font-bold text-red-600">Threats:</span>
                <ul className="ml-2 text-xs">
                  {(reportData.analysis.threats || []).map((t, i) => <li key={i}>{t.text}</li>)}
                </ul>
              </div>
            </div>
          </div>
        );
      case 'flags':
        if (!reportData.criticalFlags) {
          return <div className="p-4 text-xs text-gray-500">No critical flags found.</div>;
        }
        return (
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-2">Critical Red Flags</h2>
            {reportData.criticalFlags.length > 0 ? (
              <ul className="space-y-2 text-xs">
                {reportData.criticalFlags.map((flag, i) => (
                  <li key={i} className="bg-red-50 border border-red-200 rounded-lg p-2">
                    <span className="font-bold text-red-600">{flag.title}:</span> {flag.explanation}
                  </li>
                ))}
              </ul>
            ) : <p className="text-xs text-gray-500">No critical flags found.</p>}
          </div>
        );
      case 'negotiation':
        if (!reportData.negotiationPoints) {
          return <div className="p-4 text-xs text-gray-500">No negotiation points found.</div>;
        }
        return (
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-2">Negotiation Action Plan</h2>
            {reportData.negotiationPoints.length > 0 ? (
              <ul className="space-y-2 text-xs">
                {reportData.negotiationPoints.map((point, i) => (
                  <li key={i} className="bg-blue-50 border border-blue-200 rounded-lg p-2">
                    <span className="font-bold text-blue-600">{point.title}:</span> {point.example}
                  </li>
                ))}
              </ul>
            ) : <p className="text-xs text-gray-500">No negotiation points found.</p>}
          </div>
        );
      default:
        return null;
    }
  };

  if (!reportData) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading report...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 text-foreground flex flex-col items-center">
      <header className="bg-white/80 py-6 w-full shadow-xl backdrop-blur-lg border-b border-gray-200">
        <div className="max-w-xl mx-auto px-2 text-center">
          <h1 className="text-xl font-bold text-gray-900 tracking-tight mb-2">CovenantAI Analysis Report</h1>
          <p className="text-xs text-gray-500 mb-2">Your comprehensive legal document analysis powered by CovenantAI.</p>
        </div>
      </header>
      <div className="max-w-xl w-full mx-auto px-2 py-4 flex flex-col md:flex-row gap-6">
        <nav className="md:w-40 w-full flex md:flex-col flex-row gap-2 md:gap-3 mb-4 md:mb-0">
          {sections.map((section) => (
            <button
              key={section.key}
              className={`w-full px-3 py-2 rounded-lg text-xs font-medium border transition-colors duration-200 text-left md:text-center md:justify-center flex items-center gap-2 md:gap-0 ${activeSection === section.key ? 'bg-blue-600 text-white border-blue-600 shadow' : 'bg-white text-gray-700 border-gray-300 hover:bg-blue-50'}`}
              onClick={() => setActiveSection(section.key)}
            >
              {section.label}
            </button>
          ))}
        </nav>
        <div className="flex-1 rounded-xl bg-white/80 shadow p-2 border border-gray-200">
          {renderSection()}
        </div>
      </div>
    </div>
  );
};

export default ReportPage;
