import React, { useState } from 'react';
import axios from 'axios';
import { Upload, GitCompare, CheckCircle, XCircle, AlertTriangle, ArrowRight } from 'lucide-react';

const ComparePage = () => {
  const [documents, setDocuments] = useState([null, null]);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileSelect = (index, file) => {
    const newDocs = [...documents];
    newDocs[index] = file;
    setDocuments(newDocs);
  };

  const handleCompare = async () => {
    if (!documents[0] || !documents[1]) {
      alert('Please select both documents to compare');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('document1', documents[0]);
    formData.append('document2', documents[1]);

    try {
      const client = axios.create();
      // ensure no auth header carried through
      try { delete client.defaults.headers.common['Authorization']; } catch (e) {}

      const res = await client.post('http://localhost:8000/api/compare', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setComparison(res.data);
    } catch (err) {
      console.error(err);
      alert('Error comparing documents');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setDocuments([null, null]);
    setComparison(null);
  };

  const getRiskColor = (score) => {
    if (score >= 7) return 'text-destructive';
    if (score >= 5) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getRiskIcon = (score) => {
    if (score >= 7) return <AlertTriangle className="w-5 h-5 text-destructive" />;
    if (score >= 5) return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
    return <CheckCircle className="w-5 h-5 text-green-600" />;
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-6xl mx-auto p-4">
        <div className="bg-card rounded-lg border border-border shadow-sm h-[80vh] flex flex-col">
          <div className="p-6 border-b border-border">
            <div>
              <h1 className="text-2xl font-semibold text-foreground">Compare Documents</h1>
              <p className="text-muted-foreground text-sm mt-1">Compare two legal documents side-by-side and highlight differences or conflicts.</p>
            </div>
          </div>

          <div className="flex-1 p-6 overflow-hidden">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-full">
              {/* Document A */}
              <div className="h-full flex flex-col">
                <div className="flex-1 overflow-auto bg-background p-4 border border-border rounded-lg flex items-center justify-center">
                  {documents[0] ? (
                    <div className="text-sm text-foreground">{documents[0].name}</div>
                  ) : (
                    <div className="text-sm text-muted-foreground">Select or upload the first document to compare.</div>
                  )}
                </div>
                <div className="mt-4">
                  <input id="doc1" type="file" accept=".pdf,.doc,.docx,.txt" className="hidden" onChange={(e) => handleFileSelect(0, e.target.files[0])} />
                  <label htmlFor="doc1" className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4 text-center cursor-pointer hover:border-primary/50 transition-colors flex items-center justify-center gap-3">
                    <Upload className="w-8 h-8 text-muted-foreground" />
                    <span className="text-sm text-foreground">{documents[0] ? documents[0].name : 'Select first document'}</span>
                  </label>
                </div>
              </div>

              {/* Document B */}
              <div className="h-full flex flex-col">
                <div className="flex-1 overflow-auto bg-background p-4 border border-border rounded-lg flex items-center justify-center">
                  {documents[1] ? (
                    <div className="text-sm text-foreground">{documents[1].name}</div>
                  ) : (
                    <div className="text-sm text-muted-foreground">Select or upload the second document to compare.</div>
                  )}
                </div>
                <div className="mt-4">
                  <input id="doc2" type="file" accept=".pdf,.doc,.docx,.txt" className="hidden" onChange={(e) => handleFileSelect(1, e.target.files[0])} />
                  <label htmlFor="doc2" className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4 text-center cursor-pointer hover:border-primary/50 transition-colors flex items-center justify-center gap-3">
                    <Upload className="w-8 h-8 text-muted-foreground" />
                    <span className="text-sm text-foreground">{documents[1] ? documents[1].name : 'Select second document'}</span>
                  </label>
                </div>
              </div>
            </div>
          </div>

          <div className="p-4 border-t border-border flex items-center justify-end gap-2">
            <button onClick={handleCompare} disabled={loading || !documents[0] || !documents[1]} className="px-4 py-2 rounded-md bg-primary text-primary-foreground disabled:opacity-60">{loading ? 'Comparing...' : 'Compare'}</button>
            <button onClick={handleReset} className="px-4 py-2 rounded-md bg-background border border-border">Reset</button>
          </div>
        </div>
      </div>

      {/* Results area below */}
      <div className="max-w-6xl mx-auto p-4">
        {comparison && (
          <div className="bg-card rounded-lg p-6 border border-border shadow-sm mt-6">
            <h3 className="text-2xl font-semibold text-foreground mb-6 flex items-center space-x-2">
              <GitCompare className="w-6 h-6 text-primary" />
              <span>Comparison Results</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
              <div>
                <h4 className="text-xl font-medium text-green-600 mb-4 flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5" />
                  <span>Similarities</span>
                </h4>
                <div className="space-y-3">
                  {comparison.similarities && comparison.similarities.length > 0 ? (
                    comparison.similarities.map((s, i) => (
                      <div key={i} className="bg-green-50 border border-green-200 rounded-md p-3">
                        <p className="text-green-800 text-sm">{s}</p>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-8">
                      <CheckCircle className="w-12 h-12 mx-auto mb-2 text-muted-foreground/50" />
                      <p className="text-muted-foreground">No similarities found</p>
                    </div>
                  )}
                </div>
              </div>

              <div>
                <h4 className="text-xl font-medium text-destructive mb-4 flex items-center space-x-2">
                  <XCircle className="w-5 h-5" />
                  <span>Differences</span>
                </h4>
                <div className="space-y-3">
                  {comparison.differences && comparison.differences.length > 0 ? (
                    comparison.differences.map((d, i) => (
                      <div key={i} className="bg-destructive/10 border border-destructive/20 rounded-md p-3">
                        <p className="text-destructive text-sm">{d}</p>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-8">
                      <XCircle className="w-12 h-12 mx-auto mb-2 text-muted-foreground/50" />
                      <p className="text-muted-foreground">No differences found</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="mb-8">
              <h4 className="text-xl font-medium text-foreground mb-4 flex items-center space-x-2">
                <AlertTriangle className="w-5 h-5 text-primary" />
                <span>Risk Assessment Comparison</span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-muted/50 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    {getRiskIcon(comparison.doc1Risk)}
                    <h5 className="font-medium text-foreground">Document 1 Risk</h5>
                  </div>
                  <p className={`text-2xl font-bold ${getRiskColor(comparison.doc1Risk)}`}>
                    {comparison.doc1Risk}/10
                  </p>
                </div>
                <div className="bg-muted/50 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    {getRiskIcon(comparison.doc2Risk)}
                    <h5 className="font-medium text-foreground">Document 2 Risk</h5>
                  </div>
                  <p className={`text-2xl font-bold ${getRiskColor(comparison.doc2Risk)}`}>
                    {comparison.doc2Risk}/10
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-xl font-medium text-foreground mb-4 flex items-center space-x-2">
                <ArrowRight className="w-5 h-5 text-primary" />
                <span>Recommendations</span>
              </h4>
              <div className="bg-muted/50 rounded-lg p-4">
                <p className="text-foreground">{comparison.recommendations}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ComparePage;
