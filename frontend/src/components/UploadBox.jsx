import React, { useState, useRef } from 'react';
import { Upload, FileText, X } from 'lucide-react';

const UploadBox = ({ onFileSelect, onTextInput }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [textInput, setTextInput] = useState('');
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      onFileSelect(file);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      onFileSelect(file);
    }
  };

  const handleTextChange = (e) => {
    const text = e.target.value;
    setTextInput(text);
    onTextInput(text);
  };

  const openFileDialog = () => {
    fileInputRef.current.click();
  };

  const removeFile = () => {
    setSelectedFile(null);
    onFileSelect(null);
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      <div className="bg-card border border-border rounded-lg p-8">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-semibold text-foreground mb-2">Upload Your Document</h2>
          <p className="text-muted-foreground">Drag and drop your file here or click to browse</p>
        </div>

        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
            dragActive ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={openFileDialog}
        >
          <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground mb-2">Choose a file or drag it here</p>
          <p className="text-sm text-muted-foreground">Supported formats: PDF, DOC, DOCX, TXT</p>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".pdf,.doc,.docx,.txt"
            onChange={handleFileSelect}
          />
        </div>

        {selectedFile && (
          <div className="mt-4 p-4 bg-muted rounded-lg flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileText className="w-5 h-5 text-muted-foreground" />
              <div>
                <p className="font-medium text-foreground">{selectedFile.name}</p>
                <p className="text-sm text-muted-foreground">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            </div>
            <button
              onClick={removeFile}
              className="p-1 hover:bg-destructive/10 rounded transition-colors"
            >
              <X className="w-4 h-4 text-muted-foreground" />
            </button>
          </div>
        )}
      </div>

      <div className="bg-card border border-border rounded-lg p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">Or Paste Your Text</h3>
        <textarea
          value={textInput}
          onChange={handleTextChange}
          placeholder="Paste your legal document text here..."
          className="w-full min-h-[8rem] max-h-[40vh] p-3 border border-border rounded-lg resize-y overflow-auto focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent bg-background text-foreground"
        />
      </div>
    </div>
  );
};

export default UploadBox;
