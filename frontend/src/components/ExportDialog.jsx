import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Button } from './ui/button';
import { Checkbox } from './ui/checkbox';

const ExportDialog = ({ isOpen, onClose, document, onExport }) => {
  const [selectedSections, setSelectedSections] = useState({
    summary: true,
    key_clauses: true,
    risk_assessment: true,
    plain_english: true
  });

  const handleSectionChange = (section) => {
    setSelectedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleExport = () => {
    const sections = Object.keys(selectedSections).filter(key => selectedSections[key]);
    onExport(document, sections);
    onClose();
  };

  const sections = [
    { key: 'summary', label: 'Document Summary' },
    { key: 'key_clauses', label: 'Key Clauses' },
    { key: 'risk_assessment', label: 'Risk Assessment' },
    { key: 'plain_english', label: 'Plain English Explanation' }
  ];

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Export PDF Report</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <p className="text-sm text-gray-600">
            Select the sections you want to include in the PDF report for "{document?.filename}":
          </p>
          <div className="space-y-3">
            {sections.map(section => (
              <div key={section.key} className="flex items-center space-x-2">
                <Checkbox
                  id={section.key}
                  checked={selectedSections[section.key]}
                  onCheckedChange={() => handleSectionChange(section.key)}
                />
                <label
                  htmlFor={section.key}
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                >
                  {section.label}
                </label>
              </div>
            ))}
          </div>
        </div>
        <div className="flex justify-end space-x-2">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleExport} disabled={!Object.values(selectedSections).some(Boolean)}>
            Export PDF
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ExportDialog;
