import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-muted/30 border-t border-border mt-6">
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex flex-col md:flex-row justify-between items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-primary rounded-md flex items-center justify-center">
              <span className="text-primary-foreground font-semibold text-xs">C</span>
            </div>
            <span className="font-medium text-foreground text-sm">CovenantAI</span>
          </div>
          <p className="text-xs text-muted-foreground mb-0">
            © 2025 CovenantAI. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
