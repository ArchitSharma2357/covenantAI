import React, { useEffect, useRef } from 'react';

// Lightweight Matrix rain effect - optional, performant for small particle counts
const MatrixBackground = ({ enabled }) => {
  const canvasRef = useRef(null);
  const [visible, setVisible] = React.useState(enabled || false);

  useEffect(() => {
    if (!visible) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;
    const cols = Math.floor(width / 14);
    const drops = Array(cols).fill(0).map(() => Math.random() * height);
    let raf = null;

    const draw = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#39ff14';
      ctx.font = '14px monospace';
      for (let i = 0; i < drops.length; i++) {
        const text = String.fromCharCode(48 + Math.floor(Math.random() * 42));
        ctx.fillText(text, i * 14, drops[i]);
        drops[i] = drops[i] > height && Math.random() > 0.975 ? 0 : drops[i] + 14;
      }
      raf = requestAnimationFrame(draw);
    };

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', handleResize);
    draw();

    return () => {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener('resize', handleResize);
    };
  }, [visible]);

  useEffect(() => {
    const handler = (e) => setVisible(!!e.detail);
    window.addEventListener('matrixModeChanged', handler);
    return () => window.removeEventListener('matrixModeChanged', handler);
  }, []);

  return (
    <canvas ref={canvasRef} className={`absolute inset-0 w-full h-full pointer-events-none`} />
  );
};

export default MatrixBackground;
