import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import ParticleBackground from './ParticleBackground';
import Footer from './Footer';

const LandingPage = () => {
  const navigate = useNavigate();
  const [floatingIcons, setFloatingIcons] = useState([]);

  useEffect(() => {
    // Generate floating legal symbols
    const icons = ['§', '¶', '∑', '∆', '∫', 'Ω'];
    const newIcons = Array.from({ length: 12 }, (_, i) => ({
      id: i,
      icon: icons[i % icons.length],
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 2,
      duration: 3 + Math.random() * 4,
    }));
    setFloatingIcons(newIcons);
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring",
        stiffness: 100,
        damping: 12,
      },
    },
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 relative overflow-hidden">
      {/* Subtle background pattern */}
      <div className="absolute inset-0 opacity-[0.02] dark:opacity-[0.05]">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, rgba(0,0,0,.15) 1px, transparent 0)`,
          backgroundSize: '20px 20px'
        }} />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto p-6">
        <motion.main
          className="flex flex-col items-center justify-center min-h-screen text-center"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Hero Section */}
          <motion.div variants={itemVariants} className="mb-12">
            <motion.h1
              className="text-4xl md:text-6xl font-bold text-slate-900 dark:text-slate-100 mb-6 leading-tight max-w-4xl tracking-tight"
              style={{
                fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif'
              }}
            >
              De-risk Your Legal World.
            </motion.h1>
            <motion.div
              className="w-16 h-0.5 bg-gradient-to-r from-slate-400 to-slate-300 dark:from-slate-600 dark:to-slate-500 mx-auto rounded-full"
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ duration: 1, delay: 0.5 }}
            />
          </motion.div>

          <motion.p
            variants={itemVariants}
            className="text-lg md:text-xl text-slate-600 dark:text-slate-300 mb-12 max-w-3xl mx-auto leading-relaxed font-light"
            style={{
              fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
            }}
          >
            CovenantAI delivers instant clarity for your legal documents — concise summaries, risk scoring, and negotiation guidance, all in plain English.
          </motion.p>

          {/* CTA Button */}
          <motion.button
            variants={itemVariants}
            className="group relative bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white px-8 py-4 rounded-2xl shadow-lg text-lg font-medium hover:shadow-xl transition-all duration-300 hover:scale-[1.02] overflow-hidden"
            onClick={() => navigate('/analytics')}
            whileHover={{
              scale: 1.02,
              boxShadow: "0 20px 40px rgba(0,0,0,0.1)",
            }}
            whileTap={{ scale: 0.98 }}
            style={{
              fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif'
            }}
          >
            <span className="relative z-10">Analyze Document Now</span>
          </motion.button>


        </motion.main>
      </div>
      <Footer />
    </div>
  );
};

export default LandingPage;
