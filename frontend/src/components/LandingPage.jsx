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
        type: 'spring',
        stiffness: 100,
        damping: 12,
      },
    },
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 relative overflow-hidden">
      {/* Subtle background pattern */}
      <div className="absolute inset-0 opacity-[0.02] dark:opacity-[0.05]">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, rgba(0,0,0,.15) 1px, transparent 0)`,
            backgroundSize: '20px 20px',
          }}
        />
      </div>

      {/* Floating Legal Symbols */}
      {floatingIcons.map((icon) => (
        <motion.span
          key={icon.id}
          className="absolute text-slate-400 dark:text-slate-600 select-none text-2xl md:text-3xl"
          style={{
            left: `${icon.x}%`,
            top: `${icon.y}%`,
            fontFamily: 'serif',
            opacity: 0.15,
          }}
          animate={{
            y: ['0%', '-15%', '0%'],
            opacity: [0.15, 0.3, 0.15],
          }}
          transition={{
            duration: icon.duration,
            delay: icon.delay,
            repeat: Infinity,
          }}
        >
          {icon.icon}
        </motion.span>
      ))}

      <div className="relative z-10 max-w-5xl mx-auto p-6">
        <motion.main
          className="flex flex-col items-center justify-center min-h-screen text-center"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Hero Section */}
          <motion.div variants={itemVariants} className="mb-16 text-center">
            {/* Headline */}
            <motion.h1
              className="text-4xl md:text-6xl font-extrabold mb-6 leading-tight max-w-4xl mx-auto tracking-tight"
              style={{
                fontFamily:
                  '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif',
                background:
                  'linear-gradient(90deg, #0f172a, #334155, #64748b)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              De-risk Your Legal World.
            </motion.h1>

            {/* Accent Divider */}
            <motion.div
              className="w-20 h-0.5 bg-gradient-to-r from-slate-400 via-slate-500 to-slate-400 dark:from-slate-700 dark:via-slate-400 dark:to-slate-700 mx-auto rounded-full shadow-sm"
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ duration: 1, delay: 0.4 }}
            />

            {/* Supporting Paragraph */}
            <motion.p
              variants={itemVariants}
              className="text-lg md:text-xl text-slate-600 dark:text-slate-300 mt-10 mb-12 max-w-3xl mx-auto leading-relaxed font-light"
              style={{
                fontFamily:
                  '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif',
              }}
            >
              <span className="font-semibold text-slate-900 dark:text-white">
                CovenantAI
              </span>{' '}
              helps you make sense of legal documents — faster, clearer, and
              smarter. It delivers
              <span className="text-emerald-600 dark:text-emerald-400 font-medium">
                {' '}
                concise summaries
              </span>
              ,
              <span className="text-blue-600 dark:text-blue-400 font-medium">
                {' '}
                risk insights
              </span>
              , and
              <span className="text-purple-600 dark:text-purple-400 font-medium">
                {' '}
                negotiation guidance
              </span>{' '}
              — all written in{' '}
              <span className="font-semibold text-slate-900 dark:text-white">
                plain English
              </span>{' '}
              or your{' '}
              <span className="text-rose-600 dark:text-rose-400 font-medium">
                preferred language
              </span>
              .
              <br className="hidden md:block" />
              Type in{' '}
              <span className="text-orange-600 dark:text-orange-400 font-medium">
                Hindi
              </span>
              ,
              <span className="text-green-600 dark:text-green-400 font-medium">
                {' '}
                Spanish
              </span>
              , or any other language — and CovenantAI will reply naturally in
              the same.
            </motion.p>

            {/* CTA Button */}
            <motion.button
              variants={itemVariants}
              className="group relative overflow-hidden px-9 py-4 rounded-2xl text-lg font-semibold shadow-lg text-white dark:text-slate-900 transition-all duration-300"
              onClick={() => navigate('/analytics')}
              whileHover={{
                scale: 1.04,
                boxShadow: '0 20px 40px rgba(0,0,0,0.15)',
              }}
              whileTap={{ scale: 0.97 }}
              style={{
                fontFamily:
                  '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif',
              }}
            >
              <span className="absolute inset-0 bg-gradient-to-r from-slate-900 via-slate-700 to-slate-900 dark:from-slate-100 dark:via-slate-200 dark:to-slate-100 transition-all duration-500 group-hover:from-slate-800 group-hover:via-slate-600 group-hover:to-slate-800 dark:group-hover:from-slate-200 dark:group-hover:via-slate-300 dark:group-hover:to-slate-200"></span>
              <span className="absolute inset-0 rounded-2xl ring-1 ring-white/10 group-hover:ring-2 group-hover:ring-white/20 transition-all duration-300"></span>
              <span className="relative z-10 tracking-wide">
                Analyze Document Now
              </span>
            </motion.button>
          </motion.div>
        </motion.main>
      </div>
      <Footer />
    </div>
  );
};

export default LandingPage;
