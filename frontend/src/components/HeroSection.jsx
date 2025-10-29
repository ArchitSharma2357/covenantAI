import React from 'react';
import { motion } from 'framer-motion';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

const data = [
  { name: 'Jan', 'Your Savings': 2000, 'Avg. Lawyer Fees': 2400 },
  { name: 'Feb', 'Your Savings': 1800, 'Avg. Lawyer Fees': 2210 },
  { name: 'Mar', 'Your Savings': 2200, 'Avg. Lawyer Fees': 2290 },
  { name: 'Apr', 'Your Savings': 2780, 'Avg. Lawyer Fees': 3000 },
  { name: 'May', 'Your Savings': 1890, 'Avg. Lawyer Fees': 2181 },
  { name: 'Jun', 'Your Savings': 2390, 'Avg. Lawyer Fees': 2500 },
];

const barData = [
    { name: 'Contract Review', 'Your Cost': 400, 'Traditional Cost': 1500 },
    { name: 'NDA Analysis', 'Your Cost': 200, 'Traditional Cost': 800 },
    { name: 'Lease Agreement', 'Your Cost': 600, 'Traditional Cost': 2500 },
];

const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: i => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.2,
      duration: 0.5,
      ease: 'easeOut'
    }
  })
};

const HeroSection = () => {
  return (
    <section className="py-16 px-6">
      <div className="max-w-6xl mx-auto text-center">
        <motion.h1 
          className="text-4xl md:text-6xl font-bold text-foreground mb-6"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Legal Clarity, Unmatched Savings.
        </motion.h1>
        <motion.p 
          className="text-xl text-muted-foreground mb-12 max-w-3xl mx-auto"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          CovenantAI transforms dense legal documents into clear, actionable insights.
          Understand contracts, minimize risks, and save on legal fees—all with the power of AI.
        </motion.p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <motion.div 
              className="bg-card/50 backdrop-blur-lg p-6 rounded-2xl border border-border/20 shadow-lg"
              custom={0}
              variants={cardVariants}
              initial="hidden"
              animate="visible"
            >
                <h3 className="font-semibold text-foreground mb-4 text-lg">Monthly Savings Comparison</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorSavings" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.8}/>
                                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorFees" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                                <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" />
                        <YAxis stroke="hsl(var(--muted-foreground))" />
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', borderRadius: '10px' }} />
                        <Area type="monotone" dataKey="Avg. Lawyer Fees" stroke="#8884d8" fillOpacity={1} fill="url(#colorFees)" />
                        <Area type="monotone" dataKey="Your Savings" stroke="hsl(var(--primary))" fillOpacity={1} fill="url(#colorSavings)" />
                    </AreaChart>
                </ResponsiveContainer>
            </motion.div>
            <motion.div 
              className="bg-card/50 backdrop-blur-lg p-6 rounded-2xl border border-border/20 shadow-lg"
              custom={1}
              variants={cardVariants}
              initial="hidden"
              animate="visible"
            >
                <h3 className="font-semibold text-foreground mb-4 text-lg">Cost Comparison</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={barData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" />
                        <YAxis stroke="hsl(var(--muted-foreground))" />
                        <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', borderRadius: '10px' }} />
                        <Bar dataKey="Traditional Cost" fill="#8884d8" />
                        <Bar dataKey="Your Cost" fill="hsl(var(--primary))" />
                    </BarChart>
                </ResponsiveContainer>
            </motion.div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
