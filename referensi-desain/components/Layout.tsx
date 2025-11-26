import React from 'react';
import { View } from '../types';
import { Navigation } from './Navigation';

interface LayoutProps {
  children: React.ReactNode;
  currentView: View;
  onViewChange: (view: View) => void;
  hideNav?: boolean;
  avatarUrl?: string;
}

export const Layout: React.FC<LayoutProps> = ({ children, currentView, onViewChange, hideNav, avatarUrl }) => {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-neon-blue/30 selection:text-white">
      {/* Background Ambient Glow */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
          <div className="absolute -top-[10%] -left-[10%] w-[50%] h-[50%] bg-primary-900/20 rounded-full blur-[100px] animate-pulse-slow"></div>
          <div className="absolute top-[20%] right-[0%] w-[40%] h-[40%] bg-blue-900/10 rounded-full blur-[100px]"></div>
          <div className="absolute bottom-[0%] left-[20%] w-[60%] h-[40%] bg-purple-900/10 rounded-full blur-[100px]"></div>
      </div>

      {/* Main Content Area */}
      <div className="relative z-10 max-w-lg mx-auto min-h-screen bg-slate-950/50 shadow-2xl border-x border-slate-900">
        
        {/* Top Bar */}
        <header className="sticky top-0 z-40 glass-panel border-b border-slate-800/50 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center space-x-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-primary-600 to-neon-blue flex items-center justify-center shadow-lg shadow-primary-500/20">
                   <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                   </svg>
                </div>
                <h1 className="font-bold text-lg tracking-tight text-white">HateSpeech<span className="text-primary-400">Mod</span></h1>
            </div>
            <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center overflow-hidden">
                {avatarUrl ? (
                  <img src={avatarUrl} alt="Admin" className="w-full h-full rounded-full opacity-90 hover:opacity-100 transition-opacity object-cover" />
                ) : (
                  <div className="text-[10px] uppercase tracking-wide text-slate-300">You</div>
                )}
            </div>
        </header>

        {children}
        
      </div>

      {!hideNav && <Navigation currentView={currentView} onViewChange={onViewChange} />}
    </div>
  );
};
