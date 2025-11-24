import React from 'react';
import { View } from '../types';
import { LayoutDashboard, BarChart3, FileText, CheckCircle, ShieldAlert } from 'lucide-react';

interface NavigationProps {
  currentView: View;
  onViewChange: (view: View) => void;
}

export const Navigation: React.FC<NavigationProps> = ({ currentView, onViewChange }) => {
  const navItems = [
    { view: View.DASHBOARD, icon: LayoutDashboard, label: 'Dash' },
    { view: View.STATS, icon: BarChart3, label: 'Stats' },
    { view: View.VERIFY, icon: CheckCircle, label: 'Verify' },
    { view: View.LOGS, icon: FileText, label: 'Logs' },
    { view: View.ADMIN, icon: ShieldAlert, label: 'Admin' },
  ];

  return (
    <div className="fixed bottom-0 left-0 w-full glass-panel border-t border-slate-700 pb-safe pt-2 px-6 z-50">
      <div className="flex justify-between items-center max-w-lg mx-auto h-16">
        {navItems.map((item) => {
          const isActive = currentView === item.view;
          return (
            <button
              key={item.view}
              onClick={() => onViewChange(item.view)}
              className={`flex flex-col items-center justify-center w-12 transition-all duration-300 ${
                isActive ? 'text-neon-blue -translate-y-2' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <div className={`p-2 rounded-full transition-all duration-300 ${isActive ? 'bg-primary-500/20 shadow-[0_0_15px_rgba(99,102,241,0.5)]' : ''}`}>
                <item.icon size={isActive ? 24 : 20} strokeWidth={isActive ? 2.5 : 2} />
              </div>
              <span className={`text-[10px] mt-1 font-medium ${isActive ? 'opacity-100' : 'opacity-0'}`}>
                {item.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};