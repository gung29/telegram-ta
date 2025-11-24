import React, { useState } from 'react';
import { Search, Filter, MoreHorizontal, Download } from 'lucide-react';
import { ModerationLog } from '../types';

const mockLogs: ModerationLog[] = [
    { id: '1', timestamp: '10:45 AM', user: 'User123', userId: 'u1', action: 'Muted', hateScore: 85, content: 'This group is full of idiots and...', status: 'Auto-Muted' },
    { id: '2', timestamp: '10:42 AM', user: 'BadActor99', userId: 'u2', action: 'Banned', hateScore: 92, content: 'Go back to where you came from...', status: 'Manual Review' },
    { id: '3', timestamp: '10:30 AM', user: 'Anon55', userId: 'u3', action: 'Warned', hateScore: 65, content: 'Stop posting this garbage or else...', status: 'Active' },
    { id: '4', timestamp: '09:15 AM', user: 'TrollMaster', userId: 'u4', action: 'Deleted', hateScore: 78, content: 'Scam link: http://fake-crypto...', status: 'Auto-Muted' },
    { id: '5', timestamp: '08:50 AM', user: 'Newbie01', userId: 'u5', action: 'Flagged', hateScore: 45, content: 'Is this even real?', status: 'Manual Review' },
    { id: '6', timestamp: 'Yesterday', user: 'SpammerX', userId: 'u6', action: 'Banned', hateScore: 99, content: 'BUY BTC NOW!!! CHEAP!!', status: 'Verified' },
    { id: '7', timestamp: 'Yesterday', user: 'HateBot', userId: 'u7', action: 'Banned', hateScore: 88, content: '[Profanity Redacted]', status: 'Verified' },
];

export const Logs: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');

  const getActionColor = (action: string) => {
      switch(action) {
          case 'Banned': return 'text-red-500 bg-red-500/10 border-red-500/20';
          case 'Muted': return 'text-purple-500 bg-purple-500/10 border-purple-500/20';
          case 'Warned': return 'text-orange-500 bg-orange-500/10 border-orange-500/20';
          default: return 'text-slate-400 bg-slate-800 border-slate-700';
      }
  };

  const getScoreColor = (score: number) => {
      if (score >= 90) return 'text-red-500';
      if (score >= 70) return 'text-orange-500';
      return 'text-green-500';
  };

  return (
    <div className="p-4 h-full flex flex-col pb-24 animate-fade-in">
        <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-white">History & Logs</h2>
            <button className="p-2 rounded-full bg-slate-800 text-slate-400 hover:text-white">
                <Download size={20} />
            </button>
        </div>

        {/* Search & Filter */}
        <div className="space-y-3 mb-6">
            <div className="relative">
                <Search className="absolute left-3 top-3 text-slate-500" size={18} />
                <input 
                    type="text" 
                    placeholder="Search logs, users, content..." 
                    className="w-full bg-slate-900 border border-slate-700 rounded-xl py-2.5 pl-10 pr-4 text-sm text-white focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                />
            </div>
            <div className="flex overflow-x-auto space-x-2 pb-2 scrollbar-hide">
                {['All', 'Verified', 'Flagged', 'Muted', 'Banned'].map(filter => (
                    <button key={filter} className="px-3 py-1.5 rounded-full border border-slate-700 text-xs text-slate-300 hover:bg-slate-800 whitespace-nowrap">
                        {filter}
                    </button>
                ))}
            </div>
        </div>

        {/* Table Header */}
        <div className="grid grid-cols-12 gap-2 text-xs font-bold text-slate-500 px-2 mb-2 uppercase tracking-wider">
            <div className="col-span-3">Time/User</div>
            <div className="col-span-2 text-center">Action</div>
            <div className="col-span-2 text-center">Score</div>
            <div className="col-span-5">Content</div>
        </div>

        {/* List */}
        <div className="space-y-2 overflow-y-auto flex-1 pr-1">
            {mockLogs.filter(l => l.user.toLowerCase().includes(searchTerm.toLowerCase()) || l.content.toLowerCase().includes(searchTerm.toLowerCase())).map((log) => (
                <div key={log.id} className="glass-panel p-3 rounded-xl grid grid-cols-12 gap-2 items-center hover:bg-slate-800/50 transition-colors">
                    <div className="col-span-3">
                        <div className="text-xs text-slate-400 font-mono">{log.timestamp}</div>
                        <div className="text-sm font-bold text-white truncate">{log.user}</div>
                    </div>
                    <div className="col-span-2 flex justify-center">
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${getActionColor(log.action)}`}>
                            {log.action}
                        </span>
                    </div>
                    <div className="col-span-2 text-center">
                        <span className={`font-mono font-bold text-sm ${getScoreColor(log.hateScore)}`}>
                            {log.hateScore}%
                        </span>
                    </div>
                    <div className="col-span-4">
                        <p className="text-xs text-slate-300 truncate opacity-80">{log.content}</p>
                    </div>
                    <div className="col-span-1 flex justify-end">
                        <MoreHorizontal size={16} className="text-slate-500" />
                    </div>
                </div>
            ))}
        </div>
    </div>
  );
};