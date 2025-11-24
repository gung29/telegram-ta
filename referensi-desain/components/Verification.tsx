import React, { useState } from 'react';
import { VerificationItem } from '../types';
import { Check, X, Filter } from 'lucide-react';

const initialItems: VerificationItem[] = [
    { id: '1', user: 'Ahmad S.', timeAgo: 'Just now', content: 'Ini contoh pesan yang mengandung ujaran kebencian, harus ditindak tegas.', score: 92, avatarId: 88 },
    { id: '2', user: 'Budi P.', timeAgo: '2h ago', content: 'Selamat pagi, semoga harimu menyenangkan.', score: 15, avatarId: 44 },
    { id: '3', user: 'Citra D.', timeAgo: 'Yesterday', content: 'Jangan percaya orang-orang itu, mereka pembohong besar dan penipu.', score: 88, avatarId: 22 },
];

export const Verification: React.FC = () => {
  const [items, setItems] = useState<VerificationItem[]>(initialItems);

  const handleDecision = (id: string, isHate: boolean) => {
    // Simulate removing card
    const element = document.getElementById(`card-${id}`);
    if (element) {
        element.style.transform = isHate ? 'translateX(100vw)' : 'translateX(-100vw)';
        element.style.opacity = '0';
    }
    setTimeout(() => {
        setItems(prev => prev.filter(i => i.id !== id));
    }, 300);
  };

  return (
    <div className="p-4 pb-24 h-full flex flex-col animate-fade-in relative">
        <div className="flex justify-between items-center mb-6">
            <div>
                <h2 className="text-2xl font-bold text-white">Manual Verification</h2>
                <p className="text-sm text-slate-400">Review flagged messages</p>
            </div>
            <button className="text-neon-blue hover:text-white">
                <Filter size={24} />
            </button>
        </div>

        <div className="flex-1 relative">
            {items.length === 0 ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                    <Check size={48} className="mb-4 text-green-500 opacity-50" />
                    <p>All caught up! No pending reviews.</p>
                </div>
            ) : (
                <div className="space-y-4">
                    {items.map((item) => (
                        <div 
                            key={item.id} 
                            id={`card-${item.id}`}
                            className="glass-panel p-5 rounded-3xl border border-slate-700/50 shadow-lg transition-all duration-300 transform"
                        >
                            <div className="flex items-center space-x-3 mb-4">
                                <div className="w-10 h-10 rounded-full bg-slate-700 overflow-hidden">
                                    <img src={`https://picsum.photos/40/40?random=${item.avatarId}`} alt="Avatar" />
                                </div>
                                <div>
                                    <h4 className="text-white font-bold">{item.user}</h4>
                                    <span className="text-xs text-slate-400">{item.timeAgo}</span>
                                </div>
                                <div className="ml-auto text-right">
                                    <div className="text-[10px] text-slate-500 uppercase tracking-wide">Confidence</div>
                                    <div className={`text-xl font-mono font-bold ${item.score > 80 ? 'text-red-500' : 'text-orange-500'}`}>
                                        {item.score}%
                                    </div>
                                </div>
                            </div>

                            <div className="bg-slate-900/50 p-4 rounded-xl mb-6 border border-slate-800">
                                <p className="text-slate-200 text-sm leading-relaxed">
                                    "{item.content}"
                                </p>
                            </div>

                            <div className="flex space-x-3">
                                <button 
                                    onClick={() => handleDecision(item.id, false)}
                                    className="flex-1 py-3 rounded-xl bg-green-500/10 text-green-500 border border-green-500/30 hover:bg-green-500 hover:text-white transition-all font-bold flex items-center justify-center"
                                >
                                    <Check size={18} className="mr-2" /> Mark Safe
                                </button>
                                <button 
                                    onClick={() => handleDecision(item.id, true)}
                                    className="flex-1 py-3 rounded-xl bg-red-500/10 text-red-500 border border-red-500/30 hover:bg-red-500 hover:text-white transition-all font-bold flex items-center justify-center"
                                >
                                    <X size={18} className="mr-2" /> Hate Speech
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    </div>
  );
};