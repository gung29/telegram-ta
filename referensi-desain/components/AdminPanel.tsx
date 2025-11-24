import React, { useState } from 'react';
import { UserOffender } from '../types';
import { Trash2, AlertOctagon, ShieldCheck, Zap } from 'lucide-react';

export const AdminPanel: React.FC = () => {
  const [testText, setTestText] = useState('');
  const [analysisResult, setAnalysisResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [admins] = useState([
      { id: '1', name: 'Admin 1', uid: '12345' },
      { id: '2', name: 'Admin 2', uid: '67890' },
  ]);

  const [mutedUsers] = useState<UserOffender[]>([
      { id: 'u1', username: 'UserA', userId: '111', warnings: 2, mutes: 1, lastActivity: '2h', avatarId: 101 },
  ]);

  const handleTestModel = async () => {
    if (!testText.trim()) return;
    setLoading(true);
    // Integrasi Gemini dimatikan; tampilkan placeholder
    setTimeout(() => {
      setAnalysisResult("Integrasi model dinonaktifkan. Gunakan endpoint model Anda sendiri.");
      setLoading(false);
    }, 400);
  };

  return (
    <div className="p-4 pb-24 space-y-8 animate-fade-in">
        
        {/* Header */}
        <div>
            <h2 className="text-2xl font-bold text-white mb-1">Admin Access</h2>
            <p className="text-sm text-slate-400">Manage permissions and model parameters</p>
        </div>

        {/* Admins List */}
        <div className="glass-panel rounded-2xl p-1">
            {admins.map(admin => (
                <div key={admin.id} className="flex items-center justify-between p-3 border-b border-slate-700/50 last:border-0">
                    <div>
                        <div className="font-bold text-white text-sm">{admin.name}</div>
                        <div className="text-xs text-slate-500">ID: {admin.uid}</div>
                    </div>
                    <button className="text-xs bg-slate-800 hover:bg-red-900/30 hover:text-red-400 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors">
                        Remove
                    </button>
                </div>
            ))}
            <div className="p-2">
                <button className="w-full py-2 bg-primary-600/20 text-primary-400 border border-primary-600/30 rounded-lg text-sm font-bold hover:bg-primary-600 hover:text-white transition-all">
                    Add New Admin
                </button>
            </div>
        </div>

        {/* AI Model Tester */}
        <div className="glass-panel rounded-3xl p-5 border border-neon-purple/30 relative overflow-hidden">
             <div className="absolute -top-10 -right-10 w-32 h-32 bg-neon-purple/20 blur-3xl rounded-full pointer-events-none"></div>
             
             <div className="flex items-center space-x-2 mb-4">
                 <Zap className="text-neon-purple" size={20} />
                 <h3 className="font-bold text-white">Moderation AI Tester</h3>
             </div>

             <textarea
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Enter text to test the moderation model..."
                className="w-full bg-slate-900/80 border border-slate-700 rounded-xl p-3 text-sm text-white focus:border-neon-purple focus:ring-1 focus:ring-neon-purple transition-all min-h-[100px] mb-4"
             />

             <button 
                onClick={handleTestModel}
                disabled={loading}
                className={`w-full py-3 rounded-xl font-bold text-white flex items-center justify-center transition-all ${loading ? 'bg-slate-700 cursor-not-allowed' : 'bg-slate-800 border border-slate-700'}`}
             >
                 {loading ? 'Memproses…' : 'Analyze (placeholder)'}
             </button>

             {analysisResult && (
                 <div className="mt-4 p-4 bg-slate-900 rounded-xl border border-slate-700 animate-fade-in text-slate-300 text-sm">
                     {analysisResult}
                 </div>
             )}
        </div>

        {/* Muted Users Management */}
        <div>
            <h3 className="text-white font-bold mb-3">Active Penalties</h3>
            
            {mutedUsers.map(user => (
                <div key={user.id} className="glass-panel p-4 rounded-2xl mb-3 border border-slate-800">
                    <div className="flex justify-between items-start mb-4">
                        <div>
                            <div className="font-bold text-white">{user.username}</div>
                            <div className="text-xs text-slate-500">ID: {user.userId}</div>
                        </div>
                        <div className="flex space-x-1">
                            <div className="px-2 py-1 bg-orange-500/20 text-orange-400 rounded text-[10px] font-bold border border-orange-500/20">
                                {user.warnings} Warns
                            </div>
                             <div className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-[10px] font-bold border border-purple-500/20">
                                {user.mutes} Mutes
                            </div>
                        </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-3">
                        <button className="py-2 bg-slate-800 hover:bg-slate-700 text-red-400 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center">
                            <AlertOctagon size={14} className="mr-1.5" /> Reset Warning
                        </button>
                        <button className="py-2 bg-slate-800 hover:bg-slate-700 text-purple-400 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center">
                            <ShieldCheck size={14} className="mr-1.5" /> Unmute
                        </button>
                    </div>
                </div>
            ))}
        </div>

    </div>
  );
};
