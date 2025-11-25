import React, { useEffect, useMemo, useState } from 'react';
import { AlertOctagon, ShieldCheck, Zap, UserPlus, RefreshCw } from 'lucide-react';
import {
  fetchAdmins,
  addAdmin,
  removeAdmin,
  fetchMembers,
  fetchUserActions,
  resetUserAction,
  deleteMemberStatus,
  runTest,
  AdminEntry,
  MemberModeration,
  UserActionSummary,
  HttpError,
  PermissionCheckResult,
  checkPermissions,
  unrestrictMember,
} from "../lib/api";

type Props = { chatId: number };

type PenaltyKind = "muted" | "banned" | "stuck";


export const AdminPanel: React.FC<Props> = ({ chatId }) => {
  const [testText, setTestText] = useState('');
  const [analysisResult, setAnalysisResult] = useState<string | null>(null);
  const [loadingTest, setLoadingTest] = useState(false);

  const [admins, setAdmins] = useState<AdminEntry[]>([]);
  const [muted, setMuted] = useState<MemberModeration[]>([]);
  const [banned, setBanned] = useState<MemberModeration[]>([]);
  const [userActions, setUserActions] = useState<UserActionSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [newAdminId, setNewAdminId] = useState<string>('');
  const [restrictedMap, setRestrictedMap] = useState<Map<number, PermissionCheckResult>>(new Map());

  const notify = (msg: string) => {
    if (typeof window !== "undefined") alert(msg);
  };

  const load = async () => {
    setLoading(true);
    try {
      const [a, m, b, ua] = await Promise.all([
        fetchAdmins(chatId),
        fetchMembers(chatId, "muted"),
        fetchMembers(chatId, "banned"),
        fetchUserActions(chatId),
      ]);
      setAdmins(a);
      setMuted(m);
      setBanned(b);
      setUserActions(ua);
      if (ua.length) {
        try {
          const res = await checkPermissions(chatId, ua.map((u) => u.user_id));
          setRestrictedMap(new Map(res.map((r) => [r.user_id, r])));
        } catch (err) {
          if (err instanceof HttpError) notify(err.message);
        }
      } else {
        setRestrictedMap(new Map());
      }
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [chatId]);

  const handleTestModel = async () => {
    if (!testText.trim()) return;
    setLoadingTest(true);
    try {
      const res = await runTest(chatId, testText.trim());
      setAnalysisResult(`Hate: ${(res.prob_hate * 100).toFixed(1)}% • Non-hate: ${(res.prob_nonhate * 100).toFixed(1)}% • Prediksi: ${res.label}`);
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
      else notify("Gagal menguji teks");
    } finally {
      setLoadingTest(false);
    }
  };

  const handleAddAdmin = async () => {
    const uid = Number(newAdminId);
    if (!uid) {
      notify("Masukkan user id valid");
      return;
    }
    setPendingAction("add-admin");
    try {
      await addAdmin(chatId, uid);
      setNewAdminId('');
      await load();
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
    } finally {
      setPendingAction(null);
    }
  };

  const handleRemoveAdmin = async (userId: number) => {
    setPendingAction(`remove-${userId}`);
    try {
      await removeAdmin(chatId, userId);
      await load();
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
    } finally {
      setPendingAction(null);
    }
  };

  const handleResetWarning = async (userId: number) => {
    setPendingAction(`reset-warning-${userId}`);
    try {
      await resetUserAction(chatId, userId, "warned");
      await load();
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
    } finally {
      setPendingAction(null);
    }
  };

  const handleResetMute = async (userId: number) => {
    setPendingAction(`reset-mute-${userId}`);
    try {
      await resetUserAction(chatId, userId, "muted");
      await load();
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
    } finally {
      setPendingAction(null);
    }
  };

const handleUnmute = async (userId: number) => {
  setPendingAction(`unmute-${userId}`);
  try {
    try {
      await deleteMemberStatus(chatId, userId, "muted"); // 👈 pakai DELETE
    } catch (err) {
      if (err instanceof HttpError && err.status === 404) {
        await unrestrictMember(chatId, userId); // fallback untuk user yang tersangkut tanpa record backend
      } else {
        throw err;
      }
    }
    await load();
  } catch (err) {
    if (err instanceof HttpError) notify(err.message);
  } finally {
    setPendingAction(null);
  }
};

const handleUnban = async (userId: number) => {
  setPendingAction(`unban-${userId}`);
  try {
    await deleteMemberStatus(chatId, userId, "banned"); // 👈 pakai DELETE
    await load();
  } catch (err) {
    if (err instanceof HttpError) notify(err.message);
  } finally {
    setPendingAction(null);
  }
};


  const penalties = useMemo(() => {
    const actionsMap = new Map<number, UserActionSummary>();
    userActions.forEach((u) => actionsMap.set(u.user_id, u));
    const mix = (list: MemberModeration[], kind: PenaltyKind) =>
      list.map((m) => ({
        ...m,
        kind,
        warns: actionsMap.get(m.user_id ?? 0)?.warnings_today ?? 0,
        mutes: actionsMap.get(m.user_id ?? 0)?.mutes_total ?? 0,
      }));
    const stuck = userActions
      .filter((u) => {
        const mutedEntry = muted.find((m) => m.user_id === u.user_id);
        const bannedEntry = banned.find((b) => b.user_id === u.user_id);
        const perm = restrictedMap.get(u.user_id);
        return !mutedEntry && !bannedEntry && perm && !perm.can_send_messages;
      })
      .map((u) => ({
        id: -u.user_id, // pseudo id to keep unique key
        chat_id: chatId,
        user_id: u.user_id,
        username: u.username,
        status: "muted" as const,
        reason: "Telegram masih membatasi kirim pesan",
        expires_at: undefined,
        created_at: new Date().toISOString(),
        kind: "stuck" as PenaltyKind,
        warns: u.warnings_today,
        mutes: u.mutes_total,
      }));
    return [...mix(muted, "muted"), ...mix(banned, "banned"), ...stuck];
  }, [muted, banned, userActions, restrictedMap, chatId]);

  const usersList = useMemo(() => {
    return userActions.map((u) => {
      const bannedEntry = banned.find((b) => b.user_id === u.user_id);
      const mutedEntry = muted.find((m) => m.user_id === u.user_id);
      return {
        ...u,
        banned: Boolean(bannedEntry),
        muted: Boolean(mutedEntry),
      };
    });
  }, [userActions, banned, muted]);

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
                <div key={admin.user_id} className="flex items-center justify-between p-3 border-b border-slate-700/50 last:border-0">
                    <div>
                        <div className="font-bold text-white text-sm">Admin {admin.user_id}</div>
                        <div className="text-xs text-slate-500">ID: {admin.user_id}</div>
                        {userActions.find((u) => u.user_id === admin.user_id)?.username && (
                          <div className="text-xs text-slate-400">
                            @{userActions.find((u) => u.user_id === admin.user_id)?.username}
                          </div>
                        )}
                    </div>
                    <button
                      disabled={pendingAction === `remove-${admin.user_id}`}
                      onClick={() => handleRemoveAdmin(admin.user_id)}
                      className="text-xs bg-slate-800 hover:bg-red-900/30 hover:text-red-400 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors disabled:opacity-60"
                    >
                        Remove
                    </button>
                </div>
            ))}
            <div className="p-3 space-y-2">
              <div className="flex items-center space-x-2">
                <input
                  type="number"
                  placeholder="Admin user id"
                  value={newAdminId}
                  onChange={(e) => setNewAdminId(e.target.value)}
                  className="flex-1 bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-primary-500 focus:outline-none"
                />
                <button
                  onClick={handleAddAdmin}
                  disabled={pendingAction === "add-admin"}
                  className="px-3 py-2 rounded-lg bg-primary-600/30 text-primary-200 border border-primary-500/40 hover:bg-primary-600 hover:text-white text-xs font-bold disabled:opacity-60"
                >
                  <UserPlus size={16} />
                </button>
              </div>
              <button
                onClick={load}
                className="w-full py-2 bg-slate-800 text-slate-300 rounded-lg text-xs font-semibold border border-slate-700 hover:border-primary-400 flex items-center justify-center space-x-2"
              >
                <RefreshCw size={14} />
                <span>Reload Admins</span>
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
                disabled={loadingTest}
                className={`w-full py-3 rounded-xl font-bold text-white flex items-center justify-center transition-all ${loadingTest ? 'bg-slate-700 cursor-not-allowed' : 'bg-slate-800 border border-slate-700'}`}
             >
                 {loadingTest ? 'Memproses…' : 'Analyze (placeholder)'}
             </button>

             {analysisResult && (
                 <div className="mt-4 p-4 bg-slate-900 rounded-xl border border-slate-700 animate-fade-in text-slate-300 text-sm">
                     {analysisResult}
                 </div>
             )}
        </div>

        {/* Muted/Banned Users Management */}
        <div>
            <h3 className="text-white font-bold mb-3">Active Penalties</h3>
            {loading && <p className="text-slate-400 text-sm">Memuat...</p>}
            {!loading && penalties.length === 0 && <p className="text-slate-500 text-sm">Tidak ada penalti aktif.</p>}
            {penalties.map(user => (
                <div key={`${user.kind}-${user.id}`} className="glass-panel p-4 rounded-2xl mb-3 border border-slate-800">
                    <div className="flex justify-between items-start mb-4">
                        <div>
                            <div className="font-bold text-white">{user.username ?? `User ${user.user_id ?? '-'}`}</div>
                            <div className="text-xs text-slate-500">ID: {user.user_id}</div>
                        </div>
                        <div className="flex space-x-1">
                            <div className="px-2 py-1 bg-orange-500/20 text-orange-400 rounded text-[10px] font-bold border border-orange-500/20">
                                {user.warns} Warns
                            </div>
                             <div className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-[10px] font-bold border border-purple-500/20">
                                {user.mutes} Mutes
                            </div>
                        </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        <button
                          disabled={pendingAction === `reset-warning-${user.user_id}`}
                          onClick={() => handleResetWarning(user.user_id!)}
                          className="py-2 bg-slate-800 hover:bg-slate-700 text-red-400 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center disabled:opacity-60"
                        >
                            <AlertOctagon size={14} className="mr-1.5" /> Reset Warning
                        </button>
                        <button
                          disabled={pendingAction === `reset-mute-${user.user_id}`}
                          onClick={() => handleResetMute(user.user_id!)}
                          className="py-2 bg-slate-800 hover:bg-slate-700 text-purple-300 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center disabled:opacity-60"
                        >
                            <RefreshCw size={14} className="mr-1.5" /> Reset Mute
                        </button>
                        <button
                          disabled={pendingAction === `${user.kind === 'banned' ? 'unban' : 'unmute'}-${user.user_id}`}
                          onClick={() => (user.kind === 'banned' ? handleUnban(user.user_id!) : handleUnmute(user.user_id!))}
                          className="py-2 bg-slate-800 hover:bg-slate-700 text-purple-400 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center disabled:opacity-60"
                        >
                            <ShieldCheck size={14} className="mr-1.5" /> {user.kind === 'banned' ? 'Unban' : 'Unmute'}
                        </button>
                    </div>
                </div>
            ))}
        </div>

        {/* User Actions Overview */}
        <div className="glass-panel rounded-2xl p-4 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-white font-bold">User Actions</h3>
            <button
              onClick={load}
              className="flex items-center space-x-1 text-xs text-slate-300 px-2 py-1 rounded-lg border border-slate-700 hover:border-primary-400"
            >
              <RefreshCw size={12} /> <span>Reload</span>
            </button>
          </div>
          {loading && <p className="text-slate-400 text-sm">Memuat...</p>}
          {!loading && usersList.length === 0 && <p className="text-slate-500 text-sm">Belum ada data pengguna.</p>}
          <div className="space-y-2">
            {usersList.map((u) => (
              <div key={u.user_id} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <div className="text-white font-bold text-sm">
                      {u.username ? `@${u.username}` : `User ${u.user_id}`}
                    </div>
                    <div className="text-[11px] text-slate-500">ID: {u.user_id}</div>
                  </div>
                  <div className="flex space-x-1 text-[11px]">
                    <span className="px-2 py-1 rounded bg-orange-500/15 text-orange-300 border border-orange-500/30">
                      {u.warnings_today} Warn
                    </span>
                    <span className="px-2 py-1 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30">
                      {u.mutes_total} Mute
                    </span>
                    {u.banned && (
                      <span className="px-2 py-1 rounded bg-red-500/15 text-red-300 border border-red-500/30">
                        Banned
                      </span>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    disabled={pendingAction === `reset-warning-${u.user_id}`}
                    onClick={() => handleResetWarning(u.user_id)}
                    className="py-2 bg-slate-800 hover:bg-slate-700 text-red-400 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center disabled:opacity-60"
                  >
                    <AlertOctagon size={14} className="mr-1.5" /> Reset Warning
                  </button>
                  <button
                    disabled={pendingAction === `reset-mute-${u.user_id}`}
                    onClick={() => handleResetMute(u.user_id)}
                    className="py-2 bg-slate-800 hover:bg-slate-700 text-purple-300 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center disabled:opacity-60"
                  >
                    <RefreshCw size={14} className="mr-1.5" /> Reset Mute
                  </button>
                  {u.banned ? (
                    <button
                      disabled={pendingAction === `unban-${u.user_id}`}
                      onClick={() => handleUnban(u.user_id)}
                      className="py-2 bg-slate-800 hover:bg-slate-700 text-purple-400 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center disabled:opacity-60"
                    >
                      <ShieldCheck size={14} className="mr-1.5" /> Unban
                    </button>
                  ) : (
                    <button
                      disabled={pendingAction === `unmute-${u.user_id}`}
                      onClick={() => handleUnmute(u.user_id)}
                      className="py-2 bg-slate-800 hover:bg-slate-700 text-purple-400 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center disabled:opacity-60"
                    >
                      <ShieldCheck size={14} className="mr-1.5" /> Unmute
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

    </div>
  );
};
