import React, { useEffect, useMemo, useState } from 'react';
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import { Check, X, Filter, Shield } from 'lucide-react';
import { fetchEvents, verifyEvent, EventEntry, HttpError } from "../lib/api";

dayjs.extend(relativeTime);

type Props = { chatId: number };

export const Verification: React.FC<Props> = ({ chatId }) => {
  const [items, setItems] = useState<EventEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [verifying, setVerifying] = useState<number | null>(null);
  const [filter, setFilter] = useState<"all" | "pending" | "verified" | "hate" | "non-hate">("pending");
  const [showFilters, setShowFilters] = useState(false);

  const notify = (msg: string) => {
    if (typeof window !== "undefined") alert(msg);
  };

  const load = async () => {
    setLoading(true);
    try {
      const data = await fetchEvents(chatId, 30, 0);
      const sorted = [...data].sort((a, b) => Number(a.manual_verified) - Number(b.manual_verified));
      setItems(sorted);
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [chatId]);

  const formatTime = (iso?: string) => {
    if (!iso) return "";
    const d = dayjs(iso);
    if (!d.isValid()) return "";
    const now = dayjs();
    return now.diff(d, "hour") < 24 ? d.from(now) : d.format("DD MMM");
  };

  const handleDecision = async (id: number, label: "hate" | "non-hate") => {
    setVerifying(id);
    try {
      const updated = await verifyEvent(chatId, id, label);
      setItems((prev) => prev.map((i) => (i.id === updated.id ? { ...i, ...updated } : i)));
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
    } finally {
      setVerifying(null);
    }
  };

  const pendingCount = useMemo(() => items.filter((i) => !i.manual_verified).length, [items]);
  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      const label = item.manual_label;
      const verified = item.manual_verified;
      if (filter === "pending") return !verified;
      if (filter === "verified") return verified;
      if (filter === "hate") return verified && label === "hate";
      if (filter === "non-hate") return verified && label === "non-hate";
      return true;
    });
  }, [items, filter]);

  const avatarUrl = (username?: string, userId?: number) => {
    const seed = username || String(userId || "user");
    return `https://ui-avatars.com/api/?background=0f172a&color=fff&name=${encodeURIComponent(seed)}`;
  };

  return (
    <div className="p-4 pb-24 h-full flex flex-col animate-fade-in relative">
        <div className="flex justify-between items-center mb-6">
            <div>
                <h2 className="text-2xl font-bold text-white">Manual Verification</h2>
                <p className="text-sm text-slate-400">Review flagged messages</p>
            </div>
            <div className="flex items-center space-x-2">
              <button
                className="p-2 rounded-full border border-slate-700 text-slate-200 hover:text-white hover:border-primary-400 transition"
                onClick={() => setShowFilters((prev) => !prev)}
                title="Filter"
              >
                <Filter size={18} />
              </button>
              <button
                className="p-2 rounded-full border border-slate-700 text-slate-200 hover:text-white hover:border-primary-400 transition"
                onClick={load}
                title="Refresh"
              >
                ↻
              </button>
            </div>
        </div>

        {showFilters && (
          <div className="flex flex-wrap items-center gap-2 mb-4">
            {(["pending", "verified", "hate", "non-hate", "all"] as const).map((key) => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all ${
                  filter === key
                    ? "border-primary-400 text-white bg-slate-800"
                    : "border-slate-700 text-slate-400 hover:text-white"
                }`}
              >
                {key === "pending"
                  ? "Pending"
                  : key === "verified"
                  ? "Verified"
                  : key === "hate"
                  ? "Hate"
                  : key === "non-hate"
                  ? "Non-hate"
                  : "All"}
              </button>
            ))}
          </div>
        )}

        <div className="flex-1 relative">
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400">
                Memuat data...
              </div>
            )}
            {!loading && filteredItems.length === 0 ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                    <Check size={48} className="mb-4 text-green-500 opacity-50" />
                    <p>All caught up! No pending reviews.</p>
                </div>
            ) : (
                <div className="space-y-4">
                    {filteredItems.map((item) => (
                        <div 
                            key={item.id} 
                            id={`card-${item.id}`}
                            className={`glass-panel p-5 rounded-3xl border border-slate-700/50 shadow-lg transition-all duration-300 transform ${
                              item.manual_verified ? "border-green-600/50 bg-green-500/5" : ""
                            }`}
                        >
                            <div className="flex items-center space-x-3 mb-4">
                                <div className="w-10 h-10 rounded-full bg-slate-700 overflow-hidden">
                                    <img src={avatarUrl(item.username, item.user_id)} alt="Avatar" className="w-10 h-10 object-cover" />
                                </div>
                                <div>
                                    <h4 className="text-white font-bold">{item.username ?? `User ${item.user_id ?? "-"}`}</h4>
                                    <span className="text-xs text-slate-400">{formatTime(item.created_at)}</span>
                                </div>
                                <div className="ml-auto text-right">
                                    <div className="text-[10px] text-slate-500 uppercase tracking-wide">Confidence</div>
                                    <div className={`text-xl font-mono font-bold ${ (item.prob_hate ?? 0) > 0.8 ? 'text-red-500' : 'text-orange-500'}`}>
                                        {Math.round((item.prob_hate ?? 0) * 100)}%
                                    </div>
                                    {item.manual_verified && (
                                      <div className="mt-1 inline-flex items-center text-green-400 text-[11px] font-semibold">
                                        <Shield size={12} className="mr-1" /> Verified: {item.manual_label === "hate" ? "Hate" : "Non-hate"}
                                      </div>
                                    )}
                                </div>
                            </div>

                            <div className="bg-slate-900/50 p-4 rounded-xl mb-6 border border-slate-800">
                                <p className="text-slate-200 text-sm leading-relaxed">
                                    "{item.text ?? item.reason ?? 'Tidak ada konten'}"
                                </p>
                            </div>

                            {!item.manual_verified && (
                              <div className="flex space-x-3">
                                  <button 
                                      disabled={verifying === item.id}
                                      onClick={() => handleDecision(item.id, "non-hate")}
                                      className={`flex-1 py-3 rounded-xl bg-green-500/10 text-green-500 border border-green-500/30 hover:bg-green-500 hover:text-white transition-all font-bold flex items-center justify-center ${verifying === item.id ? 'opacity-70 cursor-not-allowed' : ''}`}
                                  >
                                      <Check size={18} className="mr-2" /> Mark Safe
                                  </button>
                                  <button 
                                      disabled={verifying === item.id}
                                      onClick={() => handleDecision(item.id, "hate")}
                                      className={`flex-1 py-3 rounded-xl bg-red-500/10 text-red-500 border border-red-500/30 hover:bg-red-500 hover:text-white transition-all font-bold flex items-center justify-center ${verifying === item.id ? 'opacity-70 cursor-not-allowed' : ''}`}
                                  >
                                      <X size={18} className="mr-2" /> Hate Speech
                                  </button>
                              </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    </div>
  );
};
