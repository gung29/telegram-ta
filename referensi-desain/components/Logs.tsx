import React, { useEffect, useMemo, useState } from 'react';
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import { Search, Filter, MoreHorizontal, Download, Clock } from 'lucide-react';
import { fetchEvents, EventEntry, HttpError, exportCsv } from '../lib/api';

dayjs.extend(relativeTime);

type Props = { chatId: number };

type TabFilter = "all" | "verified" | "flagged" | "muted" | "banned";
type TimeFilter = "all" | "24h" | "7d" | "30d";

export const Logs: React.FC<Props> = ({ chatId }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [tab, setTab] = useState<TabFilter>("all");
  const [timeFilter, setTimeFilter] = useState<TimeFilter>("all");
  const [sortNewest, setSortNewest] = useState(true);
  const [logs, setLogs] = useState<EventEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);

  const notify = (msg: string) => {
    if (typeof window !== "undefined") alert(msg);
  };

  const load = async () => {
    setLoading(true);
    try {
      const data = await fetchEvents(chatId, 200, 0);
      setLogs(data);
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [chatId]);

  const getActionColor = (action: string) => {
      switch(action) {
          case 'banned':
          case 'blocked':
            return 'text-red-500 bg-red-500/10 border-red-500/20';
          case 'muted':
          case 'warned':
            return 'text-orange-500 bg-orange-500/10 border-orange-500/20';
          case 'deleted':
            return 'text-sky-400 bg-sky-400/10 border-sky-400/20';
          default:
            return 'text-slate-400 bg-slate-800 border-slate-700';
      }
  };

  const getScoreColor = (score: number) => {
      if (score >= 0.9) return 'text-red-500';
      if (score >= 0.7) return 'text-orange-500';
      return 'text-green-500';
  };

  const passTabFilter = (item: EventEntry) => {
    switch (tab) {
      case "verified":
        return item.manual_verified;
      case "flagged":
        return !item.manual_verified;
      case "muted":
        return item.action === "muted" || item.action === "warned";
      case "banned":
        return item.action === "banned" || item.action === "blocked";
      default:
        return true;
    }
  };

  const passTimeFilter = (item: EventEntry) => {
    if (timeFilter === "all") return true;
    const d = dayjs(item.created_at);
    if (!d.isValid()) return true;
    const now = dayjs();
    const diff = now.diff(d, "hour");
    if (timeFilter === "24h") return diff <= 24;
    if (timeFilter === "7d") return diff <= 24 * 7;
    if (timeFilter === "30d") return diff <= 24 * 30;
    return true;
  };

  const filtered = useMemo(() => {
    const term = searchTerm.toLowerCase();
    const list = logs.filter((item) => {
      const text = item.text || item.reason || "";
      const user = item.username || String(item.user_id || "");
      const match = user.toLowerCase().includes(term) || text.toLowerCase().includes(term);
      return match && passTabFilter(item) && passTimeFilter(item);
    });
    const sorted = [...list].sort((a, b) => {
      const da = dayjs(a.created_at).valueOf();
      const db = dayjs(b.created_at).valueOf();
      return sortNewest ? db - da : da - db;
    });
    return sorted;
  }, [logs, searchTerm, tab, timeFilter, sortNewest]);

  const handleExport = async () => {
    setExporting(true);
    try {
      const blob = await exportCsv(chatId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `hate_guard_logs_${chatId}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      notify((err as Error).message ?? "Gagal mengunduh CSV");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="p-4 h-full flex flex-col pb-24 animate-fade-in">
        <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-white">History & Logs</h2>
            <button
              onClick={handleExport}
              disabled={exporting}
              className="p-2 rounded-full bg-slate-800 text-slate-400 hover:text-white disabled:opacity-60"
            >
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
                {[
                  { key: "all", label: "All" },
                  { key: "verified", label: "Verified" },
                  { key: "flagged", label: "Pending" },
                  { key: "muted", label: "Muted/Warned" },
                  { key: "banned", label: "Banned/Blocked" },
                ].map(({ key, label }) => (
                    <button
                      key={key}
                      onClick={() => setTab(key as TabFilter)}
                      className={`px-3 py-1.5 rounded-full border text-xs whitespace-nowrap transition ${
                        tab === key ? 'border-primary-500 text-white bg-slate-800' : 'border-slate-700 text-slate-300 hover:bg-slate-800'
                      }`}
                    >
                        {label}
                    </button>
                ))}
                <div className="flex items-center space-x-2 pl-2">
                  {[ "all", "24h", "7d", "30d" ].map((tf) => (
                    <button
                      key={tf}
                      onClick={() => setTimeFilter(tf as TimeFilter)}
                      className={`px-2.5 py-1 rounded-lg text-[11px] border transition ${
                        timeFilter === tf ? 'border-primary-500 text-white' : 'border-slate-700 text-slate-400'
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                  <button
                    onClick={() => setSortNewest((p) => !p)}
                    className="px-2.5 py-1 rounded-lg text-[11px] border border-slate-700 text-slate-300 hover:border-primary-400 transition flex items-center space-x-1"
                  >
                    <Clock size={12} />
                    <span>{sortNewest ? "Newest" : "Oldest"}</span>
                  </button>
                </div>
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
            {loading && <div className="text-slate-400 text-sm px-2">Memuat...</div>}
            {!loading && filtered.length === 0 && (
              <div className="text-slate-500 text-sm px-2">Tidak ada data.</div>
            )}
            {filtered.map((log) => {
              const score = log.prob_hate ?? 0;
              const timeLabel = dayjs(log.created_at).isValid()
                ? dayjs(log.created_at).fromNow()
                : log.created_at;
              return (
                <div key={log.id} className="glass-panel p-3 rounded-xl grid grid-cols-12 gap-2 items-center hover:bg-slate-800/50 transition-colors">
                    <div className="col-span-3">
                        <div className="text-xs text-slate-400 font-mono">{timeLabel}</div>
                        <div className="text-sm font-bold text-white truncate">{log.username ?? `User ${log.user_id ?? '-'}`}</div>
                    </div>
                    <div className="col-span-2 flex justify-center">
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${getActionColor(log.action)}`}>
                            {log.action ?? 'unknown'}
                        </span>
                    </div>
                    <div className="col-span-2 text-center">
                        <span className={`font-mono font-bold text-sm ${getScoreColor(score)}`}>
                            {Math.round(score * 100)}%
                        </span>
                    </div>
                    <div className="col-span-4">
                        <p className="text-xs text-slate-300 truncate opacity-80">{log.text ?? log.reason ?? 'Tidak ada konten'}</p>
                    </div>
                    <div className="col-span-1 flex justify-end">
                        <MoreHorizontal size={16} className="text-slate-500" />
                    </div>
                </div>
              );
            })}
        </div>
    </div>
  );
};
