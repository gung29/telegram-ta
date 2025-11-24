import React, { useEffect, useMemo, useState } from 'react';
import dayjs from "dayjs";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar, Legend } from 'recharts';
import { fetchStats, fetchActivity, StatsResponse, ActivityResponse, HttpError } from "../lib/api";

type Props = { chatId: number };

export const Stats: React.FC<Props> = ({ chatId }) => {
  const [windowKey, setWindowKey] = useState<"24h" | "7d" | "30d">("7d");
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [activity, setActivity] = useState<ActivityResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const notify = (msg: string) => {
    if (typeof window !== "undefined") alert(msg);
  };

  const load = async () => {
    setLoading(true);
    try {
      const [s, a] = await Promise.all([
        fetchStats(chatId, windowKey),
        fetchActivity(chatId, windowKey === "24h" ? 2 : windowKey === "7d" ? 7 : 30),
      ]);
      setStats(s);
      setActivity(a);
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [chatId, windowKey]);

  // gunakan angka agregat langsung dari StatsResponse untuk ringkasan/distribusi
  const aggregated = useMemo(() => {
    const warn = stats?.warned ?? 0;
    const deleted = stats?.deleted ?? 0;
    const blocked = stats?.blocked ?? 0;
    const total = stats?.total_events ?? deleted; // jika total_events ada, ikuti API
    return { warn, deleted, blocked, total };
  }, [stats]);

  const chartData = useMemo(() => {
    if (!activity) return [];
    return activity.points.map((p) => {
      const label = dayjs(p.date).isValid() ? dayjs(p.date).format(windowKey === "24h" ? "DD MMM HH:mm" : "DD MMM") : "";
      const del = p.deleted ?? 0;
      const warn = p.warned ?? 0;
      const block = p.blocked ?? 0;
      return {
        name: label,
        total: del, // total harian mengikuti field deleted (sesuai API)
        warn,
        block,
      };
    });
  }, [activity, windowKey]);

  const actionDistribution = useMemo(() => {
    return [
      { name: "Deleted", value: aggregated.deleted, color: "#6366f1" },
      { name: "Warned", value: aggregated.warn, color: "#f59e0b" },
      { name: "Blocked", value: aggregated.blocked, color: "#ef4444" },
    ];
  }, [aggregated]);

  const totals = useMemo(() => {
    const total = aggregated.total || 0;
    const peak = chartData.length ? Math.max(...chartData.map((d) => d.total ?? 0)) : 0;
    return { total, peak };
  }, [aggregated, chartData]);

  const topOffenders = useMemo(() => {
    const list = stats?.top_offenders ?? [];
    return list.map((entry, idx) => {
      const match = entry.match(/^(.*)\s+\((\d+)\)$/);
      return {
        id: `${idx}`,
        username: match ? match[1].trim() : entry,
        userId: "",
        acts: match ? Number(match[2]) : 0,
      };
    });
  }, [stats]);

  return (
    <div className="p-4 space-y-6 pb-24 animate-fade-in">
        <div className="flex bg-slate-800 p-1 rounded-xl">
            {['24h', '7d', '30d'].map((p) => (
                <button
                  key={p}
                  onClick={() => setWindowKey(p as typeof windowKey)}
                  className={`flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ${windowKey === p ? 'bg-slate-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}>
                    {p === "24h" ? "Today" : p === "7d" ? "7 Days" : "30 Days"}
                </button>
            ))}
        </div>

        <div className="glass-panel p-5 rounded-3xl border border-slate-700/50">
            <div className="flex items-center justify-between mb-2">
              <div>
                <h3 className="text-white font-bold mb-1">Moderation Activity</h3>
                <p className="text-xs text-slate-400">Jendela: {windowKey}</p>
              </div>
              {loading && <span className="text-xs text-slate-400">Loading…</span>}
            </div>
            <div className="flex space-x-4 text-xs mb-4">
                <span className="flex items-center text-blue-400"><span className="w-2 h-2 rounded-full bg-blue-400 mr-1"></span> Total</span>
                <span className="flex items-center text-orange-400"><span className="w-2 h-2 rounded-full bg-orange-400 mr-1"></span> Warned</span>
                <span className="flex items-center text-red-400"><span className="w-2 h-2 rounded-full bg-red-400 mr-1"></span> Blocked</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs text-slate-300 mb-3">
              <div className="p-2 bg-slate-800/60 rounded-lg border border-slate-700">
                <p className="text-slate-400">Total</p>
                <p className="text-white text-lg font-bold">{totals.total}</p>
              </div>
              <div className="p-2 bg-slate-800/60 rounded-lg border border-slate-700">
                <p className="text-slate-400">Peak</p>
                <p className="text-white text-lg font-bold">{totals.peak}</p>
              </div>
            </div>
            <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={chartData}
                      margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
                    >
                        <defs>
                            <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorWarn" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#f97316" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="name" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                            itemStyle={{ color: '#e2e8f0' }}
                        />
                        <Area type="monotone" dataKey="total" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorTotal)" />
                        <Area type="monotone" dataKey="warn" stroke="#f97316" strokeWidth={2} fillOpacity={1} fill="url(#colorWarn)" />
                        <Area type="monotone" dataKey="block" stroke="#ef4444" strokeWidth={2} fillOpacity={0} fill="transparent" />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>

        <div className="glass-panel p-4 rounded-3xl border border-slate-700/50">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-white font-bold">Action Distribution</h3>
            {loading && <span className="text-xs text-slate-400">Loading…</span>}
          </div>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={actionDistribution}
                margin={{ top: 4, right: 8, bottom: 0, left: -10 }}
              >
                <XAxis dataKey="name" stroke="#475569" fontSize={11} axisLine={false} tickLine={false} />
                <YAxis stroke="#475569" fontSize={11} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#020617', borderColor: '#1e293b', borderRadius: 10 }}
                  formatter={(value: number, _name: string, entry: any) => [value, entry?.payload?.name]}
                  labelStyle={{ color: '#cbd5f5' }}
                  itemStyle={{ color: '#e5e7eb' }}
                />
                <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                  {actionDistribution.map((entry, idx) => (
                    <Cell key={`bar-${idx}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
            <div className="flex justify-between items-center mb-3 px-1">
                <h3 className="text-white font-bold">Top Offenders</h3>
                <span className="text-xs text-slate-500">Window: {windowKey}</span>
            </div>
            <div className="space-y-3">
                {topOffenders.length === 0 && <p className="text-slate-400 text-sm">Belum ada data.</p>}
                {topOffenders.map((user, idx) => (
                    <div key={user.id} className="glass-panel p-3 rounded-2xl flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center text-white font-bold">
                          {user.username.slice(0,2).toUpperCase()}
                        </div>
                        <div className="flex-1 min-w-0">
                            <h4 className="text-sm font-bold text-white truncate">{user.username}</h4>
                            <div className="text-[10px] text-slate-400 font-mono">Acts: {user.acts}</div>
                        </div>
                        <div className="text-right">
                             <div className="text-sm font-bold text-neon-blue">{user.acts} acts</div>
                             <div className="w-24 h-1.5 bg-slate-800 rounded-full mt-1 overflow-hidden">
                                 <div 
                                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500" 
                                    style={{ width: `${Math.max(20, 100 - idx * 20)}%` }}
                                 ></div>
                             </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>

    </div>
  );
};
