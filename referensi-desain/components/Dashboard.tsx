import React from 'react';
import { RefreshCw, Shield, AlertTriangle, UserX, VolumeX } from 'lucide-react';

type LiveItem = { id: number; badge: string; tone: "danger" | "warning" | "muted"; text: string; time?: string };
type Metric = { title: string; value: number; subtitle: string; icon: any; color: string; bg: string };
type GroupItem = {
  id: number;
  name: string;
  cid: string;
  active: boolean;
  status: boolean;
  lastActive: string;
  groupType?: string;
};
type ModeSelection = "precision" | "balanced" | "recall" | "custom";

type Props = {
  manualMode: boolean;
  lastUpdate: string;
  refreshing: boolean;
  realtimeOn: boolean;
  onToggleRealtime: () => void;
  onRefresh: () => void;
  groups: GroupItem[];
  onSelectGroup: (id: number) => void;
  onToggleGroupStatus: (id: number, currentStatus: boolean) => void;
  metrics: Array<{ title: string; value: number; subtitle: string }>;
  liveActivity: LiveItem[];
  modeSelection: ModeSelection;
  thresholdPreview: number;
  thresholdMin: number;
  thresholdMax: number;
  onModeSelect: (mode: "precision" | "balanced" | "recall") => void;
  onThresholdChange: (value: number) => void;
  onThresholdCommit: () => void;
};

export const Dashboard: React.FC<Props> = ({
  manualMode,
  lastUpdate,
  refreshing,
  realtimeOn,
  onToggleRealtime,
  onRefresh,
  groups,
  onSelectGroup,
  onToggleGroupStatus,
  metrics,
  liveActivity,
  modeSelection,
  thresholdPreview,
  thresholdMin,
  thresholdMax,
  onModeSelect,
  onThresholdChange,
  onThresholdCommit,
}) => {
  const mappedMetrics: Metric[] = [
    { ...metrics[0], icon: Shield, color: "text-blue-400", bg: "from-blue-500/10 to-transparent" },
    { ...metrics[1], icon: UserX, color: "text-red-400", bg: "from-red-500/10 to-transparent" },
    { ...metrics[2], icon: VolumeX, color: "text-purple-400", bg: "from-purple-500/10 to-transparent" },
    { ...metrics[3], icon: AlertTriangle, color: "text-orange-400", bg: "from-orange-500/10 to-transparent" },
  ];

  return (
    <div className="p-4 space-y-6 pb-24 animate-fade-in">
      {/* Header Status Card */}
      <div className="glass-panel p-5 rounded-3xl relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-10">
          <Shield size={120} />
        </div>
        <div className="relative z-10">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                    <div
                      className={`w-3 h-3 rounded-full transition-all duration-300
                        ${manualMode
                          ? 'bg-orange-400 shadow-[0_0_6px_2px_rgba(251,146,60,0.8)]'
                          : 'bg-green-400 shadow-[0_0_8px_3px_rgba(34,197,94,0.8)]'
                        }`}
                    ></div>

                    <h2
                      className={`text-lg font-bold tracking-wide transition-colors duration-300
                        ${manualMode ? 'text-orange-100' : 'text-green-400'}
                      `}
                    >
                      {manualMode ? 'Mode manual' : 'Perlindungan otomatis'}
                    </h2>
                </div>
                <span className="text-xs text-slate-400 font-mono">{lastUpdate}</span>
            </div>
            
            <p className="text-slate-400 text-sm mb-6">
                {manualMode 
                    ? "System saat ini hanya menandai konten untuk ditinjau terlebih dahulu." 
                    : "System sedang aktif menyaring ancaman dengan confidence tinggi."}
            </p>

            <div className="flex items-center justify-between">
                <label className="flex items-center cursor-pointer select-none">
                    <div className="relative">
                        <input type="checkbox" className="sr-only" checked={realtimeOn} onChange={onToggleRealtime} />
                        <div
                          className={`block w-12 h-7 rounded-full transition-colors duration-300 ${
                            realtimeOn ? 'bg-emerald-500' : 'bg-slate-700'
                          }`}
                        ></div>
                        <div
                          className={`dot absolute top-0.5 left-0.5 w-6 h-6 rounded-full bg-white shadow-sm transition-transform duration-300 ${
                            realtimeOn ? 'translate-x-4' : ''
                          }`}
                        ></div>
                    </div>
                    <span className={`ml-3 text-sm font-medium transition-colors ${realtimeOn ? 'text-white' : 'text-slate-300'}`}>
                        {realtimeOn ? 'Realtime aktif' : 'Realtime nonaktif'}
                    </span>
                </label>
                
                <button 
                    onClick={onRefresh}
                    className="flex items-center space-x-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-full text-xs font-bold border border-slate-600 transition-all active:scale-95 hover:shadow-[0_10px_25px_rgba(15,23,42,0.8)]"
                >
                    <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
                    <span>Refresh data</span>
                </button>
            </div>
        </div>
      </div>

      {/* Monitored Groups */}
      <div>
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 pl-1">Grup yang dipantau</h3>
        <div className="flex overflow-x-auto space-x-3 pb-2 scrollbar-hide">
            {groups.map(g => (
                <button
                  key={g.id}
                  onClick={() => onSelectGroup(g.id)}
                  className={`flex-shrink-0 w-64 p-4 rounded-2xl border transition-all duration-300 text-left transform ${
                    g.active
                      ? 'bg-gradient-to-br from-slate-800 to-slate-900 border-primary-500/30 shadow-[0_18px_40px_rgba(15,23,42,0.9)]'
                      : 'bg-slate-900 border-slate-800 opacity-70 hover:opacity-100 hover:-translate-y-1 hover:shadow-[0_14px_30px_rgba(15,23,42,0.8)]'
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                      <h4 className="font-bold text-white truncate pr-2">{g.name}</h4>
                      <div
                        className={`
                          w-2 h-2 rounded-full transition-all duration-300
                          ${g.status
                            ? 'bg-[#22c55e] shadow-[0_0_10px_3px_rgba(34,197,94,0.9)]'
                            : 'bg-slate-500 shadow-[0_0_4px_1px_rgba(100,116,139,0.5)]'
                          }
                        `}
                      ></div>
                  </div>
                  <p className="text-xs text-slate-500 font-mono">ID: {g.cid}</p>
                  <p className="text-[10px] text-slate-500 mb-3">
                    Tipe:{" "}
                    <span className="uppercase tracking-wide">
                      {g.groupType === "supergroup"
                        ? "Supergroup"
                        : g.groupType === "group"
                        ? "Group"
                        : g.groupType || "Tidak diketahui"}
                    </span>
                  </p>
                  <p className="text-[10px] text-slate-400 mb-2">
                    Terakhir aktif: <span className="font-mono text-slate-300">{g.lastActive}</span>
                  </p>
                  <div className="flex items-center justify-between">
                      <span className={`text-xs font-medium ${g.status ? 'text-neon-green' : 'text-slate-500'}`}>
                          {g.status ? 'Aktif' : 'Dijeda'}
                      </span>
                       <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onToggleGroupStatus(g.id, g.status);
                        }}
                        className={`
                          relative w-10 h-5 rounded-full transition-colors duration-200
                          ${g.status ? 'bg-purple-500' : 'bg-slate-700'}
                        `}
                      >
                        <span
                          className={`
                            absolute top-0.5 left-0.5 w-3.5 h-3.5 rounded-full bg-white shadow-sm
                            transition-transform duration-200
                            ${g.status ? 'translate-x-[1.375rem]' : 'translate-x-0'}
                          `}
                        />
                      </button>
                  </div>
                </button>
            ))}
        </div>
      </div>

      {/* Summary Stats Grid */}
        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 pl-1">Hasil Moderasi</h3>
        <div className="grid grid-cols-2 gap-3">
            {mappedMetrics.map((m, idx) => (
              <StatCard 
                key={idx}
                title={m.title}
                value={m.value.toLocaleString()}
                subtitle={m.subtitle}
                icon={m.icon}
                color={m.color}
                bg={m.bg}
              />
            ))}
        </div>
      </div>

      {/* Threshold & Mode Controls */}
      <div className="glass-panel p-4 rounded-3xl border border-slate-700/50 space-y-4">
        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 pl-1">Mode</h3>
          <div className="grid grid-cols-2 gap-2">
            {(["precision", "balanced", "recall"] as const).map((mode) => {
              const active = modeSelection === mode;
              return (
                <button
                  key={mode}
                  onClick={() => onModeSelect(mode)}
                  className={`px-3 py-2 rounded-xl text-xs font-semibold capitalize transition-all border transform ${
                    active
                      ? "bg-slate-800 border-primary-400 text-white shadow hover:shadow-[0_16px_36px_rgba(15,23,42,0.9)]"
                      : "bg-slate-900/70 border-slate-700 text-slate-300 hover:border-primary-400/60 hover:text-white hover:-translate-y-0.5"
                  }`}
                >
                  {mode}
                </button>
              );
            })}
            <div
              className={`px-3 py-2 rounded-xl text-xs font-semibold capitalize border text-left ${
                modeSelection === "custom"
                  ? "bg-slate-800 border-primary-400 text-white"
                  : "bg-slate-900/70 border-slate-700 text-slate-400"
              }`}
            >
              custom
              <p className="text-[10px] text-slate-500 mt-1">Diatur manual melalui slider.</p>
            </div>
          </div>
        </div>

        <div>
          <div className="flex items-baseline justify-between mb-2">
            <h4 className="text-xs font-semibold text-slate-300">Ambang moderasi</h4>
            <span className="text-xs font-mono text-primary-300">
              {(thresholdPreview * 100).toFixed(1)}%
            </span>
          </div>
          <div className="px-3 py-3 rounded-2xl bg-slate-900/70 border border-slate-800 hover:border-primary-500/40 transition-colors">
            <input
              type="range"
              min={thresholdMin}
              max={thresholdMax}
              step={0.01}
              value={thresholdPreview}
              onChange={(e) => onThresholdChange(Number(e.target.value))}
              onMouseUp={onThresholdCommit}
              onPointerUp={onThresholdCommit}
              onTouchEnd={onThresholdCommit}
              onBlur={onThresholdCommit}
              className="w-full accent-purple-400 cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Live Feed Teaser */}
      <div className="glass-panel rounded-2xl p-4 border-l-4 border-neon-blue">
        <div className="flex justify-between items-center mb-2">
            <h4 className="font-bold text-white">Aktivitas live</h4>
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neon-blue opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-neon-blue"></span>
            </span>
        </div>
        <div className="space-y-3">
            {liveActivity.length === 0 && <p className="text-sm text-slate-400">Belum ada aktivitas terbaru.</p>}
            {liveActivity.map(item => (
              <div key={item.id} className="text-sm text-slate-300 border-b border-slate-800 pb-2 last:border-0">
                <span className={`${item.tone === 'danger' ? 'text-red-400' : item.tone === 'warning' ? 'text-orange-400' : 'text-slate-400'} font-mono font-bold text-xs`}>
                  [{item.badge}]
                </span>{" "}
                {item.text}
              </div>
            ))}
        </div>
      </div>

    </div>
  );
};

const StatCard: React.FC<{title: string, value: string, subtitle: string, icon: any, color: string, bg: string}> = ({
    title, value, subtitle, icon: Icon, color, bg
}) => (
    <div
      className={`p-4 rounded-2xl bg-gradient-to-br ${bg} border border-slate-800 relative overflow-hidden transition-transform duration-200 hover:-translate-y-0.5 hover:shadow-[0_18px_40px_rgba(15,23,42,0.85)]`}
    >
        <div className={`absolute top-3 right-3 opacity-20 ${color}`}>
            <Icon size={24} />
        </div>
        <div className="relative z-10">
            <h4 className="text-slate-400 text-xs font-medium mb-1">{title}</h4>
            <div className="text-2xl font-bold text-white font-mono mb-1">{value}</div>
            <div className="text-[10px] text-slate-500">{subtitle}</div>
        </div>
    </div>
);
