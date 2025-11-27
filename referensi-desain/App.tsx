import React, { useEffect, useMemo, useState } from 'react';
import { Layout } from './components/Layout';
import { Dashboard } from './components/Dashboard';
import { Stats } from './components/Stats';
import { Logs } from './components/Logs';
import { Verification } from './components/Verification';
import { AdminPanel } from './components/AdminPanel';
import { Restricted } from './components/Restricted';
import { View } from './types';
import {
  fetchGroups,
  fetchSettings,
  updateSettings,
  fetchStats,
  fetchEvents,
  GroupSummary,
  SettingsResponse,
  StatsResponse,
  EventEntry,
  HttpError,
} from "./lib/api";
import { getUserAvatar } from "./lib/telegram";

const THRESHOLD_MIN = 0.2;
const THRESHOLD_MAX = 0.95;
type ModeSelection = SettingsResponse["mode"] | "custom";
const MODE_PRESETS: Record<SettingsResponse["mode"], number> = {
  precision: 0.561,
  balanced: 0.561,
  recall: 0.384,
};

function App() {
  const [currentView, setCurrentView] = useState<View>(View.DASHBOARD);
  const [chatId, setChatId] = useState<number | null>(null);
  const [groups, setGroups] = useState<GroupSummary[]>([]);
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [events, setEvents] = useState<EventEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [thresholdState, setThresholdState] = useState<Record<number, { value: number; mode: ModeSelection }>>({});
  const [restricted, setRestricted] = useState<string | null>(null);
  const [avatarUrl, setAvatarUrl] = useState<string | undefined>(undefined);

  const notify = (msg: string) => {
    if (typeof window !== "undefined") alert(msg);
  };

  const manualMode = settings ? !settings.enabled : false;

  const currentThresholdState = chatId ? thresholdState[chatId] : undefined;
  const derivedModeFromSettings: ModeSelection = useMemo(() => {
    if (!settings) return "balanced";
    const preset = MODE_PRESETS[settings.mode];
    return Math.abs(settings.threshold - preset) > 0.005 ? "custom" : settings.mode;
  }, [settings]);
  const modeSelection: ModeSelection = currentThresholdState?.mode ?? derivedModeFromSettings;
  const thresholdPreview = currentThresholdState?.value ?? settings?.threshold ?? 0.62;

  const liveActivity = useMemo(() => {
    return events.slice(0, 3).map((ev) => {
      const action = ev.action ?? "event";
      const tone = action === "banned" || action === "blocked" ? "danger" : action === "warned" || action === "muted" ? "warning" : "muted";
      const badge =
        action === "banned" || action === "blocked"
          ? "BLOCKED"
          : action === "warned" || action === "muted"
            ? "FLAGGED"
            : action.toUpperCase();
      return {
        id: ev.id,
        badge,
        tone,
        text: `${ev.username ?? ev.user_id ?? "user"}: ${ev.text ?? ev.reason ?? "Tindakan tercatat"}`,
        time: ev.created_at,
      };
    });
  }, [events]);

  const metrics = useMemo(() => {
    const total = stats?.total_events ?? 0;
    const deleted = stats?.deleted ?? 0;
    const warnOnly = stats?.warn_only ?? stats?.warned ?? 0;
    const muted = stats?.muted ?? stats?.warned ?? 0;
    return [
      { title: "Total aksi", value: total, subtitle: "Periode ini" },
      { title: "Dihapus", value: deleted, subtitle: "Confidence tinggi" },
      { title: "Auto-Muted", value: muted, subtitle: "Sementara" },
      { title: "Warnings", value: warnOnly, subtitle: "Sudah diekskalasi" },
    ];
  }, [stats]);

  const loadCore = async (selectedChat: number) => {
    setRefreshing(true);
    try {
      const [s, st, evs] = await Promise.all([
        fetchSettings(selectedChat),
        fetchStats(selectedChat, "24h"),
        fetchEvents(selectedChat, 5, 0),
      ]);
      setSettings(s);
      setThresholdState((prev) => ({
        ...prev,
        [selectedChat]: {
          value: s.threshold,
          mode:
            Math.abs(s.threshold - MODE_PRESETS[s.mode]) > 0.005 ? "custom" : (s.mode as ModeSelection),
        },
      }));
      setStats(st);
      setEvents(evs);
      setLastUpdated(new Date());
    } catch (err) {
      if (err instanceof HttpError) {
        if (err.status === 401 || err.status === 403) {
          setRestricted(err.message || "Dashboard hanya untuk admin grup");
          setChatId(null);
          setSettings(null);
          setStats(null);
          setEvents([]);
        } else {
          notify(`Gagal memuat data: ${err.message}`);
        }
      }
    } finally {
      setRefreshing(false);
    }
  };

  const loadGroups = async () => {
    setLoading(true);
    try {
      const data = await fetchGroups();
      setGroups(data);
      if (data.length) {
        setRestricted(null);
        const params = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "");
        const qChat = params.get("chat_id");
        const candidate = qChat ? Number(qChat) : null;
        const matched = candidate && data.find((g) => g.chat_id === candidate) ? candidate : null;
        const target = matched ?? chatId ?? data[0].chat_id;
        if (target !== chatId) setChatId(target);
        await loadCore(target);
      } else {
        setChatId(null);
        setSettings(null);
        setStats(null);
        setEvents([]);
        setRestricted("Dashboard hanya untuk admin grup");
      }
    } catch (err) {
      if (err instanceof HttpError) {
        if (err.status === 401 || err.status === 403) {
          setRestricted(err.message || "Dashboard hanya untuk admin grup");
          setChatId(null);
          setSettings(null);
          setStats(null);
          setEvents([]);
        } else {
          notify(`Gagal mengambil grup: ${err.message}`);
        }
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadGroups();
    setAvatarUrl(getUserAvatar());
  }, []);

  useEffect(() => {
    if (!chatId || restricted) return;
    loadCore(chatId);
  }, [chatId, restricted]);

  useEffect(() => {
    if (!autoRefresh || !chatId || restricted) return;
    const timer = setInterval(() => loadCore(chatId), 1000);
    return () => clearInterval(timer);
  }, [autoRefresh, chatId, restricted]);

  const handleSettingsUpdate = async (payload: Partial<SettingsResponse>) => {
    if (!chatId) return;
    try {
      const updated = await updateSettings(chatId, payload);
      setSettings(updated);
    } catch (err) {
      if (err instanceof HttpError) notify((err as Error).message ?? "Gagal menyimpan pengaturan");
    }
  };

  const handleToggleRealtime = () => {
    setAutoRefresh((prev) => !prev);
  };

  const handleModeSelect = async (mode: SettingsResponse["mode"]) => {
    if (!chatId) return;
    const preset = MODE_PRESETS[mode];
    const clamped = Math.min(
      THRESHOLD_MAX,
      Math.max(THRESHOLD_MIN, preset ?? (settings?.threshold ?? 0.62)),
    );
    setThresholdState((prev) => ({ ...prev, [chatId]: { value: clamped, mode } }));
    const value = Number(clamped.toFixed(2));
    await handleSettingsUpdate({ mode, threshold: value });
  };

  const handleThresholdPreviewChange = (value: number) => {
    if (!chatId) return;
    setThresholdState((prev) => ({
      ...prev,
      [chatId]: {
        value,
        mode: "custom",
      },
    }));
  };

  const commitThreshold = async () => {
    if (!settings || !chatId) return;
    const state = thresholdState[chatId];
    if (!state) return;
    if (Math.abs(state.value - settings.threshold) < 0.001) return;
    const value = Number(state.value.toFixed(2));
    await handleSettingsUpdate({ threshold: value });
    setThresholdState((prev) => ({ ...prev, [chatId]: { ...state, value } }));
  };

  const lastUpdateText = lastUpdated ? `Updated ${Math.max(0, Math.floor((Date.now() - lastUpdated.getTime()) / 1000))}s ago` : "Just now";

  const formatLastActive = (value?: string | null) => {
    if (!value) return "Belum ada aktivitas";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" });
  };

  const dashboardGroups = groups.map((g) => ({
    id: g.chat_id,
    name: g.title ?? `${g.chat_id}`,
    cid: `${g.chat_id}`,
    active: g.chat_id === chatId,
    status: g.enabled,
    lastActive: formatLastActive(g.last_active),
    groupType: g.group_type ?? undefined,
  }));

  const handleToggleGroupStatus = async (groupId: number, currentStatus: boolean) => {
    try {
      await updateSettings(groupId, { enabled: !currentStatus });
      await loadGroups();
      if (chatId === groupId) {
        await loadCore(groupId);
      }
    } catch (err) {
      if (err instanceof HttpError) notify((err as Error).message ?? "Gagal mengubah status moderasi grup");
    }
  };

  const renderView = () => {
    switch (currentView) {
      case View.DASHBOARD:
        return (
          <Dashboard
            manualMode={manualMode}
            lastUpdate={lastUpdateText}
            refreshing={refreshing}
            realtimeOn={autoRefresh}
            onToggleRealtime={handleToggleRealtime}
            onRefresh={() => chatId && loadCore(chatId)}
            groups={dashboardGroups}
            onSelectGroup={(id) => setChatId(id)}
            onToggleGroupStatus={handleToggleGroupStatus}
            metrics={metrics}
            liveActivity={liveActivity}
            modeSelection={modeSelection}
            thresholdPreview={thresholdPreview}
            thresholdMin={THRESHOLD_MIN}
            thresholdMax={THRESHOLD_MAX}
            onModeSelect={handleModeSelect}
            onThresholdChange={handleThresholdPreviewChange}
            onThresholdCommit={commitThreshold}
          />
        );
      case View.STATS:
        return chatId ? <Stats chatId={chatId} /> : <div className="p-4 text-slate-400">Pilih grup terlebih dahulu.</div>;
      case View.LOGS:
        return chatId ? <Logs chatId={chatId} /> : <div className="p-4 text-slate-400">Pilih grup terlebih dahulu.</div>;
      case View.VERIFY:
        return chatId ? <Verification chatId={chatId} /> : <div className="p-4 text-slate-400">Pilih grup terlebih dahulu.</div>;
      case View.ADMIN:
        return chatId ? <AdminPanel chatId={chatId} /> : <div className="p-4 text-slate-400">Pilih grup terlebih dahulu.</div>;
      default:
        return <Dashboard />;
    }
  };

  if (restricted) {
    return (
      <Layout currentView={currentView} onViewChange={() => undefined} hideNav avatarUrl={avatarUrl}>
        <Restricted reason={restricted} />
      </Layout>
    );
  }

  return (
    <Layout currentView={currentView} onViewChange={setCurrentView} avatarUrl={avatarUrl}>
      {renderView()}
    </Layout>
  );
}

export default App;
