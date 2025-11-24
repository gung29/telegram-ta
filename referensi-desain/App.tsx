import React, { useEffect, useMemo, useState } from 'react';
import { Layout } from './components/Layout';
import { Dashboard } from './components/Dashboard';
import { Stats } from './components/Stats';
import { Logs } from './components/Logs';
import { Verification } from './components/Verification';
import { AdminPanel } from './components/AdminPanel';
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

  const notify = (msg: string) => {
    if (typeof window !== "undefined") alert(msg);
  };

  const manualMode = settings ? !settings.enabled : false;

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
    const warned = stats?.warned ?? 0;
    const blocked = stats?.blocked ?? 0;
    return [
      { title: "Total Actions", value: total, subtitle: "Last window" },
      { title: "Deleted", value: deleted, subtitle: "High confidence" },
      { title: "Auto-Muted", value: warned, subtitle: "Temporary" },
      { title: "Warnings", value: blocked, subtitle: "Escalated" },
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
      setStats(st);
      setEvents(evs);
      setLastUpdated(new Date());
    } catch (err) {
      if (err instanceof HttpError) {
        notify(`Gagal memuat data: ${err.message}`);
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
        const params = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "");
        const qChat = params.get("chat_id");
        const candidate = qChat ? Number(qChat) : null;
        const matched = candidate && data.find((g) => g.chat_id === candidate) ? candidate : null;
        const target = matched ?? chatId ?? data[0].chat_id;
        if (target !== chatId) setChatId(target);
        await loadCore(target);
      } else {
        setChatId(null);
      }
    } catch (err) {
      if (err instanceof HttpError) notify(`Gagal mengambil grup: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadGroups();
  }, []);

  useEffect(() => {
    if (!chatId) return;
    loadCore(chatId);
  }, [chatId]);

  useEffect(() => {
    if (!autoRefresh || !chatId) return;
    const timer = setInterval(() => loadCore(chatId), 1500);
    return () => clearInterval(timer);
  }, [autoRefresh, chatId]);

  const handleToggleMode = async () => {
    if (!chatId || !settings) return;
    const next = !settings.enabled;
    try {
      const updated = await updateSettings(chatId, { enabled: next });
      setSettings(updated);
    } catch (err) {
      if (err instanceof HttpError) notify(err.message);
    }
  };

  const lastUpdateText = lastUpdated ? `Updated ${Math.max(0, Math.floor((Date.now() - lastUpdated.getTime()) / 1000))}s ago` : "Just now";

  const dashboardGroups = groups.map((g) => ({
    id: g.chat_id,
    name: g.title ?? `${g.chat_id}`,
    cid: `${g.chat_id}`,
    active: g.chat_id === chatId,
    status: g.enabled,
    lastActive: g.last_active,
  }));

  const renderView = () => {
    switch (currentView) {
      case View.DASHBOARD:
        return (
          <Dashboard
            manualMode={manualMode}
            lastUpdate={lastUpdateText}
            refreshing={refreshing}
            onToggleMode={handleToggleMode}
            onRefresh={() => chatId && loadCore(chatId)}
            groups={dashboardGroups}
            onSelectGroup={(id) => setChatId(id)}
            metrics={metrics}
            liveActivity={liveActivity}
          />
        );
      case View.STATS:
        return chatId ? <Stats chatId={chatId} /> : <div className="p-4 text-slate-400">Pilih grup terlebih dahulu.</div>;
      case View.LOGS:
        return <Logs />;
      case View.VERIFY:
        return <Verification />;
      case View.ADMIN:
        return <AdminPanel />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <Layout currentView={currentView} onViewChange={setCurrentView}>
      {renderView()}
    </Layout>
  );
}

export default App;
