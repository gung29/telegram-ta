import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import {
  addAdmin,
  createMemberStatus,
  deleteMemberStatus,
  removeAdmin,
  fetchActivity,
  fetchAdmins,
  fetchEvents,
  fetchEventCount,
  fetchGroups,
  fetchMembers,
  fetchSettings,
  fetchStats,
  runTest,
  updateSettings,
  verifyEvent,
  fetchUserActions,
  resetUserAction,
  GroupSummary,
  StatsResponse,
  SettingsResponse,
  AdminEntry,
  MemberModeration,
  MemberStatus,
  ActivityResponse,
  EventEntry,
  UserActionSummary,
  HttpError,
} from "./lib/api";
import { ensureTelegramReady, getDebugChatId, getTelegram } from "./lib/telegram";
import { clsx } from "clsx";
import {
  Chart as ChartJS,
  LineElement,
  ArcElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  BarElement,
} from "chart.js";
import { Line, Doughnut, Bar } from "react-chartjs-2";
import { 
  LayoutDashboard, 
  BarChart3, 
  FileText, 
  CheckCircle, 
  ShieldAlert,
  RefreshCw,
  Power,
  Shield,
  AlertTriangle,
  UserX,
  VolumeX,
  Search,
  Filter,
  MoreHorizontal,
  Download,
  Check,
  X,
  Trash2,
  AlertOctagon,
  RefreshCcw,
  ShieldCheck,
  Zap
} from "lucide-react";

dayjs.extend(relativeTime);
ChartJS.register(LineElement, ArcElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend, BarElement);

const REFRESH_MS = 1000;
const HISTORY_PAGE = 5;
const REVIEW_PAGE = 5;
const THRESHOLD_MIN = 0.2;
const THRESHOLD_MAX = 0.95;
type ModeSelection = SettingsResponse["mode"] | "custom";
type ActionFilter = "all" | "warn" | "block" | "allow";
type TimeFilter = "all" | "24h" | "7d" | "30d";

const MODE_PRESETS: Record<SettingsResponse["mode"], number> = {
  precision: 0.561,
  balanced: 0.561,
  recall: 0.384,
};

const ACTION_META: Record<string, { label: string; tone: "success" | "warning" | "danger" | "muted" }> = {
  muted: { label: "Dimute", tone: "warning" },
  banned: { label: "Diblokir", tone: "danger" },
  blocked: { label: "Diblokir", tone: "danger" },
  allowed: { label: "Diizinkan", tone: "success" },
  warned: { label: "Diperingatkan", tone: "warning" },
  bypassed_admin: { label: "Admin", tone: "muted" },
};
const VIOLATION_ACTIONS = new Set(["warned", "muted", "banned", "blocked"]);
const ACTION_COLOR_MAP: Record<string, string> = {
  warned: "#f97316",
  muted: "#fbbf24",
  banned: "#f43f5e",
  blocked: "#22c55e",
  allowed: "#22c55e",
};
const TOP_OFFENDER_WINDOWS: Array<{ value: "24h" | "7d"; label: string }> = [
  { value: "24h", label: "1 hari" },
  { value: "7d", label: "7 hari" },
];

const notify = (message: string) => {
  const tg = getTelegram();
  if (tg) tg.showAlert(message);
  else alert(message);
};

const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

// View types for navigation
type View = "DASHBOARD" | "STATS" | "LOGS" | "VERIFY" | "ADMIN";

export default function App() {
  const [chatId, setChatId] = useState<number | null>(null);
  const [groups, setGroups] = useState<GroupSummary[]>([]);
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [statsWindow, setStatsWindow] = useState<"24h" | "7d">("24h");
  const [topOffenderWindow, setTopOffenderWindow] = useState<"24h" | "7d">("24h");
  const [offenderStats, setOffenderStats] = useState<StatsResponse | null>(null);
  const [offenderLoading, setOffenderLoading] = useState(false);
  const [activity, setActivity] = useState<ActivityResponse | null>(null);
  const [admins, setAdmins] = useState<AdminEntry[]>([]);
  const [muted, setMuted] = useState<MemberModeration[]>([]);
  const [banned, setBanned] = useState<MemberModeration[]>([]);
  const [allEvents, setAllEvents] = useState<EventEntry[]>([]);
  const [testInput, setTestInput] = useState("");
  const [testResult, setTestResult] = useState("");
  const [pending, setPending] = useState(false);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [thresholdState, setThresholdState] = useState<Record<number, { value: number; mode: ModeSelection }>>({});
  const [retentionDraft, setRetentionDraft] = useState<number | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyPage, setHistoryPage] = useState(0);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewPage, setReviewPage] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [accessDenied, setAccessDenied] = useState(false);
  const [verifyingEvent, setVerifyingEvent] = useState<number | null>(null);
  const [userActions, setUserActions] = useState<UserActionSummary[]>([]);
  const [userActionsLoading, setUserActionsLoading] = useState(false);
  const [resettingAction, setResettingAction] = useState<{ userId: number; action: "warned" | "muted" } | null>(null);
  const [currentView, setCurrentView] = useState<View>("DASHBOARD");
  const [manualMode, setManualMode] = useState(false);
  
  const currentThresholdState = chatId ? thresholdState[chatId] : undefined;
  const derivedModeFromSettings = useMemo(() => {
    if (!settings) return "balanced";
    const preset = MODE_PRESETS[settings.mode];
    return Math.abs(settings.threshold - preset) > 0.005 ? "custom" : settings.mode;
  }, [settings]);
  const modeSelection: ModeSelection = currentThresholdState?.mode ?? derivedModeFromSettings;
  const thresholdPreview = currentThresholdState?.value ?? settings?.threshold ?? 0.62;
  const [manualFilter, setManualFilter] = useState<"all" | "pending" | "verified" | "hate" | "non-hate">("all");
  const [detectionFilter, setDetectionFilter] = useState<"all" | "hate" | "non-hate">("all");
  const [actionFilter, setActionFilter] = useState<ActionFilter>("all");
  const [timeFilter, setTimeFilter] = useState<TimeFilter>("all");
  const [searchTerm, setSearchTerm] = useState("");
  const manualFilterOptions: { value: "all" | "pending" | "verified" | "hate" | "non-hate"; label: string }[] = [
    { value: "all", label: "Semua" },
    { value: "pending", label: "Belum dilabeli" },
    { value: "verified", label: "Sudah dilabeli" },
    { value: "hate", label: "Label hate" },
    { value: "non-hate", label: "Label non-hate" },
  ];
  const detectionFilterOptions: { value: "all" | "hate" | "non-hate"; label: string }[] = [
    { value: "all", label: "Semua deteksi" },
    { value: "hate", label: "Dideteksi hate" },
    { value: "non-hate", label: "Dideteksi non-hate" },
  ];
  const actionFilterOptions: { value: ActionFilter; label: string }[] = [
    { value: "all", label: "Semua tindakan" },
    { value: "warn", label: "Peringatan / mute" },
    { value: "block", label: "Ban / blokir" },
    { value: "allow", label: "Diizinkan" },
  ];
  const timeFilterOptions: { value: TimeFilter; label: string }[] = [
    { value: "all", label: "Semua waktu" },
    { value: "24h", label: "24 jam" },
    { value: "7d", label: "7 hari" },
    { value: "30d", label: "30 hari" },
  ];
  const initialChat = useRef<number | null>(null);

  useEffect(() => {
    ensureTelegramReady();
    const tg = getTelegram();
    const initChat = tg?.initDataUnsafe?.chat;
    const debugChat = getDebugChatId();
    const derived = initChat?.id ?? (debugChat ? Number(debugChat) : null);
    if (derived !== null && !Number.isNaN(derived)) {
      initialChat.current = derived;
      if (derived < 0) {
        setChatId(derived);
      }
    }
  }, []);

  useEffect(() => {
    const loadGroups = async () => {
      try {
        const data = await fetchGroups();
        setGroups(data);
        if ((!chatId || chatId > 0) && data.length) {
          const initial = initialChat.current;
          const fallback =
            initial && initial < 0 && data.some((g) => g.chat_id === initial) ? initial : data[0].chat_id;
          setChatId(fallback);
        }
      } catch (error) {
        if (error instanceof HttpError && (error.status === 401 || error.status === 403)) {
          setAccessDenied(true);
          return;
        }
        notify((error as Error).message ?? "Gagal mengambil daftar grup");
      }
    };
    loadGroups();
  }, [chatId]);

  useEffect(() => {
    if (!chatId) return;
    const canReuseStats =
      stats &&
      stats.chat_id === chatId &&
      statsWindow === topOffenderWindow &&
      stats.window === topOffenderWindow;
    if (canReuseStats) {
      setOffenderLoading(false);
      setOffenderStats(stats);
      return;
    }
    setOffenderStats(null);
    setOffenderLoading(true);
    let cancelled = false;
    const loadOffenderStats = async () => {
      try {
        const data = await fetchStats(chatId, topOffenderWindow);
        if (!cancelled) {
          setOffenderStats(data);
        }
      } catch (error) {
        if (!cancelled) {
          notify((error as Error).message ?? "Gagal memuat data top pelanggar");
        }
      } finally {
        if (!cancelled) {
          setOffenderLoading(false);
        }
      }
    };
    loadOffenderStats();
    return () => {
      cancelled = true;
    };
  }, [chatId, stats, statsWindow, topOffenderWindow]);

  const refreshUserActions = useCallback(async () => {
    if (!chatId) return;
    setUserActionsLoading(true);
    try {
      const data = await fetchUserActions(chatId);
      setUserActions(data);
    } catch (error) {
      notify((error as Error).message ?? "Gagal memuat data tindakan pengguna");
    } finally {
      setUserActionsLoading(false);
    }
  }, [chatId]);

  const loadAll = useCallback(async () => {
    if (!chatId) return;
    try {
      const settingsData = await fetchSettings(chatId);
      setSettings(settingsData);
      setAccessDenied(false);
      const [statsData, activityData, adminsData, mutedData, bannedData] = await Promise.all([
        fetchStats(chatId, statsWindow),
        fetchActivity(chatId, 14),
        fetchAdmins(chatId),
        fetchMembers(chatId, "muted"),
        fetchMembers(chatId, "banned"),
      ]);
      setStats(statsData);
      setActivity(activityData);
      setAdmins(adminsData);
      setMuted(mutedData);
      setBanned(bannedData);
      await refreshUserActions();
    } catch (error) {
      if (error instanceof HttpError && error.status === 403) {
        setAccessDenied(true);
      }
      throw error;
    }
  }, [chatId, statsWindow, refreshUserActions]);

  const loadAllEvents = useCallback(async () => {
    if (!chatId) return [];
    setHistoryLoading(true);
    setReviewLoading(true);
    try {
      const total = await fetchEventCount(chatId);
      const totalPages = Math.max(1, Math.ceil(total / HISTORY_PAGE));
      const aggregated: EventEntry[] = [];
      for (let page = 0; page < totalPages; page++) {
        const offset = page * HISTORY_PAGE;
        const chunk = await fetchEvents(chatId, HISTORY_PAGE, offset);
        aggregated.push(...chunk);
      }
      return aggregated;
    } finally {
      setHistoryLoading(false);
      setReviewLoading(false);
    }
  }, [chatId]);

  const refresh = useCallback(
    async (showSpinner = true) => {
      if (!chatId) return;
      if (showSpinner) setLoading(true);
      try {
        await loadAll();
        const aggregated = await loadAllEvents();
        setAllEvents(aggregated);
        setHistoryPage(0);
        setReviewPage(0);
        setLastUpdated(new Date());
      } catch (error) {
        if (error instanceof HttpError && error.status === 403) {
          return;
        }
        notify((error as Error).message ?? "Gagal memuat data");
      } finally {
        if (showSpinner) setLoading(false);
      }
    },
    [chatId, loadAll, loadAllEvents],
  );

  useEffect(() => {
    if (!chatId) return;
    let cancelled = false;
    const run = async () => {
      if (cancelled) return;
      await refresh();
    };
    run();
    if (!autoRefresh) {
      return () => {
        cancelled = true;
      };
    }
    const interval = setInterval(() => refresh(false), REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [chatId, statsWindow, refresh, autoRefresh]);

  useEffect(() => {
    if (!settings || !chatId) return;
    setRetentionDraft(settings.retention_days);
    const preset = MODE_PRESETS[settings.mode];
    const mode = Math.abs(settings.threshold - preset) > 0.005 ? "custom" : settings.mode;
    setThresholdState((prev) => {
      const next = { ...prev };
      next[chatId] = { value: settings.threshold, mode };
      return next;
    });
  }, [settings, chatId]);

  const handleSettingsUpdate = async (payload: Partial<SettingsResponse>) => {
    if (!chatId) return;
    setPending(true);
    try {
      const updated = await updateSettings(chatId, payload);
      setSettings(updated);
      setRetentionDraft(updated.retention_days);
      notify("Pengaturan tersimpan");
    } catch (error) {
      notify((error as Error).message ?? "Gagal menyimpan pengaturan");
    } finally {
      setPending(false);
    }
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

  const commitRetention = async () => {
    if (!settings || retentionDraft === null || retentionDraft === settings.retention_days) return;
    await handleSettingsUpdate({ retention_days: retentionDraft });
  };

  const currentGroup = useMemo(() => groups.find((g) => g.chat_id === chatId), [groups, chatId]);

  const summary = useMemo(() => {
    if (!stats) return { deleted: 0, warned: 0, blocked: 0, total: 0 };
    return {
      deleted: stats.deleted,
      warned: stats.warned,
      blocked: stats.blocked,
      total: stats.total_events,
    };
  }, [offenderStats]);
  const baseMetricCards = useMemo(
    () => [
      { label: "Total tindakan", value: summary.total, detail: "24 jam terakhir" },
      { label: "Pesan dihapus", value: summary.deleted, detail: "Setiap pelanggaran" },
      { label: "Diperingatkan", value: summary.warned, detail: "Mute otomatis" },
      { label: "Diblokir", value: summary.blocked, detail: "Ban permanen", accent: "success" as const },
    ],
    [summary],
  );
  const moderationStatusCards = useMemo(
    () => [
      {
        label: "Sedang dimute",
        value: muted.length,
        detail: "Mute aktif",
        tone: "muted" as const,
        description: "Pengguna dibatasi sementara hingga jadwal rilis otomatis.",
      },
      {
        label: "Sedang diblokir",
        value: banned.length,
        detail: "Ban aktif",
        tone: "banned" as const,
        description: "Admin dapat melepas blokir kapan pun diperlukan.",
      },
    ],
    [muted.length, banned.length],
  );
  const offenderList = offenderStats?.top_offenders ?? [];

  const lineChartData = useMemo(() => {
    if (!activity || activity.points.length === 0) return null;
    return {
      labels: activity.points.map((p) => dayjs(p.date).format("DD MMM")),
      datasets: [
        {
          label: "Total tindakan",
          data: activity.points.map((p) => p.deleted),
          borderColor: "#818cf8",
          backgroundColor: "rgba(129,140,248,0.2)",
          tension: 0.4,
          fill: true,
        },
        {
          label: "Diperingatkan",
          data: activity.points.map((p) => p.warned),
          borderColor: "#f97316",
          tension: 0.4,
          fill: false,
        },
        {
          label: "Diblokir",
          data: activity.points.map((p) => p.blocked),
          borderColor: "#22c55e",
          tension: 0.4,
          fill: false,
        },
      ],
    };
  }, [activity]);

  const doughnutData = useMemo(() => {
    if (!stats) return null;
    const dataset = [stats.deleted, stats.warned, stats.blocked];
    if (dataset.every((value) => value === 0)) return null;
    return {
      labels: ["Total", "Diperingatkan", "Diblokir"],
      datasets: [
        {
          data: dataset,
          backgroundColor: ["#818cf8", "#f97316", "#22c55e"],
          borderWidth: 0,
        },
      ],
    };
  }, [stats]);

  const topOffenderProgress = useMemo(() => {
    const offenders = offenderStats?.top_offenders ?? [];
    if (!offenders.length) return null;
    const parsed = offenders
      .map((entry) => {
        const match = entry.match(/^(.*)\s+\((\d+)\)$/);
        if (!match) return null;
        return {
          label: match[1].trim(),
          value: Number(match[2]),
        };
      })
      .filter((item): item is { label: string; value: number } => item !== null && Number.isFinite(item.value));
    if (!parsed.length) return null;
    const maxValue = Math.max(...parsed.map((item) => item.value), 1);
    return parsed.map((item) => ({
      ...item,
      percent: Math.max((item.value / maxValue) * 100, 8),
    }));
  }, [offenderStats]);

  const chatLogChartData = useMemo(() => {
    if (!allEvents.length) return null;
    const sortedEvents = [...allEvents].sort(
      (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    );
    return {
      labels: sortedEvents.map((event) => dayjs(event.created_at).format("HH:mm:ss")),
      datasets: [
        {
          label: "Skor kebencian (%)",
          data: sortedEvents.map((event) => Number((event.prob_hate * 100).toFixed(2))),
          borderColor: "#f43f5e",
          backgroundColor: "rgba(244,63,94,0.25)",
          tension: 0.35,
          fill: true,
        },
      ],
    };
  }, [allEvents]);

  const chatLogActionChartData = useMemo(() => {
    if (!allEvents.length) return null;
    const counts: Record<string, number> = {};
    allEvents.forEach((event) => {
      const key = event.action ?? "lainnya";
      counts[key] = (counts[key] ?? 0) + 1;
    });
    const entries = Object.entries(counts);
    if (!entries.length) return null;
    const labels = entries.map(([label]) => label);
    return {
      labels,
      datasets: [
        {
          label: "Jumlah tindakan",
          data: entries.map(([, value]) => value),
          backgroundColor: labels.map((label) => ACTION_COLOR_MAP[label] ?? "#94a3b8"),
          borderRadius: 8,
        },
      ],
    };
  }, [allEvents]);

  const handleRunTest = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!chatId || !testInput.trim()) {
      notify("Masukkan teks uji terlebih dahulu");
      return;
    }
    setPending(true);
    try {
      const result = await runTest(chatId, testInput.trim());
      setTestResult(`Hate: ${formatPercent(result.prob_hate)} • Non-hate: ${formatPercent(result.prob_nonhate)} • Prediksi: ${result.label}`);
    } catch (error) {
      notify((error as Error).message ?? "Uji teks gagal");
    } finally {
      setPending(false);
    }
  };

  const handleExport = async () => {
    if (!chatId) return;
    try {
      if (!filteredHistoryList.length) {
        notify("Tidak ada data yang cocok dengan filter untuk diekspor");
        return;
      }
      const header = [
        "id",
        "chat_id",
        "user_id",
        "username",
        "prob_hate",
        "prob_nonhate",
        "action",
        "reason",
        "created_at",
        "text",
        "manual_label",
        "manual_verified",
      ];
      const escape = (value: unknown) => {
        if (value === null || value === undefined) return "";
        let str = String(value);
        str = str.replace(/\r?\n/g, " ");
        if (/[",\n]/.test(str)) {
          str = `"${str.replace(/"/g, '""')}"`;
        }
        return str;
      };
      const rows = filteredHistoryList.map((event) => [
        event.id,
        event.chat_id,
        event.user_id ?? "",
        event.username ?? "",
        event.prob_hate,
        event.prob_nonhate,
        event.action,
        event.reason ?? "",
        event.created_at,
        event.text ?? "",
        event.manual_label ?? "",
        event.manual_verified ? "true" : "false",
      ]);
      const csv = [header.map(escape).join(","), ...rows.map((row) => row.map(escape).join(","))].join("\n");
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `hate_guard_${chatId}_history.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      notify((error as Error).message ?? "Gagal mengunduh CSV");
    }
  };

  const handleAddAdmin = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!chatId) return;
    const form = event.currentTarget;
    const userId = Number(new FormData(form).get("admin_id"));
    if (!userId) {
      notify("Masukkan user id valid");
      return;
    }
    try {
      await addAdmin(chatId, userId);
      await refresh(false);
      form.reset();
      notify("Admin ditambahkan");
    } catch (error) {
      notify((error as Error).message ?? "Tidak bisa menambah admin");
    }
  };

  const handleRemoveAdmin = async (userId: number) => {
    if (!chatId) return;
    try {
      await removeAdmin(chatId, userId);
      await refresh(false);
      notify("Admin dihapus");
    } catch (error) {
      notify((error as Error).message ?? "Tidak bisa menghapus admin");
    }
  };

  const handleAddMemberStatus = async (event: React.FormEvent<HTMLFormElement>, status: MemberStatus) => {
    event.preventDefault();
    if (!chatId) return;
    const form = event.currentTarget;
    const data = new FormData(form);
    const userId = Number(data.get(`${status}_user`));
    if (!userId) {
      notify("Masukkan user id valid");
      return;
    }
    try {
      await createMemberStatus(chatId, {
        user_id: userId,
        username: (data.get(`${status}_username`) as string) || undefined,
        status,
        duration_minutes: data.get(`${status}_duration`) ? Number(data.get(`${status}_duration`)) : undefined,
        reason: (data.get(`${status}_reason`) as string) || undefined,
      });
      await refresh(false);
      form.reset();
      notify("Status disimpan");
    } catch (error) {
      notify((error as Error).message ?? "Tidak bisa menyimpan status");
    }
  };

  const handleRemoveMember = async (userId: number, status: MemberStatus) => {
    if (!chatId) return;
    try {
      await deleteMemberStatus(chatId, userId, status);
      await refresh(false);
    } catch (error) {
      notify((error as Error).message ?? "Tidak bisa melepaskan status");
    }
  };

  const filterManual = useCallback(
    (event: EventEntry) => {
      switch (manualFilter) {
        case "pending":
          return !event.manual_verified;
        case "verified":
          return event.manual_verified;
        case "hate":
          return event.manual_verified && event.manual_label === "hate";
        case "non-hate":
          return event.manual_verified && event.manual_label === "non-hate";
        default:
          return true;
      }
    },
    [manualFilter],
  );

  const filterDetection = useCallback(
    (event: EventEntry) => {
      const isViolation = VIOLATION_ACTIONS.has(event.action);
      if (detectionFilter === "hate") {
        return isViolation;
      }
      if (detectionFilter === "non-hate") {
        return !isViolation;
      }
      return true;
    },
    [detectionFilter],
  );

  const filterAction = useCallback(
    (event: EventEntry) => {
      switch (actionFilter) {
        case "warn":
          return event.action === "warned" || event.action === "muted";
        case "block":
          return event.action === "banned" || event.action === "blocked";
        case "allow":
          return event.action === "allowed" || event.action === "bypassed_admin";
        default:
          return true;
      }
    },
    [actionFilter],
  );

  const filterTime = useCallback(
    (event: EventEntry) => {
      if (timeFilter === "all") return true;
      const created = dayjs(event.created_at);
      if (!created.isValid()) return true;
      const now = dayjs();
      if (timeFilter === "24h") {
        return created.isAfter(now.subtract(24, "hour"));
      }
      if (timeFilter === "7d") {
        return created.isAfter(now.subtract(7, "day"));
      }
      if (timeFilter === "30d") {
        return created.isAfter(now.subtract(30, "day"));
      }
      return true;
    },
    [timeFilter],
  );

  const filterSearch = useCallback(
    (event: EventEntry) => {
      const q = searchTerm.trim().toLowerCase();
      if (!q) return true;
      const username = (event.username ?? "").toLowerCase();
      const text = (event.text ?? "").toLowerCase();
      const userId = event.user_id ? String(event.user_id) : "";
      return username.includes(q) || text.includes(q) || userId.includes(q);
    },
    [searchTerm],
  );

  const filteredHistoryList = useMemo(
    () =>
      allEvents.filter(
        (event) => filterManual(event) && filterDetection(event) && filterAction(event) && filterTime(event) && filterSearch(event),
      ),
    [allEvents, filterManual, filterDetection, filterAction, filterTime, filterSearch],
  );
  const filteredReviewList = filteredHistoryList;

  useEffect(() => {
    setHistoryPage(0);
    setReviewPage(0);
  }, [chatId, manualFilter, detectionFilter, actionFilter, timeFilter, searchTerm]);

  useEffect(() => {
    const totalPages = Math.max(1, Math.ceil(filteredHistoryList.length / HISTORY_PAGE));
    if (historyPage >= totalPages) {
      setHistoryPage(totalPages - 1);
    }
  }, [filteredHistoryList.length, historyPage]);

  useEffect(() => {
    const totalPages = Math.max(1, Math.ceil(filteredReviewList.length / REVIEW_PAGE));
    if (reviewPage >= totalPages) {
      setReviewPage(totalPages - 1);
    }
  }, [filteredReviewList.length, reviewPage]);

  const historyPageCount = Math.max(1, Math.ceil(filteredHistoryList.length / HISTORY_PAGE));
  const visibleHistoryEvents = useMemo(
    () => filteredHistoryList.slice(historyPage * HISTORY_PAGE, historyPage * HISTORY_PAGE + HISTORY_PAGE),
    [filteredHistoryList, historyPage],
  );
  const reviewPageCount = Math.max(1, Math.ceil(filteredReviewList.length / REVIEW_PAGE));
  const visibleReviewEvents = useMemo(
    () => filteredReviewList.slice(reviewPage * REVIEW_PAGE, reviewPage * REVIEW_PAGE + REVIEW_PAGE),
    [filteredReviewList, reviewPage],
  );

  const hasPrevPage = historyPage > 0;
  const hasNextPage = historyPage + 1 < historyPageCount;
  const hasReviewPrev = reviewPage > 0;
  const hasReviewNext = reviewPage + 1 < reviewPageCount;

  const handleHistoryPrev = () => {
    if (!hasPrevPage || historyLoading) return;
    setHistoryPage((prev) => Math.max(0, prev - 1));
  };

  const handleHistoryNext = () => {
    if (!hasNextPage || historyLoading) return;
    setHistoryPage((prev) => prev + 1);
  };

  const handleReviewPrev = () => {
    if (!hasReviewPrev || reviewLoading) return;
    setReviewPage((prev) => Math.max(0, prev - 1));
  };

  const handleReviewNext = () => {
    if (!hasReviewNext || reviewLoading) return;
    setReviewPage((prev) => prev + 1);
  };

  const handleManualVerification = async (eventId: number, label: "hate" | "non-hate") => {
    if (!chatId) return;
    setVerifyingEvent(eventId);
    try {
      const updated = await verifyEvent(chatId, eventId, label);
      setAllEvents((prev) => prev.map((entry) => (entry.id === updated.id ? { ...entry, ...updated } : entry)));
    } catch (error) {
      notify((error as Error).message ?? "Gagal memverifikasi pesan");
    } finally {
      setVerifyingEvent(null);
    }
  };

  const handleAutoRefreshToggle = () => {
    const next = !autoRefresh;
    setAutoRefresh(next);
    if (next) {
      refresh();
    }
  };

  const handleResetUserAction = async (userId: number, action: "warned" | "muted") => {
    if (!chatId) return;
    setResettingAction({ userId, action });
    try {
      await resetUserAction(chatId, userId, action);
      await refreshUserActions();
      notify(`Counter ${action === "warned" ? "peringatan" : "mute"} direset`);
    } catch (error) {
      notify((error as Error).message ?? "Gagal mereset counter");
    } finally {
      setResettingAction(null);
    }
  };

  const handleModeSelect = async (mode: SettingsResponse["mode"]) => {
    if (!chatId) return;
    const preset = MODE_PRESETS[mode];
    const clamped = Math.min(THRESHOLD_MAX, Math.max(THRESHOLD_MIN, preset ?? (settings?.threshold ?? 0.62)));
    setThresholdState((prev) => ({ ...prev, [chatId]: { value: clamped, mode } }));
    const value = Number(clamped.toFixed(2));
    await handleSettingsUpdate({ mode, threshold: value });
  };

  const handleRefresh = () => {
    setLoading(true);
    setTimeout(() => {
      setLastUpdated(new Date());
      setLoading(false);
    }, 1500);
  };

  if (accessDenied) {
    return (
      <main className="empty-state">
        <h1>Akses Terbatas</h1>
        <p>Dashboard hanya dapat digunakan oleh admin grup. Minta admin menambahkan Anda melalui bot.</p>
      </main>
    );
  }

  if (!chatId) {
    return (
      <main className="empty-state">
        <h1>Pilih Grup Terlebih Dahulu</h1>
        <p>Buka dashboard melalui tombol mini-app di bot Telegram atau sertakan query <code>?chat_id=</code> saat pengembangan.</p>
      </main>
    );
  }

  // Render the current view based on navigation
  const renderView = () => {
    switch (currentView) {
      case "DASHBOARD":
        return <Dashboard 
          chatId={chatId}
          currentGroup={currentGroup}
          settings={settings}
          manualMode={manualMode}
          setManualMode={setManualMode}
          lastUpdated={lastUpdated}
          handleRefresh={handleRefresh}
          refreshing={loading}
          groups={groups}
          setChatId={setChatId}
          handleSettingsUpdate={handleSettingsUpdate}
          baseMetricCards={baseMetricCards}
        />;
      case "STATS":
        return <Stats 
          lineChartData={lineChartData}
          doughnutData={doughnutData}
          statsWindow={statsWindow}
          setStatsWindow={setStatsWindow}
          topOffenderWindow={topOffenderWindow}
          setTopOffenderWindow={setTopOffenderWindow}
          offenderLoading={offenderLoading}
          offenderStats={offenderStats}
          topOffenderProgress={topOffenderProgress}
          offenderList={offenderList}
        />;
      case "LOGS":
        return <Logs 
          allEvents={allEvents}
          filteredHistoryList={filteredHistoryList}
          historyPage={historyPage}
          historyPageCount={historyPageCount}
          hasPrevPage={hasPrevPage}
          hasNextPage={hasNextPage}
          handleHistoryPrev={handleHistoryPrev}
          handleHistoryNext={handleHistoryNext}
          historyLoading={historyLoading}
          manualFilter={manualFilter}
          setManualFilter={setManualFilter}
          manualFilterOptions={manualFilterOptions}
          detectionFilter={detectionFilter}
          setDetectionFilter={setDetectionFilter}
          detectionFilterOptions={detectionFilterOptions}
          actionFilter={actionFilter}
          setActionFilter={setActionFilter}
          actionFilterOptions={actionFilterOptions}
          timeFilter={timeFilter}
          setTimeFilter={setTimeFilter}
          timeFilterOptions={timeFilterOptions}
          searchTerm={searchTerm}
          setSearchTerm={setSearchTerm}
          handleExport={handleExport}
        />;
      case "VERIFY":
        return <Verification 
          visibleReviewEvents={visibleReviewEvents}
          reviewPage={reviewPage}
          reviewPageCount={reviewPageCount}
          hasReviewPrev={hasReviewPrev}
          hasReviewNext={hasReviewNext}
          handleReviewPrev={handleReviewPrev}
          handleReviewNext={handleReviewNext}
          reviewLoading={reviewLoading}
          verifyingEvent={verifyingEvent}
          handleManualVerification={handleManualVerification}
        />;
      case "ADMIN":
        return <AdminPanel 
          admins={admins}
          handleAddAdmin={handleAddAdmin}
          handleRemoveAdmin={handleRemoveAdmin}
          testInput={testInput}
          setTestInput={setTestInput}
          handleRunTest={handleRunTest}
          pending={pending}
          testResult={testResult}
          userActions={userActions}
          userActionsLoading={userActionsLoading}
          refreshUserActions={refreshUserActions}
          resettingAction={resettingAction}
          handleResetUserAction={handleResetUserAction}
        />;
      default:
        return <Dashboard 
          chatId={chatId}
          currentGroup={currentGroup}
          settings={settings}
          manualMode={manualMode}
          setManualMode={setManualMode}
          lastUpdated={lastUpdated}
          handleRefresh={handleRefresh}
          refreshing={loading}
          groups={groups}
          setChatId={setChatId}
          handleSettingsUpdate={handleSettingsUpdate}
          baseMetricCards={baseMetricCards}
        />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-neon-blue/30 selection:text-white">
      {/* Background Ambient Glow */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
          <div className="absolute -top-[10%] -left-[10%] w-[50%] h-[50%] bg-primary-900/20 rounded-full blur-[100px] animate-pulse-slow"></div>
          <div className="absolute top-[20%] right-[0%] w-[40%] h-[40%] bg-blue-900/10 rounded-full blur-[100px]"></div>
          <div className="absolute bottom-[0%] left-[20%] w-[60%] h-[40%] bg-purple-900/10 rounded-full blur-[100px]"></div>
      </div>

      {/* Main Content Area */}
      <div className="relative z-10 max-w-lg mx-auto min-h-screen bg-slate-950/50 shadow-2xl border-x border-slate-900">
        
        {/* Top Bar */}
        <header className="sticky top-0 z-40 glass-panel border-b border-slate-800/50 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center space-x-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-primary-600 to-neon-blue flex items-center justify-center shadow-lg shadow-primary-500/20">
                   <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                   </svg>
                </div>
                <h1 className="font-bold text-lg tracking-tight text-white">HateSpeech<span className="text-primary-400">Mod</span></h1>
            </div>
            <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                <img src="https://picsum.photos/32/32" alt="Admin" className="w-full h-full rounded-full opacity-80 hover:opacity-100 transition-opacity" />
            </div>
        </header>

        {renderView()}
        
      </div>

      <Navigation currentView={currentView} onViewChange={setCurrentView} />
    </div>
  );
}

// Navigation component
function Navigation({ currentView, onViewChange }: { currentView: View; onViewChange: (view: View) => void }) {
  const navItems = [
    { view: "DASHBOARD" as View, icon: LayoutDashboard, label: 'Dash' },
    { view: "STATS" as View, icon: BarChart3, label: 'Stats' },
    { view: "VERIFY" as View, icon: CheckCircle, label: 'Verify' },
    { view: "LOGS" as View, icon: FileText, label: 'Logs' },
    { view: "ADMIN" as View, icon: ShieldAlert, label: 'Admin' },
  ];

  return (
    <div className="fixed bottom-0 left-0 w-full glass-panel border-t border-slate-700 pb-safe pt-2 px-6 z-50">
      <div className="flex justify-between items-center max-w-lg mx-auto h-16">
        {navItems.map((item) => {
          const isActive = currentView === item.view;
          return (
            <button
              key={item.view}
              onClick={() => onViewChange(item.view)}
              className={`flex flex-col items-center justify-center w-12 transition-all duration-300 ${
                isActive ? 'text-neon-blue -translate-y-2' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <div className={`p-2 rounded-full transition-all duration-300 ${isActive ? 'bg-primary-500/20 shadow-[0_0_15px_rgba(99,102,241,0.5)]' : ''}`}>
                <item.icon size={isActive ? 24 : 20} strokeWidth={isActive ? 2.5 : 2} />
              </div>
              <span className={`text-[10px] mt-1 font-medium ${isActive ? 'opacity-100' : 'opacity-0'}`}>
                {item.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// Dashboard component
function Dashboard({ 
  chatId, 
  currentGroup, 
  settings, 
  manualMode, 
  setManualMode, 
  lastUpdated, 
  handleRefresh, 
  refreshing, 
  groups, 
  setChatId, 
  handleSettingsUpdate, 
  baseMetricCards 
}: {
  chatId: number;
  currentGroup?: GroupSummary;
  settings?: SettingsResponse | null;
  manualMode: boolean;
  setManualMode: (mode: boolean) => void;
  lastUpdated: Date | null;
  handleRefresh: () => void;
  refreshing: boolean;
  groups: GroupSummary[];
  setChatId: (id: number) => void;
  handleSettingsUpdate: (payload: Partial<SettingsResponse>) => Promise<void>;
  baseMetricCards: Array<{ label: string; value: number; detail: string; accent?: string }>;
}) {
  const [lastUpdate, setLastUpdate] = useState("Just now");

  useEffect(() => {
    if (lastUpdated) {
      setLastUpdate("Just now");
      const timer = setTimeout(() => {
        setLastUpdate(dayjs(lastUpdated).fromNow());
      }, 60000);
      return () => clearTimeout(timer);
    }
  }, [lastUpdated]);

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
                    <div className={`w-3 h-3 rounded-full ${manualMode ? 'bg-orange-500 animate-pulse' : 'bg-neon-green shadow-[0_0_10px_#0aff68]'}`}></div>
                    <h2 className="text-lg font-bold tracking-wide text-white">
                        {manualMode ? 'Manual Mode' : 'Auto-Protection'}
                    </h2>
                </div>
                <span className="text-xs text-slate-400 font-mono">{lastUpdate}</span>
            </div>
            
            <p className="text-slate-400 text-sm mb-6">
                {manualMode 
                    ? "System is currently flagging content for review only." 
                    : "System is actively filtering high-confidence threats."}
            </p>

            <div className="flex items-center justify-between">
                <label className="flex items-center cursor-pointer">
                    <div className="relative">
                        <input type="checkbox" className="sr-only" checked={manualMode} onChange={() => setManualMode(!manualMode)} />
                        <div className={`block w-14 h-8 rounded-full transition-colors duration-300 ${manualMode ? 'bg-orange-900' : 'bg-slate-700'}`}></div>
                        <div className={`dot absolute left-1 top-1 bg-white w-6 h-6 rounded-full transition-transform duration-300 ${manualMode ? 'translate-x-6 bg-orange-400' : ''}`}></div>
                    </div>
                    <span className="ml-3 text-sm font-medium text-slate-300">
                        {manualMode ? 'Realtime OFF' : 'Realtime ON'}
                    </span>
                </label>
                
                <button 
                    onClick={handleRefresh}
                    className="flex items-center space-x-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-full text-xs font-bold border border-slate-600 transition-all active:scale-95"
                >
                    <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
                    <span>Refresh Data</span>
                </button>
            </div>
        </div>
      </div>

      {/* Monitored Groups */}
      <div>
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 pl-1">Monitored Channels</h3>
        <div className="flex overflow-x-auto space-x-3 pb-2 scrollbar-hide">
            {groups.map(g => (
                <div key={g.chat_id} className={`flex-shrink-0 w-64 p-4 rounded-2xl border transition-all duration-300 ${g.chat_id === chatId ? 'bg-gradient-to-br from-slate-800 to-slate-900 border-primary-500/30' : 'bg-slate-900 border-slate-800 opacity-60'}`}>
                    <div className="flex justify-between items-start mb-2">
                        <h4 className="font-bold text-white truncate pr-2">{g.title}</h4>
                        <div className={`w-2 h-2 rounded-full ${g.chat_id === chatId ? 'bg-neon-green' : 'bg-slate-500'}`}></div>
                    </div>
                    <p className="text-xs text-slate-500 font-mono mb-3">ID: {g.chat_id}</p>
                    <div className="flex items-center justify-between">
                        <span className={`text-xs ${g.chat_id === chatId ? 'text-neon-green' : 'text-slate-500'}`}>
                            {g.chat_id === chatId ? 'Active' : 'Paused'}
                        </span>
                        {/* Toggle Switch Mini */}
                         <div className={`w-8 h-4 rounded-full relative ${g.chat_id === chatId ? 'bg-primary-600' : 'bg-slate-700'}`}>
                             <div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform ${g.chat_id === chatId ? 'translate-x-4' : ''}`}></div>
                         </div>
                    </div>
                </div>
            ))}
        </div>
      </div>

      {/* Summary Stats Grid */}
      <div>
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 pl-1">Moderation Impact</h3>
        <div className="grid grid-cols-2 gap-3">
            <StatCard 
                title="Total Actions" 
                value={baseMetricCards[0].value.toString()} 
                subtitle="Last 24 hours" 
                icon={Shield} 
                color="text-blue-400" 
                bg="from-blue-500/10 to-transparent"
            />
            <StatCard 
                title="Deleted" 
                value={baseMetricCards[1].value.toString()} 
                subtitle="High confidence" 
                icon={UserX} 
                color="text-red-400" 
                bg="from-red-500/10 to-transparent"
            />
            <StatCard 
                title="Auto-Muted" 
                value={baseMetricCards[2].value.toString()} 
                subtitle="Temporary" 
                icon={VolumeX} 
                color="text-purple-400" 
                bg="from-purple-500/10 to-transparent"
            />
             <StatCard 
                title="Warnings" 
                value={baseMetricCards[3].value.toString()} 
                subtitle="First offense" 
                icon={AlertTriangle} 
                color="text-orange-400" 
                bg="from-orange-500/10 to-transparent"
            />
        </div>
      </div>
      
      {/* Live Feed Teaser */}
      <div className="glass-panel rounded-2xl p-4 border-l-4 border-neon-blue">
          <div className="flex justify-between items-center mb-2">
              <h4 className="font-bold text-white">Live Activity</h4>
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neon-blue opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-neon-blue"></span>
              </span>
          </div>
          <div className="space-y-3">
              <div className="text-sm text-slate-300 border-b border-slate-800 pb-2">
                  <span className="text-red-400 font-mono font-bold text-xs">[BLOCKED]</span> User <span className="text-white">@BadActor99</span> posted high toxicity content.
              </div>
              <div className="text-sm text-slate-300">
                  <span className="text-orange-400 font-mono font-bold text-xs">[FLAGGED]</span> User <span className="text-white">@AnonUser</span> flagged for manual review (Score: 65%).
              </div>
          </div>
      </div>
    </div>
  );
}

// Stats component
function Stats({ 
  lineChartData, 
  doughnutData, 
  statsWindow, 
  setStatsWindow, 
  topOffenderWindow, 
  setTopOffenderWindow, 
  offenderLoading, 
  offenderStats, 
  topOffenderProgress, 
  offenderList 
}: {
  lineChartData: any;
  doughnutData: any;
  statsWindow: "24h" | "7d";
  setStatsWindow: (window: "24h" | "7d") => void;
  topOffenderWindow: "24h" | "7d";
  setTopOffenderWindow: (window: "24h" | "7d") => void;
  offenderLoading: boolean;
  offenderStats: StatsResponse | null;
  topOffenderProgress: any;
  offenderList: string[];
}) {
  const pieData = doughnutData ? [
    { name: 'Warned', value: doughnutData.datasets[0].data[1], color: '#f59e0b' }, // Amber
    { name: 'Muted', value: doughnutData.datasets[0].data[2], color: '#6366f1' },  // Indigo
    { name: 'Blocked', value: doughnutData.datasets[0].data[0], color: '#ef4444' }, // Red
  ] : [];

  return (
    <div className="p-4 space-y-6 pb-24 animate-fade-in">
        
        {/* Date Filter */}
        <div className="flex bg-slate-800 p-1 rounded-xl">
            {['Today', '7 Days', '30 Days', 'All Time'].map((p, i) => (
                <button key={p} className={`flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ${i === 1 ? 'bg-slate-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}>
                    {p}
                </button>
            ))}
        </div>

        {/* Activity Chart */}
        <div className="glass-panel p-5 rounded-3xl border border-slate-700/50">
            <h3 className="text-white font-bold mb-1">Moderation Activity</h3>
            <div className="flex space-x-4 text-xs mb-4">
                <span className="flex items-center text-blue-400"><span className="w-2 h-2 rounded-full bg-blue-400 mr-1"></span> Total</span>
                <span className="flex items-center text-orange-400"><span className="w-2 h-2 rounded-full bg-orange-400 mr-1"></span> Warned</span>
                <span className="flex items-center text-red-400"><span className="w-2 h-2 rounded-full bg-red-400 mr-1"></span> Blocked</span>
            </div>
            <div className="h-48 w-full">
                {lineChartData && <Line data={lineChartData} options={{
                  plugins: { legend: { display: false } },
                  scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                  }
                }} />}
            </div>
        </div>

        {/* Composition Chart */}
        <div className="glass-panel p-5 rounded-3xl border border-slate-700/50 flex items-center justify-between">
            <div className="w-1/2">
                <h3 className="text-white font-bold mb-4">Action<br/>Composition</h3>
                <div className="space-y-2">
                    {pieData.map((entry) => (
                        <div key={entry.name} className="flex items-center justify-between text-xs">
                             <div className="flex items-center">
                                 <div className="w-2 h-2 rounded-full mr-2" style={{ backgroundColor: entry.color }}></div>
                                 <span className="text-slate-300">{entry.name}</span>
                             </div>
                             <span className="font-mono text-white">{entry.value}%</span>
                        </div>
                    ))}
                </div>
            </div>
            <div className="w-1/2 h-32 relative">
                {doughnutData && <Doughnut data={doughnutData} options={{
                  plugins: { legend: { display: false } },
                  cutout: '60%'
                }} />}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                   <span className="text-xs font-bold text-slate-500">Total</span> 
                </div>
            </div>
        </div>

        {/* Top Offenders List */}
        <div>
            <div className="flex justify-between items-center mb-3 px-1">
                <h3 className="text-white font-bold">Top Offenders</h3>
                <span className="text-xs text-slate-500">Last 7 Days</span>
            </div>
            <div className="space-y-3">
                {offenderLoading ? (
                  <div className="glass-panel p-4 rounded-2xl text-center text-slate-400">
                    Loading...
                  </div>
                ) : offenderList.length === 0 ? (
                  <div className="glass-panel p-4 rounded-2xl text-center text-slate-400">
                    No data available
                  </div>
                ) : (
                  offenderList.map((user, idx) => (
                      <div key={idx} className="glass-panel p-3 rounded-2xl flex items-center space-x-3">
                          <img src={`https://picsum.photos/40/40?random=${idx}`} alt="avatar" className="w-10 h-10 rounded-full border border-slate-600" />
                          <div className="flex-1 min-w-0">
                              <h4 className="text-sm font-bold text-white truncate">{user}</h4>
                              <div className="text-[10px] text-slate-400 font-mono">ID: {idx}</div>
                          </div>
                          <div className="text-right">
                               <div className="text-sm font-bold text-neon-blue">{idx + 1} acts</div>
                               <div className="w-24 h-1.5 bg-slate-800 rounded-full mt-1 overflow-hidden">
                                   <div 
                                      className="h-full bg-gradient-to-r from-blue-500 to-purple-500" 
                                      style={{ width: `${(1 - idx * 0.2) * 100}%` }}
                                   ></div>
                               </div>
                          </div>
                      </div>
                  ))
                )}
            </div>
        </div>
    </div>
  );
}

// Logs component
function Logs({ 
  allEvents, 
  filteredHistoryList, 
  historyPage, 
  historyPageCount, 
  hasPrevPage, 
  hasNextPage, 
  handleHistoryPrev, 
  handleHistoryNext, 
  historyLoading, 
  manualFilter, 
  setManualFilter, 
  manualFilterOptions, 
  detectionFilter, 
  setDetectionFilter, 
  detectionFilterOptions, 
  actionFilter, 
  setActionFilter, 
  actionFilterOptions, 
  timeFilter, 
  setTimeFilter, 
  timeFilterOptions, 
  searchTerm, 
  setSearchTerm, 
  handleExport 
}: {
  allEvents: EventEntry[];
  filteredHistoryList: EventEntry[];
  historyPage: number;
  historyPageCount: number;
  hasPrevPage: boolean;
  hasNextPage: boolean;
  handleHistoryPrev: () => void;
  handleHistoryNext: () => void;
  historyLoading: boolean;
  manualFilter: string;
  setManualFilter: (filter: string) => void;
  manualFilterOptions: { value: string; label: string }[];
  detectionFilter: string;
  setDetectionFilter: (filter: string) => void;
  detectionFilterOptions: { value: string; label: string }[];
  actionFilter: string;
  setActionFilter: (filter: string) => void;
  actionFilterOptions: { value: string; label: string }[];
  timeFilter: string;
  setTimeFilter: (filter: string) => void;
  timeFilterOptions: { value: string; label: string }[];
  searchTerm: string;
  setSearchTerm: (term: string) => void;
  handleExport: () => void;
}) {
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

  const visibleHistoryEvents = useMemo(
    () => filteredHistoryList.slice(historyPage * 5, historyPage * 5 + 5),
    [filteredHistoryList, historyPage],
  );

  return (
    <div className="p-4 h-full flex flex-col pb-24 animate-fade-in">
        <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-white">History & Logs</h2>
            <button className="p-2 rounded-full bg-slate-800 text-slate-400 hover:text-white" onClick={handleExport}>
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
                {manualFilterOptions.map(filter => (
                    <button key={filter.value} className="px-3 py-1.5 rounded-full border border-slate-700 text-xs text-slate-300 hover:bg-slate-800 whitespace-nowrap" onClick={() => setManualFilter(filter.value)}>
                        {filter.label}
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
            {visibleHistoryEvents.length === 0 && (
              <div className="glass-panel p-4 rounded-xl text-center text-slate-400">
                No data available
              </div>
            )}
            {visibleHistoryEvents.map((event) => (
                <div key={event.id} className="glass-panel p-3 rounded-xl grid grid-cols-12 gap-2 items-center hover:bg-slate-800/50 transition-colors">
                    <div className="col-span-3">
                        <div className="text-xs text-slate-400 font-mono">{dayjs(event.created_at).format("HH:mm")}</div>
                        <div className="text-sm font-bold text-white truncate">{event.username ?? event.user_id}</div>
                    </div>
                    <div className="col-span-2 flex justify-center">
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${getActionColor(event.action)}`}>
                            {event.action}
                        </span>
                    </div>
                    <div className="col-span-2 text-center">
                        <span className={`font-mono font-bold text-sm ${getScoreColor(event.prob_hate * 100)}`}>
                            {Math.round(event.prob_hate * 100)}%
                        </span>
                    </div>
                    <div className="col-span-4">
                        <p className="text-xs text-slate-300 truncate opacity-80">{event.text ?? "No content"}</p>
                    </div>
                    <div className="col-span-1 flex justify-end">
                        <MoreHorizontal size={16} className="text-slate-500" />
                    </div>
                </div>
            ))}
        </div>

        {/* Pagination */}
        <div className="flex justify-between items-center mt-4">
            <button 
              className={`px-4 py-2 rounded-lg text-xs font-medium ${hasPrevPage ? 'bg-slate-800 text-white hover:bg-slate-700' : 'bg-slate-900 text-slate-600 cursor-not-allowed'}`}
              onClick={handleHistoryPrev}
              disabled={!hasPrevPage || historyLoading}
            >
                Previous
            </button>
            <span className="text-xs text-slate-400">
                Page {historyPage + 1} of {historyPageCount}
            </span>
            <button 
              className={`px-4 py-2 rounded-lg text-xs font-medium ${hasNextPage ? 'bg-slate-800 text-white hover:bg-slate-700' : 'bg-slate-900 text-slate-600 cursor-not-allowed'}`}
              onClick={handleHistoryNext}
              disabled={!hasNextPage || historyLoading}
            >
                Next
            </button>
        </div>
    </div>
  );
}

// Verification component
function Verification({ 
  visibleReviewEvents, 
  reviewPage, 
  reviewPageCount, 
  hasReviewPrev, 
  hasReviewNext, 
  handleReviewPrev, 
  handleReviewNext, 
  reviewLoading, 
  verifyingEvent, 
  handleManualVerification 
}: {
  visibleReviewEvents: EventEntry[];
  reviewPage: number;
  reviewPageCount: number;
  hasReviewPrev: boolean;
  hasReviewNext: boolean;
  handleReviewPrev: () => void;
  handleReviewNext: () => void;
  reviewLoading: boolean;
  verifyingEvent: number | null;
  handleManualVerification: (eventId: number, label: "hate" | "non-hate") => Promise<void>;
}) {
  const [items, setItems] = useState(visibleReviewEvents);

  useEffect(() => {
    setItems(visibleReviewEvents);
  }, [visibleReviewEvents]);

  const handleDecision = async (id: number, isHate: boolean) => {
    // Simulate removing card
    const element = document.getElementById(`card-${id}`);
    if (element) {
        element.style.transform = isHate ? 'translateX(100vw)' : 'translateX(-100vw)';
        element.style.opacity = '0';
    }
    await handleManualVerification(id, isHate ? "hate" : "non-hate");
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
                    {items.map((event) => (
                        <div 
                            key={event.id} 
                            id={`card-${event.id}`}
                            className="glass-panel p-5 rounded-3xl border border-slate-700/50 shadow-lg transition-all duration-300 transform"
                        >
                            <div className="flex items-center space-x-3 mb-4">
                                <div className="w-10 h-10 rounded-full bg-slate-700 overflow-hidden">
                                    <img src={`https://picsum.photos/40/40?random=${event.user_id}`} alt="Avatar" />
                                </div>
                                <div>
                                    <h4 className="text-white font-bold">{event.username ?? event.user_id}</h4>
                                    <span className="text-xs text-slate-400">{dayjs(event.created_at).fromNow()}</span>
                                </div>
                                <div className="ml-auto text-right">
                                    <div className="text-[10px] text-slate-500 uppercase tracking-wide">Confidence</div>
                                    <div className={`text-xl font-mono font-bold ${event.prob_hate > 0.8 ? 'text-red-500' : 'text-orange-500'}`}>
                                        {Math.round(event.prob_hate * 100)}%
                                    </div>
                                </div>
                            </div>

                            <div className="bg-slate-900/50 p-4 rounded-xl mb-6 border border-slate-800">
                                <p className="text-slate-200 text-sm leading-relaxed">
                                    "{event.text ?? "No content available"}"
                                </p>
                            </div>

                            <div className="flex space-x-3">
                                <button 
                                    onClick={() => handleDecision(event.id, false)}
                                    className="flex-1 py-3 rounded-xl bg-green-500/10 text-green-500 border border-green-500/30 hover:bg-green-500 hover:text-white transition-all font-bold flex items-center justify-center"
                                >
                                    <Check size={18} className="mr-2" /> Mark Safe
                                </button>
                                <button 
                                    onClick={() => handleDecision(event.id, true)}
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

        {/* Pagination */}
        <div className="flex justify-between items-center mt-4">
            <button 
              className={`px-4 py-2 rounded-lg text-xs font-medium ${hasReviewPrev ? 'bg-slate-800 text-white hover:bg-slate-700' : 'bg-slate-900 text-slate-600 cursor-not-allowed'}`}
              onClick={handleReviewPrev}
              disabled={!hasReviewPrev || reviewLoading}
            >
                Previous
            </button>
            <span className="text-xs text-slate-400">
                Page {reviewPage + 1} of {reviewPageCount}
            </span>
            <button 
              className={`px-4 py-2 rounded-lg text-xs font-medium ${hasReviewNext ? 'bg-slate-800 text-white hover:bg-slate-700' : 'bg-slate-900 text-slate-600 cursor-not-allowed'}`}
              onClick={handleReviewNext}
              disabled={!hasReviewNext || reviewLoading}
            >
                Next
            </button>
        </div>
    </div>
  );
}

// Admin Panel component
function AdminPanel({ 
  admins, 
  handleAddAdmin, 
  handleRemoveAdmin, 
  testInput, 
  setTestInput, 
  handleRunTest, 
  pending, 
  testResult, 
  userActions, 
  userActionsLoading, 
  refreshUserActions, 
  resettingAction, 
  handleResetUserAction 
}: {
  admins: AdminEntry[];
  handleAddAdmin: (e: React.FormEvent<HTMLFormElement>) => Promise<void>;
  handleRemoveAdmin: (userId: number) => Promise<void>;
  testInput: string;
  setTestInput: (input: string) => void;
  handleRunTest: (e: React.FormEvent<HTMLFormElement>) => Promise<void>;
  pending: boolean;
  testResult: string;
  userActions: UserActionSummary[];
  userActionsLoading: boolean;
  refreshUserActions: () => Promise<void>;
  resettingAction: { userId: number; action: "warned" | "muted" } | null;
  handleResetUserAction: (userId: number, action: "warned" | "muted") => Promise<void>;
}) {
  const [analysisResult, setAnalysisResult] = useState<{ category: string; score: number; reasoning: string } | null>(null);

  const handleTestModel = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    await handleRunTest(e);
    
    // Parse the test result to extract category, score, and reasoning
    if (testResult) {
      const parts = testResult.split(' • ');
      if (parts.length >= 3) {
        const prediction = parts[2].replace('Prediksi: ', '');
        const scoreMatch = parts[0].match(/Hate: ([\d.]+)/);
        const score = scoreMatch ? parseFloat(scoreMatch[1]) * 100 : 0;
        
        setAnalysisResult({
          category: prediction,
          score,
          reasoning: "Model analysis based on text patterns and content"
        });
      }
    }
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
                        <div className="font-bold text-white text-sm">Admin {admin.id}</div>
                        <div className="text-xs text-slate-500">ID: {admin.user_id}</div>
                    </div>
                    <button className="text-xs bg-slate-800 hover:bg-red-900/30 hover:text-red-400 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors" onClick={() => handleRemoveAdmin(admin.user_id)}>
                        Remove
                    </button>
                </div>
            ))}
            <div className="p-2">
                <form onSubmit={handleAddAdmin}>
                    <div className="flex space-x-2">
                        <input name="admin_id" placeholder="User ID" className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white" />
                        <button type="submit" className="px-3 py-2 bg-primary-600/20 text-primary-400 border border-primary-600/30 rounded-lg text-sm font-bold hover:bg-primary-600 hover:text-white transition-all">
                            Add
                        </button>
                    </div>
                </form>
            </div>
        </div>

        {/* AI Model Tester */}
        <div className="glass-panel rounded-3xl p-5 border border-neon-purple/30 relative overflow-hidden">
             <div className="absolute -top-10 -right-10 w-32 h-32 bg-neon-purple/20 blur-3xl rounded-full pointer-events-none"></div>
             
             <div className="flex items-center space-x-2 mb-4">
                 <Zap className="text-neon-purple" size={20} />
                 <h3 className="font-bold text-white">Moderation AI Tester</h3>
             </div>

             <form onSubmit={handleTestModel}>
                <textarea
                    value={testInput}
                    onChange={(e) => setTestInput(e.target.value)}
                    placeholder="Enter text to test the moderation model..."
                    className="w-full bg-slate-900/80 border border-slate-700 rounded-xl p-3 text-sm text-white focus:border-neon-purple focus:ring-1 focus:ring-neon-purple transition-all min-h-[100px] mb-4"
                />

                <button 
                    type="submit"
                    disabled={pending}
                    className={`w-full py-3 rounded-xl font-bold text-white flex items-center justify-center transition-all ${pending ? 'bg-slate-700 cursor-not-allowed' : 'bg-gradient-to-r from-primary-600 to-purple-600 shadow-[0_0_15px_rgba(124,58,237,0.4)] hover:shadow-[0_0_25px_rgba(124,58,237,0.6)]'}`}
                >
                     {pending ? <RefreshCcw className="animate-spin mr-2" size={18} /> : 'Analyze Text'}
                </button>
             </form>

             {analysisResult && (
                 <div className="mt-4 p-4 bg-slate-900 rounded-xl border border-slate-700 animate-fade-in">
                     <div className="flex justify-between items-center mb-2">
                         <span className="text-xs text-slate-400 uppercase tracking-wide">Category</span>
                         <span className={`text-xs font-bold px-2 py-0.5 rounded ${analysisResult.score > 70 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                             {analysisResult.category}
                         </span>
                     </div>
                     <div className="flex justify-between items-center mb-3">
                         <span className="text-xs text-slate-400 uppercase tracking-wide">Confidence Score</span>
                         <div className="flex items-center">
                             <div className="h-2 w-24 bg-slate-800 rounded-full mr-2 overflow-hidden">
                                 <div 
                                    className={`h-full ${analysisResult.score > 70 ? 'bg-red-500' : 'bg-green-500'}`} 
                                    style={{width: `${analysisResult.score}%`}}
                                 ></div>
                             </div>
                             <span className="text-sm font-mono font-bold text-white">{analysisResult.score.toFixed(1)}%</span>
                         </div>
                     </div>
                     <p className="text-xs text-slate-300 italic border-t border-slate-800 pt-2 mt-2">
                         "{analysisResult.reasoning}"
                     </p>
                 </div>
             )}
        </div>

        {/* Muted Users Management */}
        <div>
            <h3 className="text-white font-bold mb-3">Active Penalties</h3>
            
            {userActionsLoading ? (
              <div className="glass-panel p-4 rounded-2xl text-center text-slate-400">
                Loading user actions...
              </div>
            ) : userActions.length === 0 ? (
              <div className="glass-panel p-4 rounded-2xl text-center text-slate-400">
                No active penalties
              </div>
            ) : (
              userActions.map(user => (
                  <div key={user.user_id} className="glass-panel p-4 rounded-2xl mb-3 border border-slate-800">
                      <div className="flex justify-between items-start mb-4">
                          <div>
                              <div className="font-bold text-white">{user.username ?? user.user_id}</div>
                              <div className="text-xs text-slate-500">ID: {user.user_id}</div>
                          </div>
                          <div className="flex space-x-1">
                              <div className="px-2 py-1 bg-orange-500/20 text-orange-400 rounded text-[10px] font-bold border border-orange-500/20">
                                  {user.warnings_today} Warns
                              </div>
                               <div className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-[10px] font-bold border border-purple-500/20">
                                  {user.mutes_total} Mutes
                              </div>
                          </div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-3">
                          <button 
                            className="py-2 bg-slate-800 hover:bg-slate-700 text-red-400 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center"
                            onClick={() => handleResetUserAction(user.user_id, "warned")}
                            disabled={resettingAction?.userId === user.user_id && resettingAction?.action === "warned"}
                          >
                              <AlertOctagon size={14} className="mr-1.5" /> Reset Warning
                          </button>
                          <button 
                            className="py-2 bg-slate-800 hover:bg-slate-700 text-purple-400 text-xs font-bold rounded-lg border border-slate-700 flex items-center justify-center"
                            onClick={() => handleResetUserAction(user.user_id, "muted")}
                            disabled={resettingAction?.userId === user.user_id && resettingAction?.action === "muted"}
                          >
                              <ShieldCheck size={14} className="mr-1.5" /> Unmute
                          </button>
                      </div>
                  </div>
              ))
            )}
        </div>
    </div>
  );
}

// StatCard component
function StatCard({title, value, subtitle, icon: Icon, color, bg}: {
  title: string, 
  value: string, 
  subtitle: string, 
  icon: any, 
  color: string, 
  bg: string
}) {
  return (
    <div className={`p-4 rounded-2xl bg-gradient-to-br ${bg} border border-slate-800 relative overflow-hidden`}>
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
}