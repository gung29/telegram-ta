import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import {
  addAdmin,
  createMemberStatus,
  deleteMemberStatus,
  removeAdmin,
  exportCsv,
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

dayjs.extend(relativeTime);
ChartJS.register(LineElement, ArcElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend, BarElement);

const REFRESH_MS = 1000;
const HISTORY_PAGE = 5;
const REVIEW_PAGE = 5;
const THRESHOLD_MIN = 0.2;
const THRESHOLD_MAX = 0.95;
type ModeSelection = SettingsResponse["mode"] | "custom";

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

type ModerationFormProps = {
  title: string;
  status: MemberStatus;
  data: MemberModeration[];
  onSubmit: (event: React.FormEvent<HTMLFormElement>, status: MemberStatus) => void;
  onRelease: (userId: number, status: MemberStatus) => void;
};

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
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [accessDenied, setAccessDenied] = useState(false);
  const [verifyingEvent, setVerifyingEvent] = useState<number | null>(null);
  const [userActions, setUserActions] = useState<UserActionSummary[]>([]);
  const [userActionsLoading, setUserActionsLoading] = useState(false);
  const [resettingAction, setResettingAction] = useState<{ userId: number; action: "warned" | "muted" } | null>(null);
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
  const initialChat = useRef<number | null>(null);
  const overviewRef = useRef<HTMLElement | null>(null);
  const analyticsRef = useRef<HTMLElement | null>(null);
  const historyRef = useRef<HTMLElement | null>(null);
  const adminRef = useRef<HTMLElement | null>(null);
  const membersRef = useRef<HTMLElement | null>(null);
  type SectionKey = "overview" | "analytics" | "history" | "admin" | "members";
  const sectionRefs = useMemo(
    () => ({
      overview: overviewRef,
      analytics: analyticsRef,
      history: historyRef,
      admin: adminRef,
      members: membersRef,
    }),
    [],
  );
  const [activeSection, setActiveSection] = useState<SectionKey>("overview");

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

  const handleSectionClick = (key: SectionKey) => {
    setActiveSection(key);
    const target = sectionRefs[key].current;
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    setSidebarOpen(false);
  };

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
          borderColor: "#667eea",
          backgroundColor: "rgba(102, 126, 234, 0.2)",
          tension: 0.4,
          fill: true,
        },
        {
          label: "Diperingatkan",
          data: activity.points.map((p) => p.warned),
          borderColor: "#f093fb",
          tension: 0.4,
          fill: false,
        },
        {
          label: "Diblokir",
          data: activity.points.map((p) => p.blocked),
          borderColor: "#4facfe",
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
          backgroundColor: ["#667eea", "#f093fb", "#4facfe"],
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
          borderColor: "#f093fb",
          backgroundColor: "rgba(240, 147, 251, 0.25)",
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
      const blob = await exportCsv(chatId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `hate_guard_${chatId}.csv`;
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

  const filteredHistoryList = useMemo(
    () => allEvents.filter((event) => filterManual(event) && filterDetection(event)),
    [allEvents, filterManual, filterDetection],
  );
  const filteredReviewList = filteredHistoryList;

  useEffect(() => {
    setHistoryPage(0);
    setReviewPage(0);
  }, [chatId, manualFilter, detectionFilter]);

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

  const navigation: { key: SectionKey; label: string; icon: string }[] = [
    { key: "overview", label: "Ikhtisar", icon: "📊" },
    { key: "analytics", label: "Analitik", icon: "📈" },
    { key: "history", label: "Riwayat", icon: "📜" },
    { key: "admin", label: "Admin", icon: "👥" },
    { key: "members", label: "Anggota", icon: "👤" },
  ];

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  if (accessDenied) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">🔒</div>
        <h2 className="empty-state-title">Akses Terbatas</h2>
        <p className="empty-state-description">
          Dashboard hanya dapat digunakan oleh admin grup. Minta admin menambahkan Anda melalui bot.
        </p>
      </div>
    );
  }

  if (!chatId) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📱</div>
        <h2 className="empty-state-title">Pilih Grup Terlebih Dahulu</h2>
        <p className="empty-state-description">
          Buka dashboard melalui tombol mini-app di bot Telegram atau sertakan query <code>?chat_id=</code> saat pengembangan.
        </p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <aside className={clsx("sidebar", sidebarOpen && "open")}>
        <div className="brand">
          <div className="logo">CM</div>
          <div className="brand-text">
            <h1>Content Moderator</h1>
            <p>Control Tower</p>
          </div>
        </div>
        
        <nav>
          <ul className="nav-list">
            {navigation.map((item) => (
              <li key={item.key} className="nav-item">
                <a
                  href="#"
                  className={clsx("nav-link", activeSection === item.key && "active")}
                  onClick={(e) => {
                    e.preventDefault();
                    handleSectionClick(item.key);
                  }}
                >
                  <span className="nav-icon">{item.icon}</span>
                  <span>{item.label}</span>
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </aside>

      <div className={clsx("sidebar-overlay", sidebarOpen && "visible")} onClick={() => setSidebarOpen(false)} />

      <main className="content">
        <nav className="top-nav">
          <div className="nav-left">
            <button className="sidebar-toggle" onClick={toggleSidebar}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18" />
              </svg>
            </button>
            <div className="status-indicator">
              <span className={clsx("status-dot", autoRefresh && "live")}></span>
              <span>{autoRefresh ? "Realtime aktif" : "Mode manual"}</span>
            </div>
          </div>
          
          <div className="nav-actions">
            <button className="btn btn-ghost btn-sm" onClick={() => refresh()}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
              </svg>
              <span className="hidden sm:inline">Refresh</span>
            </button>
            <button className="btn btn-primary btn-sm" onClick={handleExport}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
              </svg>
              <span className="hidden sm:inline">CSV</span>
            </button>
            <label className="toggle">
              <input
                type="checkbox"
                className="toggle-input"
                checked={autoRefresh}
                onChange={handleAutoRefreshToggle}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>
        </nav>

        <section ref={overviewRef} data-section="overview" className="panel fade-in">
          <div className="panel-header">
            <div>
              <h2 className="panel-title">{currentGroup?.title ?? "Grup"}</h2>
              <p className="panel-subtitle">ID: {chatId}</p>
            </div>
            <button
              className={clsx("btn btn-sm", settings?.enabled ? "btn-success" : "btn-secondary")}
              onClick={() => handleSettingsUpdate({ enabled: !(settings?.enabled ?? false) })}
            >
              {settings?.enabled ? "Aktif" : "Nonaktif"}
            </button>
          </div>

          <div className="grid grid-auto">
            {baseMetricCards.map((card, index) => (
              <div key={card.label} className="card" style={{ animationDelay: `${index * 0.1}s` }}>
                <div className="card-body">
                  <div className="metric-value">{card.value}</div>
                  <div className="metric-label">{card.label}</div>
                  <small className="text-muted">{card.detail}</small>
                </div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2" style={{ marginTop: "1rem" }}>
            {moderationStatusCards.map((card, index) => (
              <div key={card.label} className="card" style={{ animationDelay: `${index * 0.1}s` }}>
                <div className="card-header">
                  <h3 className="card-title">{card.label}</h3>
                  <span className="badge badge-secondary">{card.detail}</span>
                </div>
                <div className="card-body">
                  <div className="metric-value">{card.value}</div>
                  <p className="text-sm">{card.description}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section ref={analyticsRef} data-section="analytics" className="panel fade-in">
          <div className="panel-header">
            <div>
              <h2 className="panel-title">Analitik</h2>
              <p className="panel-subtitle">Timeline moderasi</p>
            </div>
            <div className="tabs">
              {(["24h", "7d"] as const).map((window) => (
                <button
                  key={window}
                  className={clsx("tab", statsWindow === window && "active")}
                  onClick={() => setStatsWindow(window)}
                >
                  {window}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3">
            <div className="lg:col-span-2">
              {lineChartData ? (
                <div className="chart-container">
                  <Line data={lineChartData} options={{ responsive: true, maintainAspectRatio: false }} />
                </div>
              ) : (
                <div className="empty-state">
                  <p>Belum ada data historis.</p>
                </div>
              )}
            </div>
            
            <div>
              {doughnutData ? (
                <div className="chart-container">
                  <Doughnut data={doughnutData} options={{ responsive: true, maintainAspectRatio: false }} />
                </div>
              ) : (
                <div className="empty-state">
                  <p>Belum ada tindakan yang tercatat.</p>
                </div>
              )}
            </div>
          </div>
        </section>

        <section ref={historyRef} data-section="history" className="panel fade-in">
          <div className="panel-header">
            <div>
              <h2 className="panel-title">Riwayat</h2>
              <p className="panel-subtitle">Tindakan + chat log terbaru</p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2" style={{ marginBottom: "1rem" }}>
            <div>
              <h3 className="text-lg font-semibold mb-2">Filter Verifikasi</h3>
              <div className="flex flex-wrap gap-2">
                {manualFilterOptions.map((option) => (
                  <button
                    key={option.value}
                    className={clsx("btn btn-sm", manualFilter === option.value ? "btn-primary" : "btn-ghost")}
                    onClick={() => setManualFilter(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-2">Deteksi Bot</h3>
              <div className="flex flex-wrap gap-2">
                {detectionFilterOptions.map((option) => (
                  <button
                    key={option.value}
                    className={clsx("btn btn-sm", detectionFilter === option.value ? "btn-primary" : "btn-ghost")}
                    onClick={() => setDetectionFilter(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Pengguna</th>
                  <th>Skor</th>
                  <th>Alasan</th>
                  <th>Tindakan</th>
                  <th>Verifikasi</th>
                  <th>Waktu</th>
                </tr>
              </thead>
              <tbody>
                {visibleHistoryEvents.length === 0 && (
                  <tr>
                    <td colSpan={6} className="text-center text-muted">
                      Tidak ada data yang cocok dengan filter.
                    </td>
                  </tr>
                )}
                {visibleHistoryEvents.map((event) => {
                  const meta = ACTION_META[event.action] ?? { label: event.action, tone: "muted" };
                  return (
                    <tr key={event.id}>
                      <td>
                        <div>
                          <strong>{event.username ?? event.user_id ?? "unknown"}</strong>
                          <p className="text-sm text-muted">{event.text ? (event.text.length > 30 ? `${event.text.slice(0, 30)}...` : event.text) : "-"}</p>
                        </div>
                      </td>
                      <td>{formatPercent(event.prob_hate)}</td>
                      <td>{event.reason ?? "-"}</td>
                      <td>
                        <span className={clsx("badge", `badge-${meta.tone}`)}>{meta.label}</span>
                      </td>
                      <td>
                        {event.manual_verified ? (
                          <span className={clsx("badge", event.manual_label === "hate" ? "badge-danger" : "badge-success")}>
                            {event.manual_label === "hate" ? "Hate" : "Non-hate"}
                          </span>
                        ) : (
                          <span className="badge badge-secondary">Belum</span>
                        )}
                      </td>
                      <td>{dayjs(event.created_at).format("DD MMM HH:mm")}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="flex justify-between items-center mt-4 flex-wrap gap-2">
            <button className="btn btn-ghost btn-sm" disabled={!hasPrevPage || historyLoading} onClick={handleHistoryPrev}>
              Sebelumnya
            </button>
            <span className="text-muted text-sm">
              Halaman {Math.min(historyPage + 1, historyPageCount)} dari {historyPageCount}
            </span>
            <button className="btn btn-ghost btn-sm" disabled={!hasNextPage || historyLoading} onClick={handleHistoryNext}>
              Berikutnya
            </button>
          </div>
        </section>

        <section ref={adminRef} data-section="admin" className="panel fade-in">
          <div className="panel-header">
            <div>
              <h2 className="panel-title">Admin</h2>
              <p className="panel-subtitle">Kelola admin dan uji model</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2">
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Tambah Admin</h3>
              </div>
              <form onSubmit={handleAddAdmin} className="space-y-3">
                <input name="admin_id" placeholder="User ID" className="form-input" />
                <button type="submit" className="btn btn-primary w-full">
                  Tambah
                </button>
              </form>
              
              <div className="mt-4">
                <h4 className="font-semibold mb-2">Daftar Admin</h4>
                <div className="space-y-2">
                  {admins.length === 0 && (
                    <p className="text-muted text-sm">Belum ada admin tambahan.</p>
                  )}
                  {admins.map((admin) => (
                    <div key={admin.id} className="flex justify-between items-center p-2 bg-glass rounded-lg">
                      <div>
                        <strong className="text-sm">{admin.user_id}</strong>
                        <p className="text-xs text-muted">{dayjs(admin.added_at).fromNow()}</p>
                      </div>
                      <button className="btn btn-sm btn-danger" onClick={() => handleRemoveAdmin(admin.user_id)}>
                        Hapus
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Uji Model</h3>
              </div>
              <form onSubmit={handleRunTest}>
                <textarea
                  className="form-input form-textarea"
                  placeholder="Masukkan pesan"
                  value={testInput}
                  onChange={(event) => setTestInput(event.target.value)}
                  rows={3}
                />
                <button type="submit" className="btn btn-primary w-full mt-2" disabled={pending}>
                  {pending ? <span className="loading"></span> : "Jalankan"}
                </button>
              </form>
              {testResult && (
                <div className="mt-4 p-3 bg-glass rounded-lg">
                  <p className="text-sm">{testResult}</p>
                </div>
              )}
            </div>
          </div>
        </section>

        <section ref={membersRef} data-section="members" className="panel fade-in">
          <div className="panel-header">
            <div>
              <h2 className="panel-title">Anggota</h2>
              <p className="panel-subtitle">Kelola muted dan banned users</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2">
            <ModerationForm title="Pengguna dimute" status="muted" data={muted} onSubmit={handleAddMemberStatus} onRelease={handleRemoveMember} />
            <ModerationForm title="Pengguna diblokir" status="banned" data={banned} onSubmit={handleAddMemberStatus} onRelease={handleRemoveMember} />
          </div>
        </section>

        {loading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Memuat data dashboard...</p>
          </div>
        )}
      </main>
    </div>
  );
}

function ModerationForm({ title, status, data, onSubmit, onRelease }: ModerationFormProps) {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">{title}</h3>
      </div>
      
      <form onSubmit={(event) => onSubmit(event, status)} className="space-y-3">
        <div className="form-row">
          <input name={`${status}_user`} className="form-input" placeholder="User ID" />
          <input name={`${status}_username`} className="form-input" placeholder="Username (opsional)" />
        </div>
        {status === "muted" && (
          <input name="muted_duration" className="form-input" placeholder="Durasi (menit)" />
        )}
        <input name={`${status}_reason`} className="form-input" placeholder="Alasan" />
        <button type="submit" className="btn btn-primary w-full">
          {status === "muted" ? "Tambah Mute" : "Ban Pengguna"}
        </button>
      </form>

      <div className="mt-4">
        <h4 className="font-semibold mb-2">Daftar {title}</h4>
        <div className="table-container">
          <table className="table">
            <thead>
              <tr>
                <th>Pengguna</th>
                <th>Alasan</th>
                <th>Sejak</th>
                <th>Berakhir</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {data.length === 0 && (
                <tr>
                  <td colSpan={5} className="text-center text-muted">
                    Tidak ada data.
                  </td>
                </tr>
              )}
              {data.slice(0, 3).map((item) => (
                <tr key={item.id}>
                  <td>
                    <strong className="text-sm">{item.username ?? item.user_id}</strong>
                    <p className="text-xs text-muted">{item.user_id}</p>
                  </td>
                  <td className="text-sm">{item.reason ?? "otomatis"}</td>
                  <td className="text-sm">{dayjs(item.created_at).format("DD MMM HH:mm")}</td>
                  <td className="text-sm">{item.expires_at ? dayjs(item.expires_at).format("DD MMM HH:mm") : status === "muted" ? "Sampai dibuka" : "Permanent"}</td>
                  <td>
                    <button className="btn btn-sm btn-danger" onClick={() => onRelease(item.user_id, status)}>
                      Lepas
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}