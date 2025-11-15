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
const MODE_PRESETS: Record<SettingsResponse["mode"], number> = {
  precision: 0.75,
  balanced: 0.62,
  recall: 0.5,
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
  const [thresholdPreview, setThresholdPreview] = useState<number | null>(null);
  const [retentionDraft, setRetentionDraft] = useState<number | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyPage, setHistoryPage] = useState(0);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewPage, setReviewPage] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [accessDenied, setAccessDenied] = useState(false);
  const [verifyingEvent, setVerifyingEvent] = useState<number | null>(null);
  const [userActions, setUserActions] = useState<UserActionSummary[]>([]);
  const [userActionsLoading, setUserActionsLoading] = useState(false);
  const [resettingAction, setResettingAction] = useState<{ userId: number; action: "warned" | "muted" } | null>(null);
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
    if (typeof window !== "undefined" && window.innerWidth < 900) {
      setSidebarOpen(false);
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
  };

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible[0]) {
          const sectionId = visible[0].target.getAttribute("data-section") as SectionKey | null;
          if (sectionId) {
            setActiveSection(sectionId);
          }
        }
      },
      { threshold: 0.35, rootMargin: "-25% 0px -25% 0px" },
    );
    Object.values(sectionRefs).forEach((ref) => {
      if (ref.current) {
        observer.observe(ref.current);
      }
    });
    return () => observer.disconnect();
  }, [sectionRefs]);

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
    if (!settings) return;
    setThresholdPreview(settings.threshold);
    setRetentionDraft(settings.retention_days);
  }, [settings]);

  const handleSettingsUpdate = async (payload: Partial<SettingsResponse>) => {
    if (!chatId) return;
    setPending(true);
    try {
      const updated = await updateSettings(chatId, payload);
      setSettings(updated);
      setThresholdPreview(updated.threshold);
      setRetentionDraft(updated.retention_days);
      notify("Pengaturan tersimpan");
    } catch (error) {
      notify((error as Error).message ?? "Gagal menyimpan pengaturan");
    } finally {
      setPending(false);
    }
  };

  const commitThreshold = async () => {
    if (!settings || thresholdPreview === null) return;
    if (Math.abs(thresholdPreview - settings.threshold) < 0.001) return;
    await handleSettingsUpdate({ threshold: Number(thresholdPreview.toFixed(2)) });
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
    const preset = MODE_PRESETS[mode];
    const clamped = Math.min(THRESHOLD_MAX, Math.max(THRESHOLD_MIN, preset ?? (settings?.threshold ?? 0.62)));
    setThresholdPreview(clamped);
    await handleSettingsUpdate({ mode, threshold: Number(clamped.toFixed(2)) });
  };

  const navigation: { key: SectionKey; label: string }[] = [
    { key: "overview", label: "Ikhtisar" },
    { key: "analytics", label: "Analitik" },
    { key: "history", label: "Riwayat" },
    { key: "admin", label: "Admin" },
    { key: "members", label: "Anggota" },
  ];
  const toggleSidebar = () => setSidebarOpen((prev) => !prev);

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

  return (
    <div className="app-shell">
      <div className={clsx("sidebar-overlay", sidebarOpen && "visible")} onClick={() => sidebarOpen && toggleSidebar()} />
      <aside className={clsx("sidebar panel", sidebarOpen ? "open" : "collapsed")}>
        <div className="brand compact">
          <div className="logo-dot" />
          <div>
            <p className="eyebrow">Hate Guard</p>
            <h2>Control Tower</h2>
          </div>
        </div>
        <ViewportDebug />  {/* debug panel */}
        <div className="nav-list">
          {navigation.map((item) => (
            <button
              key={item.key}
              type="button"
              className={clsx("nav-item", activeSection === item.key && "active")}
              onClick={() => handleSectionClick(item.key)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </aside>
      <div className="content">
        <nav className="top-nav panel">
          <button className={clsx("sidebar-toggle", sidebarOpen && "open")} type="button" onClick={toggleSidebar} aria-label="Toggle menu">
            <span className="toggle-line" />
            <span className="toggle-line" />
            <span className="toggle-line" />
          </button>
          <div className="brand">
            <div className="logo-dot">
              <span>CM</span>
            </div>
            <div>
              <p className="eyebrow">Content Moderator</p>
              <h1>Control Tower</h1>
            </div>
          </div>
          <div className="status-group">
            <div className="live-status">
              <span className={clsx("status-indicator", autoRefresh ? (loading ? "syncing" : "live") : "paused")} />
              <div>
                <strong>{autoRefresh ? (loading ? "Sinkronisasi" : "Realtime aktif") : "Mode manual"}</strong>
                <p>{lastUpdated ? `Last update ${dayjs(lastUpdated).fromNow()}` : "Menunggu data..."}</p>
              </div>
            </div>
            <div className="nav-actions">
              <button className={clsx("toggle", autoRefresh ? "on" : "off")} type="button" onClick={handleAutoRefreshToggle}>
                <span className="bullet" />
                {autoRefresh ? "Realtime ON" : "Realtime OFF"}
              </button>
              <button className="btn ghost" onClick={() => refresh()}>
                Refresh Data
              </button>
              <button className="btn primary" onClick={handleExport}>
                Unduh CSV
              </button>
            </div>
          </div>
        </nav>
        {!sidebarOpen && <div className="mobile-nav-spacer" />}

        <section ref={overviewRef} data-section="overview" className="section-stack">
          <div className="panel group-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Grup diawasi</p>
                <h2>{currentGroup?.title ?? "Grup"}</h2>
                <p className="muted">ID: {chatId}</p>
              </div>
              <button
                className={clsx("toggle", settings?.enabled ? "on" : "off")}
                onClick={() => handleSettingsUpdate({ enabled: !(settings?.enabled ?? false) })}
              >
                <span className="bullet" />
                {settings?.enabled ? "Moderasi aktif" : "Moderasi nonaktif"}
              </button>
            </div>
            <div className="group-body">
              <div className="group-list">
                {groups.length === 0 && <p className="muted">Belum ada grup terdaftar.</p>}
                {groups.map((group) => (
                  <button
                    key={group.chat_id}
                    type="button"
                    className={clsx("group-chip", group.chat_id === chatId && "active")}
                    onClick={() => setChatId(group.chat_id)}
                  >
                    <div>
                      <strong>{group.title ?? group.chat_id}</strong>
                      <p>{group.group_type ?? "unknown"}</p>
                    </div>
                    <small>{dayjs(group.last_active).fromNow()}</small>
                  </button>
                ))}
              </div>
              <div className="group-controls">
                <div className="control-row">
                  <span>Mode</span>
                  <div className="tab-group">
                    {(["precision", "balanced", "recall"] as SettingsResponse["mode"][]).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        className={clsx("tab", settings?.mode === mode && "active")}
                        onClick={() => handleModeSelect(mode)}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="control-row threshold">
                  <div>
                    <span>Ambang moderasi</span>
                    <strong>{formatPercent(thresholdPreview ?? settings?.threshold ?? 0)}</strong>
                  </div>
                  <input
                    type="range"
                    min={THRESHOLD_MIN}
                    max={THRESHOLD_MAX}
                    step={0.01}
                    value={thresholdPreview ?? settings?.threshold ?? 0.62}
                    onChange={(event) => setThresholdPreview(Number(event.target.value))}
                    onMouseUp={commitThreshold}
                    onPointerUp={commitThreshold}
                    onTouchEnd={commitThreshold}
                    onBlur={commitThreshold}
                  />
                </div>
                <div className="control-row retention">
                  <label htmlFor="retention">
                    Retensi log (hari)
                    <input
                      id="retention"
                      type="number"
                      min={1}
                      max={90}
                      value={retentionDraft ?? ""}
                      onChange={(event) => setRetentionDraft(event.target.value ? Number(event.target.value) : null)}
                      onBlur={commitRetention}
                    />
                  </label>
                  <p className="muted">Data lama akan dibersihkan otomatis.</p>
                </div>
              </div>
            </div>
          </div>
          <div className="panel metrics-panel">
            {baseMetricCards.map((card) => (
              <div key={card.label} className="metric-card">
                <p>{card.label}</p>
                <h3 className={clsx(card.accent)}>{card.value}</h3>
                <small>{card.detail}</small>
              </div>
            ))}
            <div className="moderation-status">
              {moderationStatusCards.map((card) => (
                <div key={card.label} className={clsx("status-card", card.tone)}>
                  <div className="status-top">
                    <p>{card.label}</p>
                    <span>{card.detail}</span>
                  </div>
                  <div className="status-value">{card.value}</div>
                  <p className="status-description">{card.description}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section ref={analyticsRef} data-section="analytics" className="panel charts-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Timeline moderasi</p>
              <h3>Aktivitas {statsWindow}</h3>
            </div>
            <div className="tab-group">
              {(["24h", "7d"] as const).map((window) => (
                <button
                  key={window}
                  type="button"
                  className={clsx("tab", statsWindow === window && "active")}
                  onClick={() => setStatsWindow(window)}
                >
                  {window}
                </button>
              ))}
            </div>
          </div>
          {lineChartData ? (
            <Line
              data={lineChartData}
              options={{
                plugins: { legend: { position: "bottom" } },
                scales: { y: { beginAtZero: true } },
              }}
            />
          ) : (
            <p className="muted">Belum ada data historis.</p>
          )}
          <div className="chart-side">
            <div className="panel-subcard">
              <h4>Komposisi tindakan</h4>
              {doughnutData ? (
                <Doughnut
                  data={doughnutData}
                  options={{
                    plugins: { legend: { position: "bottom" } },
                  }}
                />
              ) : (
                <p className="muted">Belum ada tindakan yang tercatat.</p>
              )}
            </div>
            <div className="panel-subcard offenders">
              <div className="offender-header">
                <h4>Top pelanggar</h4>
                <div className="chip-row compact">
                  {TOP_OFFENDER_WINDOWS.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      className={clsx("chip", topOffenderWindow === option.value && "active")}
                      onClick={() => setTopOffenderWindow(option.value)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              {offenderLoading ? (
                <p className="muted">Memuat data top pelanggar…</p>
              ) : !offenderStats ? (
                <p className="muted">Data top pelanggar belum tersedia.</p>
              ) : offenderList.length === 0 ? (
                <p className="muted">Belum ada pelaku dominan.</p>
              ) : (
                <div className="offender-bars">
                  {topOffenderProgress?.map((item) => (
                    <div key={item.label} className="offender-row">
                      <div className="offender-label">
                        <strong>{item.label}</strong>
                        <span>{item.value}x</span>
                      </div>
                      <div className="offender-meter">
                        <div className="offender-meter-fill" style={{ width: `${item.percent}%` }} />
                      </div>
                    </div>
                  ))}
                  <ul>
                    {offenderList.map((entry) => (
                      <li key={entry}>{entry}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </section>

        <section ref={historyRef} data-section="history" className="panel history-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Riwayat realtime</p>
              <h3>Tindakan + chat log terbaru</h3>
            </div>
            <div className="history-meta">
              {autoRefresh && <p className="muted">Merefresh otomatis tiap {REFRESH_MS / 1000} detik.</p>}
              <p className="muted">5 entri per halaman.</p>
            </div>
          </div>
          <div className="history-filters">
            <div>
              <p className="eyebrow">Filter verifikasi</p>
              <div className="chip-row">
                {manualFilterOptions.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={clsx("chip", manualFilter === option.value && "active")}
                    onClick={() => setManualFilter(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <p className="eyebrow">Deteksi bot</p>
              <div className="chip-row">
                {detectionFilterOptions.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={clsx("chip", detectionFilter === option.value && "active")}
                    onClick={() => setDetectionFilter(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
          {(chatLogChartData || chatLogActionChartData) && (
            <div className="history-charts">
              {chatLogChartData && (
                <div className="history-chart">
                  <h4>Tren skor hate (riwayat terbaru)</h4>
                  <Line
                    data={chatLogChartData}
                    options={{
                      plugins: { legend: { display: false } },
                      maintainAspectRatio: false,
                      scales: {
                        y: {
                          beginAtZero: true,
                          suggestedMax: 100,
                          ticks: {
                            callback: (value) => `${value}%`,
                            color: "#e2e8f0",
                          },
                          grid: { color: "rgba(255,255,255,0.07)" },
                        },
                        x: {
                          grid: { display: false },
                          ticks: { color: "#cbd5f5" },
                        },
                      },
                    }}
                  />
                </div>
              )}
              {chatLogActionChartData && (
                <div className="history-chart action">
                  <h4>Distribusi tindakan (riwayat terbaru)</h4>
                  <Bar
                    data={chatLogActionChartData}
                    options={{
                      plugins: { legend: { display: false } },
                      maintainAspectRatio: false,
                      scales: {
                        y: {
                          beginAtZero: true,
                          ticks: { color: "#e2e8f0", precision: 0, stepSize: 1 },
                          grid: { color: "rgba(255,255,255,0.07)" },
                        },
                        x: {
                          ticks: { color: "#cbd5f5" },
                          grid: { display: false },
                        },
                      },
                    }}
                  />
                </div>
              )}
            </div>
          )}
          <div className="table-wrapper">
            <table>
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
                    <td colSpan={6} className="muted">
                      Tidak ada data yang cocok dengan filter.
                    </td>
                  </tr>
                )}
                {visibleHistoryEvents.map((event) => {
                  const meta = ACTION_META[event.action] ?? { label: event.action, tone: "muted" };
                  return (
                    <tr key={event.id}>
                      <td>
                        <strong>{event.username ?? event.user_id ?? "unknown"}</strong>
                        <p className="muted">{event.text ?? "-"}</p>
                      </td>
                      <td>{formatPercent(event.prob_hate)}</td>
                      <td>{event.reason ?? "-"}</td>
                      <td>
                        <span className={clsx("badge", meta.tone)}>{meta.label}</span>
                      </td>
                      <td>
                        {event.manual_verified ? (
                          <span className={clsx("badge", event.manual_label === "hate" ? "danger" : "success")}>
                            {event.manual_label === "hate" ? "Hate Speech" : "Non-hate"}
                          </span>
                        ) : (
                          <span className="muted">Belum</span>
                        )}
                      </td>
                      <td>{dayjs(event.created_at).format("DD MMM YYYY HH:mm:ss")}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
        </div>
        <div className="user-actions-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Kontrol pelanggar</p>
              <h3>Peringatan & mute pengguna</h3>
            </div>
            <button className="btn ghost small" type="button" onClick={refreshUserActions} disabled={userActionsLoading}>
              {userActionsLoading ? "Memuat..." : "Refresh"}
            </button>
          </div>
          <div className="user-actions-grid">
            {userActions.length === 0 && !userActionsLoading && (
              <p className="muted">Belum ada data pengguna untuk ditampilkan.</p>
            )}
            {userActionsLoading && <p className="muted">Memuat data pelanggar…</p>}
            {userActions.map((entry) => (
              <article key={entry.user_id} className="user-action-card">
                <header>
                  <div>
                    <strong>{entry.username ?? entry.user_id}</strong>
                    <p className="muted">ID {entry.user_id}</p>
                  </div>
                </header>
                <div className="user-stats">
                  <div className="metric-pill warning">
                    <span className="value">{entry.warnings_today}</span>
                    <span className="label">peringatan</span>
                  </div>
                  <div className="metric-pill mute">
                    <span className="value">{entry.mutes_total}</span>
                    <span className="label">mute total</span>
                  </div>
                </div>
                <div className="last-activity">
                  <span>Warning: {entry.last_warning ? dayjs(entry.last_warning).fromNow() : "-"}</span>
                  <span>Mute: {entry.last_mute ? dayjs(entry.last_mute).fromNow() : "-"}</span>
                </div>
                <div className="action-buttons horizontal">
                  <button
                    type="button"
                    className="btn outline small"
                    disabled={resettingAction?.userId === entry.user_id && resettingAction?.action === "warned"}
                    onClick={() => handleResetUserAction(entry.user_id, "warned")}
                  >
                    Reset warning
                  </button>
                  <button
                    type="button"
                    className="btn outline small"
                    disabled={resettingAction?.userId === entry.user_id && resettingAction?.action === "muted"}
                    onClick={() => handleResetUserAction(entry.user_id, "muted")}
                  >
                    Reset mute
                  </button>
                </div>
              </article>
            ))}
          </div>
        </div>
        <div className="history-actions">
          <button className="btn ghost" type="button" disabled={!hasPrevPage || historyLoading} onClick={handleHistoryPrev}>
            Sebelumnya
          </button>
          <span className="muted">
            Halaman {Math.min(historyPage + 1, historyPageCount)} dari {historyPageCount}
          </span>
          <button className="btn ghost" type="button" disabled={!hasNextPage || historyLoading} onClick={handleHistoryNext}>
            Berikutnya
          </button>
        </div>
        <div className="verification-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Verifikasi manual</p>
              <h3>Labeli pesan secara cepat</h3>
            </div>
            <p className="muted">Pilih label hate/non-hate untuk meningkatkan akurasi model.</p>
          </div>
          <div className="verification-grid">
            {!reviewLoading && visibleReviewEvents.length === 0 && <p className="muted">Tidak ada pesan yang cocok dengan filter.</p>}
            {reviewLoading && <p className="muted">Memuat data verifikasi…</p>}
            {visibleReviewEvents.map((event) => (
              <article key={`verify-${event.id}`} className={clsx("verification-card", event.manual_verified && "verified")}>
                <header>
                  <div>
                    <strong>{event.username ?? event.user_id ?? "unknown"}</strong>
                    <p className="muted">{dayjs(event.created_at).format("DD MMM YYYY HH:mm")}</p>
                  </div>
                  <span className="badge muted">{formatPercent(event.prob_hate)}</span>
                </header>
                <p className="text">
                  {event.text ? event.text.slice(0, 140) : "Tidak ada pesan tersimpan."}
                  {event.text && event.text.length > 140 ? "…" : ""}
                </p>
                {event.manual_verified ? (
                  <div className="verified-pill">
                    Ditandai {event.manual_label === "hate" ? "Hate Speech" : "Non-hate"} ·{" "}
                    {event.manual_verified_at ? dayjs(event.manual_verified_at).fromNow() : "baru saja"}
                  </div>
                ) : (
                  <div className="verify-actions">
                    <button
                      type="button"
                      className="btn danger"
                      disabled={verifyingEvent === event.id}
                      onClick={() => handleManualVerification(event.id, "hate")}
                    >
                      Tandai hatespeech
                    </button>
                    <button
                      type="button"
                      className="btn success"
                      disabled={verifyingEvent === event.id}
                      onClick={() => handleManualVerification(event.id, "non-hate")}
                    >
                      Tandai non-hate
                    </button>
                  </div>
                )}
              </article>
            ))}
          </div>
          <div className="history-actions">
            <button className="btn ghost" type="button" disabled={!hasReviewPrev || reviewLoading} onClick={handleReviewPrev}>
              Sebelumnya
            </button>
            <span className="muted">
              Halaman {Math.min(reviewPage + 1, reviewPageCount)} dari {reviewPageCount}
            </span>
            <button className="btn ghost" type="button" disabled={!hasReviewNext || reviewLoading} onClick={handleReviewNext}>
              Berikutnya
            </button>
          </div>
        </div>
      </section>

        <section ref={adminRef} data-section="admin" className="panel admin-panel">
          <div>
            <div className="panel-header">
              <div>
                <p className="eyebrow">Kelola admin</p>
                <h3>Akses dashboard</h3>
              </div>
            </div>
            <form onSubmit={handleAddAdmin} className="form-grid">
              <input name="admin_id" placeholder="User ID" className="input" />
              <button className="btn primary" type="submit">
                Tambah
              </button>
            </form>
            <ul className="list">
              {admins.length === 0 && <li className="list-empty">Belum ada admin tambahan.</li>}
              {admins.map((admin) => (
                <li key={admin.id} className="list-item">
                  <div>
                    <strong>{admin.user_id}</strong>
                    <p className="muted">{dayjs(admin.added_at).fromNow()}</p>
                  </div>
                  <button className="btn ghost" type="button" onClick={() => handleRemoveAdmin(admin.user_id)}>
                    Hapus
                  </button>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="panel-header">
              <div>
                <p className="eyebrow">Simulasi teks</p>
                <h3>Uji model</h3>
              </div>
            </div>
            <form onSubmit={handleRunTest} className="form-grid">
              <textarea className="input" placeholder="Masukkan pesan" value={testInput} onChange={(event) => setTestInput(event.target.value)} />
              <button className="btn primary" type="submit" disabled={pending}>
                Jalankan
              </button>
            </form>
            {testResult && <p className="muted">{testResult}</p>}
          </div>
        </section>

        <section ref={membersRef} data-section="members" className="panel members-panel">
          <ModerationForm title="Pengguna dimute" status="muted" data={muted} onSubmit={handleAddMemberStatus} onRelease={handleRemoveMember} />
          <ModerationForm title="Pengguna diblokir" status="banned" data={banned} onSubmit={handleAddMemberStatus} onRelease={handleRemoveMember} />
        </section>

        {loading && (
          <div className="loading-overlay">
            <div className="spinner" />
            <p>Memuat data dashboard…</p>
          </div>
        )}
      </div>
    </div>
  );
}

function ModerationForm({ title, status, data, onSubmit, onRelease }: ModerationFormProps) {
  return (
    <div className="moderation-card">
      <div className="panel-header">
        <div>
          <p className="eyebrow">{status === "muted" ? "Mute otomatis" : "Ban otomatis"}</p>
          <h3>{title}</h3>
        </div>
      </div>
      <form onSubmit={(event) => onSubmit(event, status)} className="form-grid">
        <input name={`${status}_user`} className="input" placeholder="User ID" />
        <input name={`${status}_username`} className="input" placeholder="username (opsional)" />
        {status === "muted" && <input name="muted_duration" className="input" placeholder="Durasi (menit)" />}
        <input name={`${status}_reason`} className="input" placeholder="Alasan" />
        <button className="btn ghost" type="submit">
          {status === "muted" ? "Tambah mute" : "Ban pengguna"}
        </button>
      </form>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Pengguna</th>
              <th>Alasan</th>
              <th>Sejak</th>
              <th>Berakhir</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data.length === 0 && (
              <tr>
                <td colSpan={5} className="muted">
                  Tidak ada data.
                </td>
              </tr>
            )}
            {data.map((item) => (
              <tr key={item.id}>
                <td>
                  <strong>{item.username ?? item.user_id}</strong>
                  <p className="muted">{item.user_id}</p>
                </td>
                <td>{item.reason ?? "otomatis"}</td>
                <td>{dayjs(item.created_at).format("DD MMM HH:mm")}</td>
                <td>{item.expires_at ? dayjs(item.expires_at).format("DD MMM HH:mm") : status === "muted" ? "Sampai dibuka" : "Permanent"}</td>
                <td>
                  <button className="btn danger" type="button" onClick={() => onRelease(item.user_id, status)}>
                    Lepas
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ViewportDebug() {
  const [size, setSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const update = () => setSize({ w: window.innerWidth, h: window.innerHeight });
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  return (
    <div
      style={{
        position: "fixed",
        bottom: 10,
        right: 10,
        padding: "6px 10px",
        borderRadius: 6,
        background: "rgba(0,0,0,0.7)",
        color: "white",
        fontSize: 12,
        zIndex: 99999,
      }}
    >
      {size.w} x {size.h}
    </div>
  );
}
