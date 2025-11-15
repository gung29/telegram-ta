import TelegramBot from "node-telegram-bot-api";
import logger from "./logger";
import { botConfig } from "./config";
import {
  addAdmin,
  fetchActionCount,
  fetchAdmins,
  fetchMemberModerations,
  fetchSettings,
  fetchStats,
  logEvent,
  predictText,
  removeAdmin,
  removeMemberModeration,
  runAdminTest,
  sendSettings,
  syncGroup,
  upsertMemberModeration,
} from "./apiClient";
import { TTLCache } from "./cache";
import { EventPayload, MemberModeration, MemberStatus, SettingsResponse } from "./types";

const settingsCache = new TTLCache<SettingsResponse>(60_000);
const backendAdminCache = new TTLCache<number[]>(120_000);
const telegramAdminCache = new TTLCache<number[]>(300_000);
const offenseCache = new TTLCache<number>(15 * 60 * 1000);
const dailyOffenseMap: Map<string, { count: number; day: string }> = new Map();
const muteCountCache = new TTLCache<number>(30 * 24 * 60 * 60 * 1000);
const manualStatusCache = new Map<number, Map<number, MemberStatus>>();

const isInvalidUserError = (error: unknown) =>
  error instanceof Error && /invalid user identifier/i.test(error.message);

const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

const persistEvent = async (payload: EventPayload) => {
  try {
    await logEvent(payload);
  } catch (error) {
    logger.error({ err: error, payload }, "Failed to persist moderation event");
  }
};

const bold = (text: string) => `<b>${text}</b>`;

const getBackendAdmins = async (chatId: number): Promise<number[]> => {
  const cached = backendAdminCache.get(chatId);
  if (cached) return cached;
  const admins = await fetchAdmins(chatId);
  const ids = admins.map((admin) => admin.user_id);
  backendAdminCache.set(chatId, ids);
  return ids;
};

const getTelegramAdmins = async (bot: TelegramBot, chatId: number): Promise<number[]> => {
  const cached = telegramAdminCache.get(chatId);
  if (cached) return cached;
  try {
    const admins = await bot.getChatAdministrators(chatId);
    const ids = admins.map((admin) => admin.user.id);
    telegramAdminCache.set(chatId, ids);
    return ids;
  } catch (error) {
    logger.error({ err: error, chatId }, "Failed to fetch telegram admins");
    return [];
  }
};

const requireAdmin = async (bot: TelegramBot, chatId: number, userId: number | undefined): Promise<boolean> => {
  if (!userId) return false;
  if (botConfig.adminIds.includes(userId)) return true;
  const backend = await getBackendAdmins(chatId);
  if (backend.includes(userId)) return true;
  const telegramAdmins = await getTelegramAdmins(bot, chatId);
  return telegramAdmins.includes(userId);
};

const getSettings = async (chatId: number) => {
  const cached = settingsCache.get(chatId);
  if (cached) return cached;
  const settings = await fetchSettings(chatId);
  settingsCache.set(chatId, settings);
  return settings;
};

const setSettings = async (chatId: number, payload: Partial<SettingsResponse>) => {
  const updated = await sendSettings(chatId, payload);
  settingsCache.set(chatId, updated);
  return updated;
};

const isHttpsUrl = (url: string | undefined): boolean => {
  if (!url) return false;
  return /^https:\/\/.+/i.test(url);
};

const notifyAdminOnly = async (bot: TelegramBot, msg: TelegramBot.Message) => {
  const target = msg.chat?.type === "private" ? msg.chat.id : msg.from?.id;
  if (target) {
    await bot.sendMessage(target, "Perintah ini hanya untuk admin.");
  }
};

const trackedChats = new Set<number>();
const groupSyncCache = new TTLCache<boolean>(60 * 60 * 1000);

const mutePermissions = {
  can_send_messages: false,
  can_send_media_messages: false,
  can_send_polls: false,
  can_send_other_messages: false,
  can_add_web_page_previews: false,
  can_change_info: false,
  can_invite_users: false,
  can_pin_messages: false,
} as TelegramBot.ChatPermissions;

const defaultPermissions = {
  can_send_messages: true,
  can_send_media_messages: true,
  can_send_polls: true,
  can_send_other_messages: true,
  can_add_web_page_previews: true,
  can_change_info: false,
  can_invite_users: true,
  can_pin_messages: false,
} as TelegramBot.ChatPermissions;

const LOCAL_TIMEZONE = "Asia/Singapore";

const offenseKey = (chatId: number, userId: number) => `${chatId}:${userId}`;

const getLocalDayKey = () =>
  new Intl.DateTimeFormat("en-CA", {
    timeZone: LOCAL_TIMEZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());

const incrementOffense = (chatId: number, userId: number): number => {
  const key = offenseKey(chatId, userId);
  const current = offenseCache.get(key) ?? 0;
  const next = current + 1;
  offenseCache.set(key, next);
  return next;
};

const ensureDailyRecord = async (chatId: number, userId: number) => {
  const key = offenseKey(chatId, userId);
  const today = getLocalDayKey();
  let record = dailyOffenseMap.get(key);
  if (!record || record.day !== today) {
    const remote = await fetchActionCount(chatId, userId, "warned", "day");
    record = { count: remote, day: today };
    dailyOffenseMap.set(key, record);
  }
  return record;
};

const incrementDailyOffense = async (chatId: number, userId: number): Promise<number> => {
  const record = await ensureDailyRecord(chatId, userId);
  record.count += 1;
  dailyOffenseMap.set(offenseKey(chatId, userId), record);
  return record.count;
};

const getMuteCount = async (chatId: number, userId: number): Promise<number> => {
  const key = offenseKey(chatId, userId);
  const cached = muteCountCache.get(key);
  if (cached !== null && cached !== undefined) return cached;
  const remote = await fetchActionCount(chatId, userId, "muted", "all");
  muteCountCache.set(key, remote);
  return remote;
};

const incrementMuteCount = async (chatId: number, userId: number): Promise<number> => {
  const current = await getMuteCount(chatId, userId);
  const next = current + 1;
  muteCountCache.set(offenseKey(chatId, userId), next);
  return next;
};

const formatLocalDateTime = () =>
  new Intl.DateTimeFormat("id-ID", {
    timeZone: LOCAL_TIMEZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date());

const ensureGroupSync = async (bot: TelegramBot, chat: TelegramBot.Chat) => {
  if (chat.type !== "group" && chat.type !== "supergroup") return;
  const cached = groupSyncCache.get(chat.id);
  if (cached) return;
  await syncGroup(chat.id, { title: chat.title ?? chat.username ?? undefined, group_type: chat.type });
  groupSyncCache.set(chat.id, true);
};

const releaseManualStatus = async (bot: TelegramBot, chatId: number, userId: number, status: MemberStatus) => {
  try {
    if (status === "muted") {
      await bot.restrictChatMember(chatId, userId, { permissions: defaultPermissions });
    } else if (status === "banned") {
      await bot.unbanChatMember(chatId, userId, { only_if_banned: true });
    }
  } catch (error) {
    logger.warn({ err: error, chatId, userId, status }, "Failed to release manual status");
  }
};

const applyMute = async (
  bot: TelegramBot,
  chatId: number,
  userId: number,
  username: string | undefined,
  durationMinutes = 10,
  reason = "Automatic moderation",
) => {
  const until = Math.floor(Date.now() / 1000) + durationMinutes * 60;
  await bot.restrictChatMember(chatId, userId, { permissions: mutePermissions, until_date: until });
  await upsertMemberModeration(chatId, {
    user_id: userId,
    username,
    status: "muted",
    duration_minutes: durationMinutes,
    reason,
  });
};

const applyBan = async (
  bot: TelegramBot,
  chatId: number,
  userId: number,
  username: string | undefined,
  reason = "Automatic moderation",
) => {
  await bot.banChatMember(chatId, userId);
  await upsertMemberModeration(chatId, {
    user_id: userId,
    username,
    status: "banned",
    reason,
  });
};

const releaseMuteIfExpired = async (bot: TelegramBot, chatId: number, member: MemberModeration) => {
  await bot.restrictChatMember(chatId, member.user_id, { permissions: defaultPermissions });
  await removeMemberModeration(chatId, member.user_id, member.status);
};

const releaseBanIfExpired = async (bot: TelegramBot, chatId: number, member: MemberModeration) => {
  await bot.unbanChatMember(chatId, member.user_id, { only_if_banned: true });
  await removeMemberModeration(chatId, member.user_id, member.status);
};

const enforceManualStatuses = async (bot: TelegramBot, chatId: number) => {
  try {
    const members = await fetchMemberModerations(chatId);
    const now = Date.now();
    const prevStatuses = manualStatusCache.get(chatId) ?? new Map<number, MemberStatus>();
    const nextStatuses = new Map<number, MemberStatus>();
    for (const member of members) {
      nextStatuses.set(member.user_id, member.status);
      try {
        if (member.status === "muted") {
          if (member.expires_at && new Date(member.expires_at).getTime() <= now) {
            await releaseMuteIfExpired(bot, chatId, member);
            continue;
          }
          const until = member.expires_at ? Math.floor(new Date(member.expires_at).getTime() / 1000) : undefined;
          await bot.restrictChatMember(chatId, member.user_id, { permissions: mutePermissions, until_date: until });
        } else if (member.status === "banned") {
          if (member.expires_at && new Date(member.expires_at).getTime() <= now) {
            await releaseBanIfExpired(bot, chatId, member);
            continue;
          }
          await bot.banChatMember(chatId, member.user_id);
        }
      } catch (error) {
        if (isInvalidUserError(error)) {
          logger.warn({ chatId, memberId: member.user_id }, "Skip enforcement for user not found");
          continue;
        }
        throw error;
      }
    }
    const releases: Array<[number, MemberStatus]> = [];
    prevStatuses.forEach((status, userId) => {
      if (!nextStatuses.has(userId)) {
        releases.push([userId, status]);
      }
    });
    for (const [userId, status] of releases) {
      await releaseManualStatus(bot, chatId, userId, status);
    }
    manualStatusCache.set(chatId, nextStatuses);
  } catch (error) {
    logger.error({ err: error, chatId }, "Failed to enforce member statuses");
  }
};

const formatModerationMessage = ({
  username,
  userId,
  score,
  action,
  text,
  warnCount,
  reason,
}: {
  username?: string;
  userId: number;
  score: number;
  action: "muted" | "banned";
  text: string;
  warnCount: number;
  reason: string;
}) => {
  const name = username ? `@${username}` : `ID ${userId}`;
  const actionLabel = action === "banned" ? "diblokir" : "dimute";
  const time = formatLocalDateTime();
  return (
    "🚨 Pesan Telah Dihapus: Hate Speech Terdeteksi!\n" +
    "─────────────────────────────\n" +
    `👤 Nama     : ${name}\n` +
    `📊 Skor     : ${(score * 100).toFixed(2)}%\n` +
    `🕒 Waktu    : ${time}\n` +
    "─────────────────────────────\n" +
    `💬 Pesan: <tg-spoiler>${text.slice(0, 200)}</tg-spoiler>\n\n` +
    "🙏 Jaga diskusi tetap santun!\n" +
    "🤖 Pesan ini dicek otomatis oleh HateSpeechBot\n\n" +
    `⚠️ Peringatan ke-${warnCount} untuk user ini hari ini.\n` +
    `📌 Tindakan: ${actionLabel}\n` +
    `📝 Alasan  : ${reason}`
  );
};

const formatWarningMessage = ({
  username,
  userId,
  score,
  warnCount,
  text,
}: {
  username?: string;
  userId: number;
  score: number;
  warnCount: number;
  text: string;
}) => {
  const name = username ? `@${username}` : `ID ${userId}`;
  const time = formatLocalDateTime();
  return (
    "⚠️ Peringatan Otomatis\n" +
    "─────────────────────────────\n" +
    `👤 Nama     : ${name}\n` +
    `📊 Skor     : ${(score * 100).toFixed(2)}%\n` +
    `🕒 Waktu    : ${time}\n` +
    "─────────────────────────────\n" +
    `💬 Pesan: ${text.slice(0, 200)}\n\n` +
    `⚠️ Peringatan ke-${warnCount} hari ini (maks 4). Pesan berikutnya akan dimoderasi lebih ketat.`
  );
};

export const registerHandlers = (bot: TelegramBot) => {
  setInterval(() => {
    trackedChats.forEach((chatId) => {
      enforceManualStatuses(bot, chatId).catch((err) => logger.error({ err, chatId }, "enforce_error"));
    });
  }, 15_000);

  bot.onText(/\/start/, async (msg: TelegramBot.Message) => {
    if (!msg.chat) return;
    const chatId = msg.chat.id;
    trackedChats.add(chatId);
    await ensureGroupSync(bot, msg.chat);
    const baseUrl = botConfig.miniAppUrl;
    if (isHttpsUrl(baseUrl)) {
      const button =
        msg.chat.type === "private"
          ? {
              text: "Open Dashboard",
              web_app: { url: `${baseUrl}?chat_id=${chatId}` },
            }
          : {
              text: "Open Dashboard",
              url: `${baseUrl}?chat_id=${chatId}`,
            };
      const keyboard = {
        inline_keyboard: [[button]],
      };
      await bot.sendMessage(
        chatId,
        "👋 Selamat datang di Hate Guard!\nKlik tombol di bawah untuk membuka dashboard mini app.",
        { reply_markup: keyboard },
      );
    } else {
      await bot.sendMessage(
        chatId,
        "👋 Selamat datang di Hate Guard!\nMini app membutuhkan URL HTTPS. Harap konfigurasikan `MINI_APP_BASE_URL` ke alamat HTTPS (mis. via ngrok) sebelum membuka dashboard.",
      );
    }
  });

  bot.onText(/\/moderation_on/, async (msg: TelegramBot.Message) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    await setSettings(chatId, { enabled: true });
    await bot.sendMessage(chatId, "✅ Moderasi otomatis telah diaktifkan.");
  });

  bot.onText(/\/moderation_off/, async (msg: TelegramBot.Message) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    await setSettings(chatId, { enabled: false });
    await bot.sendMessage(chatId, "⛔ Moderasi otomatis dimatikan sementara.");
  });

  bot.onText(/\/set_threshold(?:@\w+)?(?:\s+([\d.]+))?/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    const raw = match?.[1];
    if (!raw) return bot.sendMessage(chatId, "Gunakan format: /set_threshold 0.65");
    const value = Number(raw);
    if (Number.isNaN(value) || value <= 0 || value >= 1) {
      return bot.sendMessage(chatId, "Threshold harus di antara 0 dan 1.");
    }
    await setSettings(chatId, { threshold: value });
    await bot.sendMessage(chatId, `Threshold diperbarui menjadi ${value.toFixed(2)}.`);
  });

  bot.onText(/\/set_mode(?:@\w+)?(?:\s+(\w+))?/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    const value = match?.[1]?.toLowerCase();
    if (!value || !["precision", "balanced", "recall"].includes(value)) {
      return bot.sendMessage(chatId, "Gunakan: /set_mode precision|balanced|recall");
    }
    await setSettings(chatId, { mode: value as SettingsResponse["mode"] });
    await bot.sendMessage(chatId, `Mode diubah ke ${bold(value)}.`, { parse_mode: "HTML" });
  });

  bot.onText(/\/stats(?:@\w+)?(?:\s+(24h|7d))?/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    const window = match?.[1] ?? "24h";
    try {
      const stats = await fetchStats(chatId, window);
      await bot.sendMessage(
        chatId,
        [
          `📊 Statistik ${window}`,
          `Total: ${stats.total_events}`,
          `Deleted: ${stats.deleted}`,
          `Warned: ${stats.warned}`,
          `Blocked: ${stats.blocked}`,
          `Top offenders: ${stats.top_offenders.join(", ") || "-"}`,
        ].join("\n"),
      );
    } catch (error) {
      logger.error({ err: error }, "Failed to fetch stats");
      await bot.sendMessage(chatId, "Tidak bisa mengambil statistik saat ini.");
    }
  });

  bot.onText(/\/test(?:@\w+)?\s+(.+)/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    const text = match?.[1];
    if (!text) return bot.sendMessage(chatId, "Masukkan teks setelah perintah.");
    try {
      const result = await runAdminTest(text);
      await bot.sendMessage(
        chatId,
        `🔬 Hasil uji:\nHate: ${(result.prob_hate * 100).toFixed(1)}%\nNon-hate: ${(result.prob_nonhate * 100).toFixed(1)}%\nLabel: ${result.label}`,
      );
    } catch (error) {
      logger.error({ err: error }, "Failed to run test");
      await bot.sendMessage(chatId, "Tidak bisa menjalankan uji teks sekarang.");
    }
  });

  bot.onText(/\/admins/, async (msg: TelegramBot.Message) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    const admins = await fetchAdmins(chatId);
    const lines = admins.map((admin, idx) => `${idx + 1}. ${admin.user_id}`);
    await bot.sendMessage(chatId, lines.length ? `Daftar admin terotorisasi:\n${lines.join("\n")}` : "Belum ada admin tambahan.");
  });

  bot.onText(/\/admin_add(?:@\w+)?\s+(\d+)/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    const userId = Number(match?.[1]);
    if (!userId) return bot.sendMessage(chatId, "Gunakan format: /admin_add <user_id>");
    await addAdmin(chatId, userId);
    backendAdminCache.delete(chatId);
    await bot.sendMessage(chatId, `Admin ${userId} ditambahkan.`);
  });

  bot.onText(/\/admin_remove(?:@\w+)?\s+(\d+)/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    const userId = Number(match?.[1]);
    if (!userId) return bot.sendMessage(chatId, "Gunakan format: /admin_remove <user_id>");
    await removeAdmin(chatId, userId);
    backendAdminCache.delete(chatId);
    await bot.sendMessage(chatId, `Admin ${userId} dihapus.`);
  });

  bot.onText(/\/mute(?:@\w+)?\s+(\d+)(?:\s+(\d+))?/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    const userId = Number(match?.[1]);
    const minutes = match?.[2] ? Number(match[2]) : 30;
    if (!userId) return bot.sendMessage(chatId, "Gunakan format: /mute <user_id> [menit]");
    try {
      await applyMute(bot, chatId, userId, undefined, minutes, `Manual mute oleh ${msg.from?.username ?? msg.from?.id}`);
      await bot.sendMessage(chatId, `User ${userId} dimute selama ${minutes} menit.`);
    } catch (error) {
      logger.error({ err: error }, "Manual mute failed");
      await bot.sendMessage(chatId, "Gagal melakukan mute.");
    }
  });

  bot.onText(/\/ban(?:@\w+)?\s+(\d+)/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    const userId = Number(match?.[1]);
    if (!userId) return bot.sendMessage(chatId, "Gunakan format: /ban <user_id>");
    try {
      await applyBan(bot, chatId, userId, undefined, `Manual ban oleh ${msg.from?.username ?? msg.from?.id}`);
      await bot.sendMessage(chatId, `User ${userId} diblokir.`);
    } catch (error) {
      logger.error({ err: error }, "Manual ban failed");
      await bot.sendMessage(chatId, "Gagal melakukan ban.");
    }
  });

  bot.onText(/\/why(?:@\w+)?\s+(.+)/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    const text = match?.[1];
    if (!text) {
      return bot.sendMessage(chatId, "Gunakan format: /why <teks>");
    }
    try {
      const result = await predictText(text);
      await bot.sendMessage(chatId, `🤖 Skor kebencian ${result.prob_hate.toFixed(2)}.`);
    } catch (error) {
      logger.error({ err: error }, "Why command failed");
      await bot.sendMessage(chatId, "Gagal memproses permintaan.");
    }
  });

  bot.on("message", async (msg: TelegramBot.Message) => {
    if (!msg.chat || !msg.from) return;
    if (msg.chat.type !== "group" && msg.chat.type !== "supergroup") return;
    const text = msg.text ?? msg.caption;
    if (!text || text.startsWith("/")) return;
    if (msg.from.is_bot) return;

    try {
      trackedChats.add(msg.chat.id);
      await ensureGroupSync(bot, msg.chat);
      const settings = await getSettings(msg.chat.id);
      if (!settings.enabled) return;
      const isPrivileged = await requireAdmin(bot, msg.chat.id, msg.from.id);
      const prediction = await predictText(text);
      const threshold = settings.threshold;
      const baseEvent: EventPayload = {
        chat_id: msg.chat.id,
        user_id: msg.from.id,
        username: msg.from.username ?? msg.from.first_name ?? undefined,
        message_id: msg.message_id,
        text: text.slice(0, 1000),
        prob_hate: prediction.prob_hate,
        prob_nonhate: prediction.prob_nonhate,
        action: "allowed",
      };
      if (isPrivileged && !botConfig.moderateAdmins) {
        await persistEvent({
          ...baseEvent,
          action: "bypassed_admin",
          reason: "Pengguna admin dibebaskan dari moderasi",
        });
        return;
      }
      if (prediction.prob_hate < threshold) {
        await persistEvent({
          ...baseEvent,
          action: "allowed",
          reason: `Skor ${formatPercent(prediction.prob_hate)} < ambang ${formatPercent(threshold)}`,
        });
        return;
      }

      const rapidOffenses = incrementOffense(msg.chat.id, msg.from.id);
      const dailyOffenses = await incrementDailyOffense(msg.chat.id, msg.from.id);
      const severity = prediction.prob_hate - threshold;
      if (dailyOffenses < 5) {
        const warningReason = `Peringatan ke-${dailyOffenses} hari ini (batas 4)`;
        await persistEvent({
          ...baseEvent,
          action: "warned",
          reason: warningReason,
        });

        try {
          await bot.deleteMessage(msg.chat.id, msg.message_id);
        } catch (error) {
          logger.warn({ err: error }, "Failed to delete warning message");
        }

        const warningText = formatWarningMessage({
          username: msg.from.username ?? msg.from.first_name,
          userId: msg.from.id,
          score: prediction.prob_hate,
          warnCount: dailyOffenses,
          text,
        });
        await bot.sendMessage(msg.chat.id, warningText);
        return;
      }

      let moderationAction: "muted" | "banned" = "muted";
      let muteDuration = severity >= 0.2 ? 45 : 20;
      let moderationReason = severity >= 0.2 ? "Auto mute (tingkat tinggi)" : "Auto mute (5 pelanggaran)";

      if (rapidOffenses > 1) {
        muteDuration = Math.min(180, muteDuration + (rapidOffenses - 1) * 5);
      }

      const priorMuteCount = await getMuteCount(msg.chat.id, msg.from.id);
      if (priorMuteCount >= 3) {
        moderationAction = "banned";
        moderationReason = `Auto ban (telah dimute ${priorMuteCount}x)`;
      }

      if (moderationAction === "banned") {
        await applyBan(bot, msg.chat.id, msg.from.id, msg.from.username, moderationReason);
      } else {
        const muteCount = await incrementMuteCount(msg.chat.id, msg.from.id);
        moderationReason = `${moderationReason} · mute ke-${muteCount}`;
        await applyMute(
          bot,
          msg.chat.id,
          msg.from.id,
          msg.from.username,
          muteDuration,
          moderationReason,
        );
      }

      const notifyText = formatModerationMessage({
        username: msg.from.username ?? msg.from.first_name,
        userId: msg.from.id,
        score: prediction.prob_hate,
        action: moderationAction,
        text,
        warnCount: dailyOffenses,
        reason: moderationReason,
      });

      await persistEvent({
        ...baseEvent,
        action: moderationAction,
        reason: `${moderationReason} · pelanggaran ke-${dailyOffenses} hari ini`,
      });

      try {
        await bot.deleteMessage(msg.chat.id, msg.message_id);
      } catch (error) {
        logger.warn({ err: error }, "Failed to delete message");
      }

      await bot.sendMessage(msg.chat.id, notifyText);
    } catch (error) {
      logger.error({ err: error }, "Failed to process message");
    }
  });
};
