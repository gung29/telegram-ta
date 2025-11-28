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
  fetchUserActions,
  fetchGroups,
} from "./apiClient";
import { TTLCache } from "./cache";
import { normalizeObfuscatedTerms } from "./textNormalization";
import { EventPayload, MemberModeration, MemberStatus, SettingsResponse, UserActionSummary } from "./types";

const settingsCache = new TTLCache<SettingsResponse>(5_000);
const backendAdminCache = new TTLCache<number[]>(120_000);
const telegramAdminCache = new TTLCache<number[]>(300_000);
const offenseCache = new TTLCache<number>(15 * 60 * 1000);
const dailyOffenseMap: Map<string, { count: number; day: string }> = new Map();
const muteCountCache = new TTLCache<number>(30 * 24 * 60 * 60 * 1000);
const manualStatusCache = new Map<number, Map<number, MemberStatus>>();

type ExtendedPermissions = TelegramBot.ChatPermissions & { can_send_media_messages?: boolean };

const PERMISSION_KEYS: Array<keyof ExtendedPermissions> = [
  "can_send_messages",
  "can_send_audios",
  "can_send_documents",
  "can_send_media_messages",
  "can_send_photos",
  "can_send_videos",
  "can_send_video_notes",
  "can_send_voice_notes",
  "can_send_polls",
  "can_send_other_messages",
  "can_add_web_page_previews",
];

const setManualStatus = (chatId: number, userId: number, status: MemberStatus) => {
  let chatMap = manualStatusCache.get(chatId);
  if (!chatMap) {
    chatMap = new Map<number, MemberStatus>();
    manualStatusCache.set(chatId, chatMap);
  }
  chatMap.set(userId, status);
};

const clearManualStatus = (chatId: number, userId: number) => {
  const chatMap = manualStatusCache.get(chatId);
  if (!chatMap) return;
  chatMap.delete(userId);
  if (chatMap.size === 0) {
    manualStatusCache.delete(chatId);
  }
};

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

const mutePermissions: TelegramBot.ChatPermissions = {
  can_send_messages: false,
  can_send_audios: false,
  can_send_documents: false,
  can_send_media_messages: false,
  can_send_photos: false,
  can_send_videos: false,
  can_send_video_notes: false,
  can_send_voice_notes: false,
  can_send_polls: false,
  can_send_other_messages: false,
  can_add_web_page_previews: false,
  can_change_info: false,
  can_invite_users: false,
  can_pin_messages: false,
  can_manage_topics: false,
} as ExtendedPermissions;

const allowPermissions: TelegramBot.ChatPermissions = {
  can_send_messages: true,
  can_send_audios: true,
  can_send_documents: true,
  can_send_media_messages: true,
  can_send_photos: true,
  can_send_videos: true,
  can_send_video_notes: true,
  can_send_voice_notes: true,
  can_send_polls: true,
  can_send_other_messages: true,
  can_add_web_page_previews: true,
  can_change_info: false,
  can_invite_users: true,
  can_pin_messages: false,
  can_manage_topics: true,
} as ExtendedPermissions;

const defaultPermissions = allowPermissions;

const resolveChatPermissions = async (bot: TelegramBot, chatId: number): Promise<ExtendedPermissions> => {
  try {
    const chat = await bot.getChat(chatId);
    if (chat?.permissions) {
      return chat.permissions as ExtendedPermissions;
    }
  } catch (error) {
    logger.warn({ err: error, chatId }, "Failed to resolve chat permissions, using default");
  }
  return defaultPermissions;
};

const hasSendAccess = (member: TelegramBot.ChatMember | undefined) => {
  if (!member) return false;
  if (member.status === "kicked" || member.status === "left") return false;
  const perms = (member as TelegramBot.ChatMember & { permissions?: ExtendedPermissions }).permissions;
  if (!perms) return member.status !== "restricted";
  return PERMISSION_KEYS.every((key) => perms[key] !== false);
};

const LOCAL_TIMEZONE = "Asia/Singapore";
const WARN_LIMIT_PER_DAY = 3;

const getOffsetSuffix = (timeZone: string): string => {
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone,
      hour: "2-digit",
      minute: "2-digit",
      timeZoneName: "shortOffset",
    }).formatToParts(new Date());
    const zone = parts.find((p) => p.type === "timeZoneName")?.value ?? "GMT";
    const match = zone.match(/GMT([+-]?)(\d{1,2})(?::?(\d{2}))?/i);
    if (!match) return "+00:00";
    const sign = match[1] === "-" ? "-" : "+";
    const hours = match[2].padStart(2, "0");
    const minutes = match[3] ?? "00";
    return `${sign}${hours}:${minutes}`;
  } catch (error) {
    logger.warn({ err: error }, "Failed to resolve timezone offset");
    return "+00:00";
  }
};

const LOCAL_TZ_OFFSET = getOffsetSuffix(LOCAL_TIMEZONE);

const normalizeIsoForParse = (value: string): string => {
  const timePart = value.includes("T") ? value : value.replace(" ", "T");
  const hasTimezone = /[zZ]|[+-]\d{2}:?\d{2}$/.test(timePart);
  const withZone = hasTimezone ? timePart : `${timePart}${LOCAL_TZ_OFFSET}`;

  // Trim fractional seconds to milliseconds; some backends return microseconds which Date.parse rejects.
  const match = withZone.match(/^(.*?)(\.\d+)?([zZ]|[+-]\d{2}:?\d{2})$/);
  if (!match) return withZone;
  const [, datetime, fractional = "", zone] = match;
  const trimmedFractional = fractional.length > 4 ? `.${fractional.slice(1, 4)}` : fractional;
  return `${datetime}${trimmedFractional}${zone}`;
};

const parseExpiresAtMs = (value?: string | null): number | null => {
  if (!value) return null;
  const normalized = normalizeIsoForParse(value);
  const parsed = Date.parse(normalized);
  if (Number.isNaN(parsed)) {
    logger.warn({ expiresAt: value, normalized }, "Failed to parse expires_at from backend; treating as expired");
    return Date.now() - 1; // treat as expired to avoid permanent restriction
  }
  return parsed;
};

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

const ensureDailyRecord = (chatId: number, userId: number) => {
  const key = offenseKey(chatId, userId);
  const today = getLocalDayKey();
  let record = dailyOffenseMap.get(key);
  if (!record || record.day !== today) {
    record = { count: 0, day: today };
    dailyOffenseMap.set(key, record);
  }
  return record;
};

const incrementDailyOffense = (chatId: number, userId: number): Promise<number> => {
  const record = ensureDailyRecord(chatId, userId);
  record.count += 1;
  dailyOffenseMap.set(offenseKey(chatId, userId), record);
  return Promise.resolve(record.count);
};

const getMuteCount = async (chatId: number, userId: number): Promise<number> => {
  const key = offenseKey(chatId, userId);
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

const liftRestrictions = async (bot: TelegramBot, chatId: number, userId: number) => {
  const chatPerms = await resolveChatPermissions(bot, chatId);
  const mergedPerms = { ...allowPermissions, ...chatPerms };

  const request = async (useIndependent: boolean) => {
    await bot.restrictChatMember(
      chatId,
      userId,
      {
        permissions: mergedPerms,
        use_independent_chat_permissions: useIndependent,
        until_date: 0,
      } as any,
    );
  };

  // Coba dua kali: dengan independent permissions dan kembali ke default chat
  await request(true);
  await request(false);

  const state = await bot.getChatMember(chatId, userId);
  if (!hasSendAccess(state)) {
    // fallback: retry with merged perms + non-independent to avoid missing flags
    await bot.restrictChatMember(
      chatId,
      userId,
      {
        permissions: mergedPerms,
        use_independent_chat_permissions: false,
        until_date: 0,
      } as any,
    );
    const retryState = await bot.getChatMember(chatId, userId);
    if (!hasSendAccess(retryState)) {
      throw new Error("User still restricted after unmute");
    }
  }
};

const releaseManualStatus = async (bot: TelegramBot, chatId: number, userId: number, status: MemberStatus) => {
  try {
    if (status === "muted") {
      await liftRestrictions(bot, chatId, userId);
    } else if (status === "banned") {
      await bot.unbanChatMember(chatId, userId, { only_if_banned: true });
    }
    clearManualStatus(chatId, userId);
  } catch (error) {
    logger.warn({ err: error, chatId, userId, status }, "Failed to release manual status");
  }
};

const manualUnmute = async (bot: TelegramBot, chatId: number, userId: number) => {
  await liftRestrictions(bot, chatId, userId);
  try {
    await removeMemberModeration(chatId, userId, "muted");
  } catch (error) {
    logger.warn({ err: error, chatId, userId }, "Failed to remove backend mute record");
  }
  clearManualStatus(chatId, userId);
};

const manualUnban = async (bot: TelegramBot, chatId: number, userId: number) => {
  await bot.unbanChatMember(chatId, userId, { only_if_banned: true });
  try {
    await removeMemberModeration(chatId, userId, "banned");
  } catch (error) {
    logger.warn({ err: error, chatId, userId }, "Failed to remove backend ban record");
  }
  clearManualStatus(chatId, userId);
};

const applyMute = async (
  bot: TelegramBot,
  chatId: number,
  userId: number,
  username: string | undefined,
  durationMinutes = 10,
  reason = "Automatic moderation",
  nowSeconds?: number, // opsional: kalau mau pakai msg.date
) => {
  // pastikan durasi minimal 1 menit dan integer
  const safeDuration = Math.max(1, Math.round(durationMinutes));

  // basis waktu: pakai nowSeconds (mis. msg.date) kalau ada, kalau tidak Date.now()
  const baseUtcSeconds =
    typeof nowSeconds === "number" ? nowSeconds : Math.floor(Date.now() / 1000);

  const untilUtcSeconds = baseUtcSeconds + safeDuration * 60;

  // (opsional) log buat cek di console
  logger.info(
    {
      chatId,
      userId,
      reason,
      durationMinutes: safeDuration,
      baseUtcSeconds,
      untilUtcSeconds,
      untilIso: new Date(untilUtcSeconds * 1000).toISOString(),
    },
    "applyMute_computed_until",
  );

  await bot.restrictChatMember(
    chatId,
    userId,
    {
      permissions: mutePermissions,
      use_independent_chat_permissions: true,
      until_date: untilUtcSeconds,
    } as any,
  );

  await upsertMemberModeration(chatId, {
    user_id: userId,
    username,
    status: "muted",
    duration_minutes: safeDuration,
    reason,
  });

  setManualStatus(chatId, userId, "muted");
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

  setManualStatus(chatId, userId, "banned");
};



const releaseMuteIfExpired = async (bot: TelegramBot, chatId: number, member: MemberModeration) => {
  logger.info({ chatId, userId: member.user_id }, "Attempting to UNMUTE user");

  try {
    await liftRestrictions(bot, chatId, member.user_id);

    logger.info({ chatId, userId: member.user_id }, "User successfully unmuted");

    await removeMemberModeration(chatId, member.user_id, member.status);
    clearManualStatus(chatId, member.user_id);

  } catch (err) {
    logger.error({ err, chatId, userId: member.user_id }, "FAILED to unmute user");
  }
};


const releaseBanIfExpired = async (bot: TelegramBot, chatId: number, member: MemberModeration) => {
  await bot.unbanChatMember(chatId, member.user_id, { only_if_banned: true });
  await removeMemberModeration(chatId, member.user_id, member.status);
  clearManualStatus(chatId, member.user_id);
};

const enforceManualStatuses = async (bot: TelegramBot, chatId: number) => {
  try {
    const members = await fetchMemberModerations(chatId);
    const now = Date.now();
    const prevStatuses = manualStatusCache.get(chatId) ?? new Map<number, MemberStatus>();
    const nextStatuses = new Map<number, MemberStatus>();
    for (const member of members) {
      try {
        if (member.status === "muted") {
          const expiresAtMs = parseExpiresAtMs(member.expires_at);
          // jika tidak ada expires_at (korup) atau sudah lewat masa berlaku -> lepaskan mute agar tidak nyangkut
          if (expiresAtMs === null || expiresAtMs <= now) {
            await releaseMuteIfExpired(bot, chatId, member);
            continue;
          }

          // jika admin Telegram sudah meng-unmute secara manual,
          // status user di chat biasanya bukan lagi "restricted" atau sudah bisa kirim pesan
          try {
            const chatMember = await bot.getChatMember(chatId, member.user_id);
            if (chatMember && (chatMember.status !== "restricted" || hasSendAccess(chatMember))) {
              // hormati unmute manual: hapus record backend, jangan remute
              await removeMemberModeration(chatId, member.user_id, member.status);
              continue;
            }
          } catch (error) {
            if (isInvalidUserError(error)) {
              logger.warn({ chatId, memberId: member.user_id }, "Skip enforcement for user not found (getChatMember)");
              continue;
            }
            throw error;
          }

          const until = Math.floor(expiresAtMs / 1000);
          await bot.restrictChatMember(
            chatId,
            member.user_id,
            { permissions: mutePermissions, use_independent_chat_permissions: true, until_date: until } as any,
          );
        } else if (member.status === "banned") {
          const expiresAtMs = parseExpiresAtMs(member.expires_at);
          if (expiresAtMs !== null && expiresAtMs <= now) {
            await releaseBanIfExpired(bot, chatId, member);
            continue;
          }

          // kalau admin sudah unban manual (user kembali jadi member/administrator),
          // jangan paksa ban ulang
          try {
            const chatMember = await bot.getChatMember(chatId, member.user_id);
            if (chatMember && chatMember.status !== "kicked") {
              await removeMemberModeration(chatId, member.user_id, member.status);
              continue;
            }
          } catch (error) {
            if (isInvalidUserError(error)) {
              logger.warn({ chatId, memberId: member.user_id }, "Skip enforcement for user not found (getChatMember ban)");
              continue;
            }
            throw error;
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
      // hanya simpan status untuk member yang memang masih harus dimoderasi
      nextStatuses.set(member.user_id, member.status);
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
  durationMinutes,
}: {
  username?: string;
  userId: number;
  score: number;
  action: "muted" | "banned";
  text: string;
  warnCount: number;
  reason: string;
  durationMinutes?: number;
}) => {
  const name = username ? `@${username}` : `ID ${userId}`;
  const actionLabel = action === "banned" ? "diblokir" : "dimute";
  const time = formatLocalDateTime();
  const spoiler = `<tg-spoiler>${text.slice(0, 200)}</tg-spoiler>`;
  const durationLine =
    action === "muted" && typeof durationMinutes === "number"
      ? `⏳ Durasi   : ${durationMinutes} menit\n`
      : "";
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
    durationLine +
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
  const spoiler = `<tg-spoiler>${text.slice(0, 200)}</tg-spoiler>`;
  return (
    "<b>⚠️ Peringatan Otomatis</b>\n" +
    "─────────────────────────────\n" +
    `👤 Pengguna : ${name}\n` +
    `📊 Skor kebencian : ${(score * 100).toFixed(2)}% (prob. hate speech)\n` +
    `🕒 Waktu    : ${time}\n` +
    "─────────────────────────────\n" +
    `💬 Pesan diawasi: ${spoiler}\n\n` +
    `⚠️ Peringatan ke-${warnCount} hari ini (maks ${WARN_LIMIT_PER_DAY}). Pesan berikutnya akan dimoderasi lebih ketat.`
  );
};

export const registerHandlers = (bot: TelegramBot) => {
  setInterval(async () => {
    try {
      const groups = await fetchGroups(); // /admin/groups di core API
      for (const group of groups) {
        await enforceManualStatuses(bot, group.chat_id);
      }
    } catch (err) {
      logger.error({ err }, "enforce_error_global");
    }
  }, 15_000);

type NextModeration =
  | { action: "muted"; durationMinutes: number; reason: string }
  | { action: "banned"; reason: string };

const getNextModerationStep = (priorMuteCount: number): NextModeration => {
  // priorMuteCount = berapa kali sudah PERNAH dimute sebelumnya (dari backend)
  switch (priorMuteCount) {
    case 0:
      return {
        action: "muted",
        durationMinutes: 10,
        reason: "Auto mute (pelanggaran pertama, 10 menit)",
      };
    case 1:
      return {
        action: "muted",
        durationMinutes: 30,
        reason: "Auto mute (pelanggaran berulang, 30 menit)",
      };
    case 2:
      return {
        action: "muted",
        durationMinutes: 60,
        reason: "Auto mute (riwayat berat, 60 menit)",
      };
    case 3:
      return {
        action: "muted",
        durationMinutes: 360,
        reason: "Auto mute (riwayat berat, 6 jam)",
      };
    default:
      return {
        action: "banned",
        reason: "Auto ban (riwayat pelanggaran sangat berat, melewati batas mute)",
      };
  }
};

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

  bot.onText(/\/getid/, async (msg: TelegramBot.Message) => {
    const chatId = msg.chat?.id;
    const userId = msg.from?.id;
    if (!chatId || !userId) return;
    const username = msg.from?.username ? `@${msg.from.username}` : msg.from?.first_name ?? "User";
    const text = `Halo ${username}!\nUser ID: ${userId}\nChat ID: ${chatId}\n\nKirim User ID ini ke admin agar bisa diberikan akses dashboard.`;
    await bot.sendMessage(chatId, text);
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
      const { normalized } = normalizeObfuscatedTerms(text);
      const result = await runAdminTest(normalized);
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

  bot.onText(/\/unmute(?:@\w+)?(?:\s+(\d+))?/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    const repliedUser = msg.reply_to_message?.from;
    const userId = match?.[1] ? Number(match[1]) : repliedUser?.id;
    if (!userId) {
      return bot.sendMessage(chatId, "Gunakan format: /unmute <user_id> atau balas pesan pengguna yang ingin di-unmute.");
    }
    try {
      await manualUnmute(bot, chatId, userId);
      const name = repliedUser?.username ? `@${repliedUser.username}` : `ID ${userId}`;
      await bot.sendMessage(chatId, `User ${name} sudah di-unmute.`);
    } catch (error) {
      logger.error({ err: error }, "Manual unmute failed");
      await bot.sendMessage(chatId, "Gagal melakukan unmute.");
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

  bot.onText(/\/unban(?:@\w+)?(?:\s+(\d+))?/, async (msg: TelegramBot.Message, match: RegExpExecArray | null) => {
    const chatId = msg.chat?.id;
    if (!chatId) return;
    if (!(await requireAdmin(bot, chatId, msg.from?.id))) {
      await notifyAdminOnly(bot, msg);
      return;
    }
    const repliedUser = msg.reply_to_message?.from;
    const userId = match?.[1] ? Number(match[1]) : repliedUser?.id;
    if (!userId) {
      return bot.sendMessage(chatId, "Gunakan format: /unban <user_id> atau balas pesan pengguna yang ingin di-unban.");
    }
    try {
      await manualUnban(bot, chatId, userId);
      const name = repliedUser?.username ? `@${repliedUser.username}` : `ID ${userId}`;
      await bot.sendMessage(chatId, `User ${name} sudah di-unban.`);
    } catch (error) {
      logger.error({ err: error }, "Manual unban failed");
      await bot.sendMessage(chatId, "Gagal melakukan unban.");
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
      const { normalized } = normalizeObfuscatedTerms(text);
      const result = await predictText(normalized);
      await bot.sendMessage(chatId, `🤖 Skor kebencian ${result.prob_hate.toFixed(2)}.`);
    } catch (error) {
      logger.error({ err: error }, "Why command failed");
      await bot.sendMessage(chatId, "Gagal memproses permintaan.");
    }
  });

  bot.on("message", async (msg: TelegramBot.Message) => {
    if (!msg.chat || !msg.from) return;
    if (msg.chat.type !== "group" && msg.chat.type !== "supergroup") return;
    const originalText = msg.text ?? msg.caption;
    if (!originalText || originalText.startsWith("/")) return;
    if (msg.from.is_bot) return;

    try {
      trackedChats.add(msg.chat.id);
      await ensureGroupSync(bot, msg.chat);
      const settings = await getSettings(msg.chat.id);
      if (!settings.enabled) return;
      const { normalized: normalizedText } = normalizeObfuscatedTerms(originalText);
      const isPrivileged = await requireAdmin(bot, msg.chat.id, msg.from.id);
      const prediction = await predictText(normalizedText);
      const threshold = settings.threshold;
      const baseEvent: EventPayload = {
        chat_id: msg.chat.id,
        user_id: msg.from.id,
        username: msg.from.username ?? msg.from.first_name ?? undefined,
        message_id: msg.message_id,
        text: originalText.slice(0, 1000),
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

      // gunakan summary backend agar konsisten dengan dashboard
      let dailyOffenses = 1;
      try {
        const summaries = await fetchUserActions(msg.chat.id);
        const entry: UserActionSummary | undefined = summaries.find((u) => u.user_id === msg.from!.id);
        const warningsToday = entry?.warnings_today ?? 0;
        dailyOffenses = warningsToday + 1;
      } catch (err) {
        logger.warn({ err, chatId: msg.chat.id, userId: msg.from?.id }, "Failed to fetch user_actions, fallback to local counter");
        const record = ensureDailyRecord(msg.chat.id, msg.from!.id);
        dailyOffenses = record.count + 1;
        dailyOffenseMap.set(offenseKey(msg.chat.id, msg.from!.id), { ...record, count: dailyOffenses });
      }
      const severity = prediction.prob_hate - threshold;

      logger.info(
        {
          chatId: msg.chat.id,
          userId: msg.from.id,
          username: msg.from.username,
          dailyOffenses,
          warnLimit: WARN_LIMIT_PER_DAY,
          prob: prediction.prob_hate,
          threshold,
        },
        "moderation_decision",
      );

      // Warn sampai WARN_LIMIT_PER_DAY kali, setelah itu moderasi (mute/ban)
      if (dailyOffenses <= WARN_LIMIT_PER_DAY) {
        const warningReason = `Peringatan ke-${dailyOffenses} hari ini (batas ${WARN_LIMIT_PER_DAY})`;
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
          text: originalText,
        });
        await bot.sendMessage(msg.chat.id, warningText, { parse_mode: "HTML" });
        return;
      }

            const priorMuteCount = await getMuteCount(msg.chat.id, msg.from.id);
      const step = getNextModerationStep(priorMuteCount);

      let moderationAction: "muted" | "banned" = step.action;
      let muteDuration: number | undefined =
        step.action === "muted" ? step.durationMinutes : undefined;
      let moderationReason = step.reason;

      // Kalau pengguna spam cepat dalam 15 menit, tambahin durasi sedikit (kalau masih mute)
      if (step.action === "muted" && rapidOffenses > 1 && muteDuration) {
        muteDuration = Math.min(360, muteDuration + (rapidOffenses - 1) * 5);
        moderationReason = `${moderationReason} · eskalasi karena spam cepat (${rapidOffenses} pelanggaran dalam 15 menit)`;
      }

      // Di grup biasa (bukan supergroup), Telegram tidak mengizinkan restrictChatMember.
      // Jika seharusnya dimute, kita naikkan menjadi ban supaya moderasi tetap jalan.
      if (msg.chat.type === "group" && moderationAction === "muted") {
        moderationAction = "banned";
        moderationReason = `${moderationReason} · (grup biasa tidak mendukung mute, dinaikkan ke ban)`;
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
          muteDuration ?? 10,
          moderationReason,
          msg.date, // 👈 pakai timestamp dari Telegram
        );
      }

      const notifyText = formatModerationMessage({
        username: msg.from.username ?? msg.from.first_name,
        userId: msg.from.id,
        score: prediction.prob_hate,
        action: moderationAction,
        text: originalText,
        warnCount: dailyOffenses,
        reason: moderationReason,
        durationMinutes: moderationAction === "muted" ? muteDuration : undefined,
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

      await bot.sendMessage(msg.chat.id, notifyText, { parse_mode: "HTML" });
    } catch (error) {
      logger.error({ err: error }, "Failed to process message");
    }
  });
};
