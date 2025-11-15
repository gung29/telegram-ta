import type { TelegramWebApp } from "@/types/telegram";

export const getTelegram = (): TelegramWebApp | null => {
  if (typeof window === "undefined") return null;
  return window.Telegram?.WebApp ?? null;
};

export const getInitData = (): string => {
  const tg = getTelegram();
  if (tg?.initData) {
    return tg.initData;
  }
  const params = new URLSearchParams(window.location.search);
  return params.get("initData") || "";
};

export const getDebugChatId = (): string | null => {
  const params = new URLSearchParams(window.location.search);
  return params.get("chat_id");
};

export const getDebugUserId = (): string | null => {
  const params = new URLSearchParams(window.location.search);
  return params.get("debug_user_id");
};

export const withHeaders = (): HeadersInit => {
  const initData = getInitData();
  const debugChat = getDebugChatId();
  const debugUser = getDebugUserId();
  const headers: HeadersInit = {
    "Content-Type": "application/json",
  };
  if (initData) {
    headers["X-Init-Data"] = initData;
  } else if (debugChat) {
    headers["X-Debug-Chat-Id"] = debugChat;
    if (debugUser) {
      headers["X-Debug-User-Id"] = debugUser;
    }
  }
  return headers;
};

export const ensureTelegramReady = () => {
  const tg = getTelegram();
  if (!tg) return;
  tg.ready();
  tg.setHeaderColor("secondary_bg_color");
  tg.setBackgroundColor("#05060a");
};
