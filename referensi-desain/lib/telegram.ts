type TelegramInitDataUnsafe = {
  chat?: { id: number };
  user?: {
    id: number;
    username?: string;
    first_name?: string;
    last_name?: string;
    photo_url?: string;
  };
};

type TelegramWebApp = {
  initData?: string;
  initDataUnsafe?: TelegramInitDataUnsafe;
  ready: () => void;
  setHeaderColor: (color: string) => void;
  setBackgroundColor: (color: string) => void;
  showAlert: (msg: string) => void;
};

declare global {
  interface Window {
    Telegram?: { WebApp?: TelegramWebApp };
  }
}

export const getTelegram = (): TelegramWebApp | null => {
  if (typeof window === "undefined") return null;
  return window.Telegram?.WebApp ?? null;
};

export const getInitData = (): string => {
  const tg = getTelegram();
  if (tg?.initData) {
    return tg.initData;
  }
  const params = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "");
  return params.get("initData") || "";
};

export const getUserAvatar = (): string | undefined => {
  const tg = getTelegram();
  return tg?.initDataUnsafe?.user?.photo_url;
};

export const getUserName = (): string | undefined => {
  const tg = getTelegram();
  const user = tg?.initDataUnsafe?.user;
  if (!user) return undefined;
  return user.username || [user.first_name, user.last_name].filter(Boolean).join(" ");
};

export const getDebugChatId = (): string | null => {
  const params = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "");
  return params.get("chat_id");
};

export const getDebugUserId = (): string | null => {
  const params = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "");
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
