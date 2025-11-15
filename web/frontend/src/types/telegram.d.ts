export interface TelegramChat {
  id: number;
  title?: string;
  type?: string;
}

export interface TelegramUser {
  id: number;
  username?: string;
  first_name?: string;
  last_name?: string;
}

export interface TelegramInitDataUnsafe {
  chat?: TelegramChat;
  user?: TelegramUser;
}

export interface TelegramWebApp {
  initData: string;
  initDataUnsafe?: TelegramInitDataUnsafe;
  ready(): void;
  showAlert(message: string): void;
  setHeaderColor(colorKey: string): void;
  setBackgroundColor(color: string): void;
}

declare global {
  interface Window {
    Telegram?: {
      WebApp?: TelegramWebApp;
    };
  }
}

export {};
