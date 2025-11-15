import { config as loadEnv } from "dotenv";
import path from "node:path";

const envPath = path.resolve(__dirname, "../../.env");
loadEnv({ path: envPath });

const requireEnv = (key: string): string => {
  const value = process.env[key];
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
  return value;
};

export const botConfig = {
  token: requireEnv("BOT_TOKEN"),
  apiUrl: process.env.INFERENCE_API_URL ?? "http://localhost:8000",
  apiKey: requireEnv("API_KEY"),
  miniAppUrl: process.env.MINI_APP_BASE_URL ?? "http://localhost:8080",
  defaultThreshold: Number(process.env.DEFAULT_THRESHOLD ?? 0.62),
  adminIds: (process.env.ADMIN_IDS ?? "")
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean)
    .map((v) => Number(v)),
  webhookUrl: process.env.WEBHOOK_URL ?? "",
  webhookPort: Number(process.env.BOT_WEBHOOK_PORT ?? process.env.WEBHOOK_PORT ?? 8443),
  webhookHost: process.env.WEBHOOK_HOST ?? "0.0.0.0",
  webhookPath: process.env.WEBHOOK_PATH ?? "/webhook",
  webhookSecret: process.env.WEBHOOK_SECRET ?? "hate-guard-secret",
  moderateAdmins: ["1", "true", "yes"].includes((process.env.MODERATE_ADMINS ?? "false").toLowerCase()),
};

export const useWebhook = Boolean(botConfig.webhookUrl);
