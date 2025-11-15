import TelegramBot from "node-telegram-bot-api";
import { botConfig, useWebhook } from "./config";
import logger from "./logger";
import { registerHandlers } from "./botHandlers";
import { startWebhookServer } from "./server";

async function bootstrap() {
  const bot = new TelegramBot(botConfig.token, { polling: !useWebhook });

  registerHandlers(bot);

  if (useWebhook) {
    await bot.setWebHook(botConfig.webhookUrl, { secret_token: botConfig.webhookSecret });
    startWebhookServer(bot);
    logger.info({ url: botConfig.webhookUrl, port: botConfig.webhookPort }, "Webhook mode enabled");
  } else {
    logger.info("Polling mode enabled");
  }
}

bootstrap().catch((error) => {
  logger.error({ err: error }, "Bot failed to start");
  process.exit(1);
});
