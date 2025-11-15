import express from "express";
import bodyParser from "body-parser";
import TelegramBot from "node-telegram-bot-api";

import { botConfig } from "./config";
import logger from "./logger";

export function startWebhookServer(bot: TelegramBot) {
  const app = express();
  app.use(bodyParser.json());

  app.post(botConfig.webhookPath, (req, res) => {
    const secretHeader = req.headers["x-telegram-bot-api-secret-token"];
    if (secretHeader !== botConfig.webhookSecret) {
      return res.sendStatus(401);
    }
    bot.processUpdate(req.body);
    res.sendStatus(200);
  });

  app.listen(botConfig.webhookPort, botConfig.webhookHost, () => {
    logger.info(
      { host: botConfig.webhookHost, port: botConfig.webhookPort, path: botConfig.webhookPath },
      "Webhook listener running",
    );
  });
}
