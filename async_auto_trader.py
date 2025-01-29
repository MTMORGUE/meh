#!/usr/bin/env python3

"""
async_auto_trader.py
--------------------
Automated trader that uses the model (or stored signals) to buy/sell tokens
via GMGN’s Telegram bot commands. Minimal placeholder logic.
"""

import os
import sys
import logging
import configparser
import asyncio
import numpy as np
import pandas as pd
from telethon import TelegramClient
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_config(path="config.ini"):
    if not os.path.exists(path):
        logger.error(f"Config not found: {path}")
        sys.exit(1)
    config = configparser.ConfigParser()
    config.read(path)
    return config

class MyTradingModel:
    """Demo model. Real usage might load a joblib/pickle from your trained pipeline."""
    def __init__(self):
        self.model = SGDRegressor(random_state=42)
        self.scaler = MinMaxScaler()

    def predict_price_change(self, features: np.ndarray) -> float:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return float(self.model.predict(features)[0])

async def send_telegram_message(client, chat_id, text):
    if not chat_id:
        return
    try:
        await client.send_message(chat_id, text)
        logger.info(f"Sent to Telegram {chat_id}: {text}")
    except Exception as e:
        logger.exception(f"Error sending telegram msg: {e}")

async def gmgn_bot_command(client, gmgn_bot, command):
    logger.info(f"Sending GMGN bot command: {command}")
    try:
        await client.send_message(gmgn_bot, command)
    except Exception as e:
        logger.exception(f"Failed to send command to GMGN bot: {e}")

async def auto_trading_loop(config, model_obj: MyTradingModel):
    # Telegram config
    api_id = config["telegram"].getint("api_id", 0)
    api_hash = config["telegram"].get("api_hash", "")
    session_name = config["telegram"].get("session_name", "trader_session")
    notify_chat = config["telegram"].get("notification_chat", "")
    gmgn_bot = config["telegram"].get("gmgn_bot_account", "@GmgnTradingBot")

    trade_interval_sec = config["trading"].getint("trade_interval_sec", 120)
    buy_threshold = config["trading"].getfloat("buy_threshold", 0.8)
    sell_threshold = config["trading"].getfloat("sell_threshold", -0.8)

    positions = {}  # track holdings

    client = TelegramClient(session_name, api_id, api_hash)
    await client.start()
    logger.info("Telegram client started for auto trading loop.")

    while True:
        logger.info("=== Checking trade signals ===")
        try:
            # Example data
            tokens_to_eval = [
                {"symbol": "TKN1", "price": 1.23, "feature1": 0.9, "feature2": 0.2},
                {"symbol": "TKN2", "price": 0.04, "feature1": -0.3, "feature2": 0.1},
            ]
            for t in tokens_to_eval:
                symbol = t["symbol"]
                features = np.array([t["feature1"], t["feature2"]], dtype=float)
                pred_change = model_obj.predict_price_change(features)
                logger.info(f"{symbol} => predicted change: {pred_change:.3f}")

                if pred_change > buy_threshold:
                    if symbol not in positions:
                        cmd = f"/buy {symbol} 100"
                        await gmgn_bot_command(client, gmgn_bot, cmd)
                        msg = f"AutoTrader: Bought {symbol} on model signal {pred_change:.3f}"
                        await send_telegram_message(client, notify_chat, msg)
                        positions[symbol] = {
                            "buy_time": asyncio.get_event_loop().time(),
                            "buy_price": t["price"]
                        }
                elif pred_change < sell_threshold:
                    if symbol in positions:
                        cmd = f"/sell {symbol} all"
                        await gmgn_bot_command(client, gmgn_bot, cmd)
                        msg = f"AutoTrader: Sold {symbol} on model signal {pred_change:.3f}"
                        await send_telegram_message(client, notify_chat, msg)
                        del positions[symbol]
                else:
                    logger.debug(f"{symbol}: no trade triggered. pred={pred_change:.3f}")

        except Exception as e:
            logger.exception(f"Error in trading loop iteration: {e}")
            await send_telegram_message(client, notify_chat, f"[AutoTrader] Error: {e}")

        logger.info(f"Sleeping {trade_interval_sec}s before next trade cycle...")
        await asyncio.sleep(trade_interval_sec)

def main():
    config = load_config("config.ini")
    model_obj = MyTradingModel()
    try:
        asyncio.run(auto_trading_loop(config, model_obj))
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt => stopping auto trader.")
    except Exception as e:
        logger.exception(f"Unhandled in main: {e}")

if __name__ == "__main__":
    main()