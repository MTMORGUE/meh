#!/usr/bin/env python3

"""
automated_training_bot.py
-------------------------
Fetches new data from GMGN (placeholder), does incremental model training,
logs results, and sends updates to a Telegram channel.
"""

import os
import sys
import logging
import configparser
import asyncio
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from telethon import TelegramClient

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.ini"):
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    config = configparser.ConfigParser()
    config.read(config_path)
    logger.info(f"Loaded config from {config_path}")
    return config

class GmgnTrainingModel:
    """
    Example incremental learning model with partial_fit.
    """
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.scaler = MinMaxScaler()
        # Initialize with an SGDRegressor for demonstration
        self.model = SGDRegressor(
            max_iter=1,
            eta0=0.001,
            learning_rate="invscaling",
            random_state=42,
            warm_start=True
        )
        logger.info("Created GmgnTrainingModel with partial-fit SGDRegressor.")

    def train_on_data(self, df: pd.DataFrame):
        """
        Expects a 'successScore' column as the target.
        Everything else is features.
        """
        if df.empty:
            logger.warning("Empty DataFrame => skipping training.")
            return
        if "successScore" not in df.columns:
            logger.error("Missing 'successScore' in data. Cannot train.")
            return

        X_cols = [c for c in df.columns if c != "successScore"]
        X = df[X_cols].values
        y = df["successScore"].values

        X_scaled = self.scaler.fit_transform(X)  # simplistic approach

        self.model.partial_fit(X_scaled, y)
        preds = self.model.predict(X_scaled)
        mse = np.mean((preds - y)**2)
        logger.info(f"Partial fit done. Local MSE: {mse:.4f}")


def fetch_new_training_data(config) -> pd.DataFrame:
    """
    Example: fetch from GMGN or random placeholder data.
    """
    logger.info("Fetching new training data (placeholder).")
    # Let's do random data for example
    n_samples = 50
    n_features = 5
    data = np.random.randn(n_samples, n_features)
    success_score = np.random.randn(n_samples) * 50 + 100  # random target
    columns = [f"feat{i}" for i in range(n_features)] + ["successScore"]
    df = pd.DataFrame(np.column_stack([data, success_score]), columns=columns)
    return df

async def send_telegram_message(client, chat_id, text):
    """Helper to send a telegram message."""
    if not chat_id:
        return
    try:
        logger.debug(f"Sending telegram message: {text}")
        await client.send_message(chat_id, text)
    except Exception as e:
        logger.exception(f"Failed to send telegram message: {e}")

async def training_loop(config, model_obj: GmgnTrainingModel):
    """
    Repeatedly fetch new data, partial-fit, log MSE, and send telegram updates.
    """
    telegram_api_id = config["telegram"].getint("api_id", 0)
    telegram_api_hash = config["telegram"].get("api_hash", "")
    telegram_session = config["telegram"].get("session_name", "my_training_session")
    notification_chat = config["telegram"].get("notification_chat", "")
    train_interval_sec = config["training"].getint("train_interval_sec", 300)

    client = TelegramClient(telegram_session, telegram_api_id, telegram_api_hash)
    await client.start()
    logger.info("Telethon client started for training loop.")

    while True:
        try:
            df = fetch_new_training_data(config)
            if not df.empty:
                model_obj.train_on_data(df)
                # Evaluate quickly
                X_cols = [c for c in df.columns if c != "successScore"]
                X_scaled = model_obj.scaler.transform(df[X_cols].values)
                preds = model_obj.model.predict(X_scaled)
                mse = np.mean((preds - df["successScore"])**2)
                msg = f"[TrainingBot] Fetched {len(df)} data points. MSE: {mse:.2f}"
                logger.info(msg)
                await send_telegram_message(client, notification_chat, msg)
            else:
                logger.info("No new data from GMGN this round.")
        except Exception as e:
            logger.exception(f"Error in training loop iteration: {e}")
            await send_telegram_message(client, notification_chat, f"[TrainingBot] Exception: {e}")

        logger.info(f"Sleeping {train_interval_sec}s before next training cycle.")
        await asyncio.sleep(train_interval_sec)

def main():
    config = load_config("config.ini")
    model_obj = GmgnTrainingModel()
    try:
        asyncio.run(training_loop(config, model_obj))
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt => shutting down training.")
    except Exception as e:
        logger.exception(f"Unhandled exception in main: {e}")

if __name__ == "__main__":
    main()