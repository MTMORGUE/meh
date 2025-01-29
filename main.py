#!/usr/bin/env python3
"""
main.py
--------------
Main analysis script with full example "normal analysis code" included.
Prompts user to optionally start asynchronous training and trading scripts.

You can run: python my_main_bot.py
"""

import logging
import subprocess
import sys
import time
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fetch_analysis_data():
    """
    Example function that 'fetches' or simulates data needed for normal analysis.
    In reality, you might query a database, call an API, or parse logs, etc.
    """
    logger.info("Fetching analysis data (placeholder).")
    # Simulate some data points
    data_points = [random.gauss(50, 10) for _ in range(100)]
    return data_points

def run_analysis():
    """
    Example of "normal analysis code".
    We'll do a very simple demonstration with average/median or some placeholder tasks.
    """
    logger.info("=== Starting normal analysis code... ===")

    # 1. Fetch data
    data = fetch_analysis_data()
    if not data:
        logger.warning("No data retrieved. Analysis aborted.")
        return

    # 2. Basic stats
    average_val = sum(data) / len(data)
    sorted_data = sorted(data)
    mid_idx = len(data) // 2
    if len(data) % 2 == 0:
        median_val = (sorted_data[mid_idx - 1] + sorted_data[mid_idx]) / 2.0
    else:
        median_val = sorted_data[mid_idx]

    # 3. Log some results
    logger.info(f"Analysis complete: average={average_val:.2f}, median={median_val:.2f}")

    # 4. Possibly store results or create a report
    # We'll just log for demonstration
    logger.info("Analysis results logged successfully.")

def main():
    """
    Main function that runs the normal analysis and then prompts user
    about starting asynchronous processes for training/trading.
    """
    logger.info("=== my_main_bot.py started ===")

    # 1. Run the normal analysis code
    run_analysis()

    # 2. Prompt user about asynchronous training
    user_input = input("Start the asynchronous training process? (y/N): ").strip().lower()
    if user_input == "y":
        logger.info("Spawning 'automated_training_bot.py' as separate process.")
        subprocess.Popen([sys.executable, "automated_training_bot.py"])
    else:
        logger.info("Skipping asynchronous training process.")

    # 3. Prompt user about asynchronous trading
    user_input = input("Start the automated trading process? (y/N): ").strip().lower()
    if user_input == "y":
        logger.info("Spawning 'async_auto_trader.py' as separate process.")
        subprocess.Popen([sys.executable, "async_auto_trader.py"])
    else:
        logger.info("Skipping automated trading process.")

    logger.info("=== Main script done (normal analysis and prompts complete). ===")

if __name__ == "__main__":
    main()