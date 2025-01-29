#!/usr/bin/env python3
"""
app.py
------
A sleek Flask web GUI that:
  - Embeds GMGN chart
  - Shows logs
  - Lets you view the 'tokens' table in your DB
"""

import os
import logging
import sqlite3

from flask import Flask, render_template

app = Flask(__name__)

# Config
DB_PATH = os.environ.get("DB_PATH", "db.sqlite")
LOG_FILE = os.environ.get("LOG_FILE", "my_bot.log")

# Example tokens for ticker
TOKENS = [
    ("sol", "ukHH6c7mMyiWCf1b9pnWe25TSpkDDt3H5pQZgZ74J82", "Solana Sample"),
    ("eth", "0x6982508145454ce325ddbe47a25d4ec3d2311933", "ETH Sample"),
    ("blast", "0xd43d8adac6a4c7d9aeece7c3151fca8f23752cf8", "Blast Sample"),
]

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    # show main embedded chart for the first token
    chain, token, name = TOKENS[0]
    chart_url = f"https://www.gmgn.cc/kline/{chain}/{token}?theme=light&interval=15"
    return render_template("index.html",
                           tokens=TOKENS,
                           main_chart_url=chart_url)

@app.route("/logs")
def view_logs():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.readlines()
            logs = content[-300:]  # show last 300
    logs.reverse()
    return render_template("logs.html", logs=logs)

@app.route("/db")
def db_view():
    rows = []
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, token_name, symbol, status FROM tokens ORDER BY id DESC LIMIT 100")
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        logging.exception("Error reading tokens from DB: %s", e)
        rows = []
    return render_template("db_view.html", tokens=rows)

if __name__ == "__main__":
    app.run(debug=True, port=5000)