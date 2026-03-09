import customtkinter as ctk
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
import threading
import time
import requests
import feedparser
import operator
import pytz
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
CONFIG_FILE = "bot_config.json"

class ServerPicker(ctk.CTkToplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.title("Select Exness Server")
        self.geometry("350x500")
        self.attributes("-topmost", True)
        self.callback = callback
        self.scroll_frame = ctk.CTkScrollableFrame(self, label_text="Exness MT5 Servers")
        self.scroll_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        trials = ["-".join(["Exness", f"MT5Trial{i}"]) for i in range(1, 31) if i != 13]
        reals = ["-".join(["Exness", f"MT5Real{i}"]) for i in range(1, 60)]
        for s in ["-".join(["Exness", "MT5Trial"]), "-".join(["Exness", "MT5Real"])] + trials + reals:
            ctk.CTkButton(self.scroll_frame, text=s, fg_color="transparent", anchor="w",
                          command=lambda s=s: self.finish(s)).pack(fill="x", padx=5, pady=2)

    def finish(self, server):
        self.callback(server)
        self.destroy()

class UltimateAutotrader(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("XAUUSD Commander v22.9")
        self.geometry("850x1350")
        self.is_running = False
        self.webhook_url = ""
        self.selected_server = "Exness-MT5Trial17"
        
        self.live_balance = 0.0
        self.session_start_balance = 0.0
        self.live_atr = 0.0
        self.last_api_pull = 0
        self.last_scan_log = 0
        self.cached_sentiment = "NEUTRAL"
        self.cached_dxy_trend = "NEUTRAL"
        self.latest_headlines = []

        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill="both", expand=True)

        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Sessions
        self.clock_frame = ctk.CTkFrame(self.main_container, fg_color="#111", border_width=1, border_color="#333")
        self.clock_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        self.clock_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.syd_lbl = self.create_clock_widget(self.clock_frame, "SYDNEY", 0)
        self.tok_lbl = self.create_clock_widget(self.clock_frame, "TOKYO", 1)
        self.lon_lbl = self.create_clock_widget(self.clock_frame, "LONDON", 2)
        self.ny_lbl = self.create_clock_widget(self.clock_frame, "NEW YORK", 3)
        self.update_clocks()

        # Header
        self.header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.header_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=5)
        ctk.CTkLabel(self.header_frame, text="INSTITUTIONAL INTEL ENGINE", font=("Impact", 24), text_color="#00D2FF").pack(side="left")
        self.on_top_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.header_frame, text="Pin Window", variable=self.on_top_var, command=self.toggle_on_top).pack(side="right")

        self.tracker_frame = ctk.CTkFrame(self.main_container, fg_color="#1a1a1a", border_width=1, border_color="#00D2FF")
        self.tracker_frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        self.total_profit_lbl = self.create_stat(self.tracker_frame, "TOTAL SESSION PROFIT", "0.00", 0, 0)
        self.total_profit_lbl.configure(text_color="#FFD700")

        # Auth
        self.auth_frame = ctk.CTkFrame(self.main_container)
        self.auth_frame.grid(row=3, column=0, padx=20, pady=5, sticky="nsew")
        self.acc_entry = self.create_input(self.auth_frame, "Account:", 0, 0)
        self.pass_entry = ctk.CTkEntry(self.auth_frame, show="*")
        self.pass_entry.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        self.show_pass_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.auth_frame, text="Show", variable=self.show_pass_var, command=self.toggle_password).grid(row=1, column=3, sticky="w")
        self.server_btn = ctk.CTkButton(self.auth_frame, text=f" Server: {self.selected_server}", fg_color="#333", command=self.open_server_picker)
        self.server_btn.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        # Dashboard
        self.dash_frame = ctk.CTkFrame(self.main_container, fg_color="#111")
        self.dash_frame.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        self.dash_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.balance_lbl = self.create_stat(self.dash_frame, "BALANCE", "Wait", 0, 0)
        self.pl_lbl = self.create_stat(self.dash_frame, "LIVE P/L", "0.00", 0, 1)
        self.lot_calc_lbl = self.create_stat(self.dash_frame, "ADAPTIVE LOT", "0.00", 0, 2)
        self.spread_lbl = self.create_stat(self.dash_frame, "LIVE SPREAD", "Wait", 0, 3)
        self.macro_lbl = self.create_stat(self.dash_frame, "MACRO (M15)", "Wait", 1, 0)
        self.micro_lbl = self.create_stat(self.dash_frame, "MICRO (M5)", "Wait", 1, 1)
        self.atr_lbl = self.create_stat(self.dash_frame, "VOLATILITY (ATR)", "Wait", 1, 2)
        self.dxy_lbl = self.create_stat(self.dash_frame, "DXY DOLLAR", "Wait", 1, 3)

        # Auto Exit
        self.exit_frame = ctk.CTkFrame(self.main_container, border_width=1, border_color="#555")
        self.exit_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(self.exit_frame, text="AUTOMATED PROFIT EXIT", font=("Helvetica", 12, "bold"), text_color="#FFD700").grid(row=0, column=0, columnspan=4, pady=5)
        self.exit_type = ctk.StringVar(value="amount")
        ctk.CTkRadioButton(self.exit_frame, text="PHP Target", variable=self.exit_type, value="amount").grid(row=1, column=0, padx=10, pady=5)
        ctk.CTkRadioButton(self.exit_frame, text="% Target", variable=self.exit_type, value="percent").grid(row=1, column=1, padx=10, pady=5)
        self.exit_val_entry = ctk.CTkEntry(self.exit_frame, width=100, placeholder_text="Value")
        self.exit_val_entry.grid(row=1, column=2, padx=10, pady=5)
        self.auto_exit_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.exit_frame, text="Enable Auto Exit", variable=self.auto_exit_var, text_color="#28a745").grid(row=1, column=3, padx=10)

        # Mode Selection
        self.mode_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.mode_frame.grid(row=6, column=0, padx=20, pady=5, sticky="ew")
        self.mode_var = ctk.StringVar(value="Super Aggressive")
        ctk.CTkRadioButton(self.mode_frame, text="Normal", variable=self.mode_var, value="Normal").pack(side="left", padx=10)
        ctk.CTkRadioButton(self.mode_frame, text="Aggressive", variable=self.mode_var, value="Aggressive").pack(side="left", padx=10)
        ctk.CTkRadioButton(self.mode_frame, text="Super Aggressive", variable=self.mode_var, value="Super Aggressive", text_color="#ff0000").pack(side="left", padx=10)
        ctk.CTkRadioButton(self.mode_frame, text="SCALPER MODE", variable=self.mode_var, value="Scalper", text_color="#00D2FF").pack(side="left", padx=10)

        # Settings
        self.settings_frame = ctk.CTkFrame(self.main_container)
        self.settings_frame.grid(row=7, column=0, padx=20, pady=5, sticky="nsew")
        self.risk_pct_entry = self.create_input(self.settings_frame, "Risk %:", 0, 0, default="1.0")
        self.atr_sl_entry = self.create_input(self.settings_frame, "SL Multi:", 0, 2, default="1.5")
        self.atr_tp_entry = self.create_input(self.settings_frame, "TP Multi:", 0, 4, default="3.0")
        self.max_spread_entry = self.create_input(self.settings_frame, "Max Spread:", 1, 0, default="300")
        self.trail_dist_entry = self.create_input(self.settings_frame, "Trail Start:", 1, 2, default="1.0")
        self.max_trades_entry = self.create_input(self.settings_frame, "Max Trades:", 1, 4, default="3")

        self.trail_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.settings_frame, text="Trailing Stop", variable=self.trail_var).grid(row=2, column=0, pady=10, padx=10)
        self.sc_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.settings_frame, text="Smart Close", variable=self.sc_var).grid(row=2, column=1, pady=10, padx=10)
        self.partial_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.settings_frame, text="Partial Profits (1:1 RR)", variable=self.partial_var, text_color="#00D2FF").grid(row=2, column=2, pady=10, padx=10)
        
        # --- NEW: Webhook Controls ---
        self.webhook_box = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.webhook_box.grid(row=2, column=3, pady=10, padx=10, sticky="ew")
        self.discord_btn = ctk.CTkButton(self.webhook_box, text="Set Discord", fg_color="#5865F2", width=100, command=self.open_webhook_config)
        self.discord_btn.pack(side="left", padx=2)
        self.test_btn = ctk.CTkButton(self.webhook_box, text="TEST", fg_color="#555", width=40, command=self.test_webhook)
        self.test_btn.pack(side="left", padx=2)

        # Logs
        ctk.CTkLabel(self.main_container, text="LIVE MARKET INTELLIGENCE", font=("Helvetica", 10, "bold"), text_color="gray").grid(row=8, column=0, pady=(10,0))
        self.news_box = ctk.CTkTextbox(self.main_container, height=120, font=("Helvetica", 11), border_width=1)
        self.news_box.grid(row=9, column=0, padx=20, pady=5, sticky="nsew")
        self.news_box.configure(state="disabled")
        self.log_box = ctk.CTkTextbox(self.main_container, height=350, font=("Consolas", 12), border_width=2)
        self.log_box.grid(row=10, column=0, padx=20, pady=10, sticky="nsew")
        self.log_box.configure(state="disabled")

        # Control Buttons
        self.btn_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.btn_frame.grid(row=11, column=0, pady=10)
        self.start_btn = ctk.CTkButton(self.btn_frame, text="START SCAN", fg_color="#28a745", command=self.start_bot)
        self.start_btn.pack(side="left", padx=10)
        ctk.CTkButton(self.btn_frame, text="KILL ALL", fg_color="#8b0000", command=self.kill_all_trades).pack(side="left", padx=10)
        self.stop_btn = ctk.CTkButton(self.btn_frame, text="STOP", fg_color="#555", state="disabled", command=self.stop_bot)
        self.stop_btn.pack(side="left", padx=10)

        self.force_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.force_frame.grid(row=12, column=0, pady=10)
        ctk.CTkButton(self.force_frame, text="FORCE BUY", fg_color="#28a745", width=100, command=lambda: self.trigger_manual_trade("BUY")).pack(side="left", padx=5)
        ctk.CTkButton(self.force_frame, text="FORCE SELL", fg_color="#dc3545", width=100, command=lambda: self.trigger_manual_trade("SELL")).pack(side="left", padx=5)
        ctk.CTkButton(self.force_frame, text="SECURE PROFITS", fg_color="#ffc107", text_color="black", width=120, command=self.secure_profits).pack(side="left", padx=5)

    def create_clock_widget(self, frame, name, col):
        ctk.CTkLabel(frame, text=name, font=("Helvetica", 9, "bold")).grid(row=0, column=col, pady=(5,0))
        lbl = ctk.CTkLabel(frame, text="00:00:00", font=("Consolas", 14, "bold"), text_color="gray")
        lbl.grid(row=1, column=col, pady=(0,5))
        return lbl

    def update_clocks(self):
        pht = pytz.timezone('Asia/Manila')
        now = datetime.now(pht)
        time_str = now.strftime("%H:%M:%S")
        syd_on, tok_on, lon_on, ny_on = 5 <= now.hour < 14, 8 <= now.hour < 17, 15 <= now.hour < 24, (now.hour >= 21 or now.hour < 6)
        self.syd_lbl.configure(text=time_str, text_color="#28a745" if syd_on else "gray")
        self.tok_lbl.configure(text=time_str, text_color="#28a745" if tok_on else "gray")
        self.lon_lbl.configure(text=time_str, text_color="#28a745" if lon_on else "gray")
        self.ny_lbl.configure(text=time_str, text_color="#28a745" if ny_on else "gray")
        self.after(1000, self.update_clocks)

    def toggle_password(self): self.pass_entry.configure(show="" if self.show_pass_var.get() else "*")
    def open_server_picker(self): ServerPicker(self, self.set_server)
    def set_server(self, s): self.selected_server = s; self.server_btn.configure(text=f" Server: {s}")
    def log_news(self, headlines):
        self.news_box.configure(state="normal"); self.news_box.delete("1.0", "end")
        for h in headlines: self.news_box.insert("end", f" • {h}\n")
        self.news_box.configure(state="disabled")

    def log_message(self, msg):
        self.log_box.configure(state="normal"); self.log_box.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_box.see("end"); self.log_box.configure(state="disabled")

    # --- UPDATED: Better Webhook Logic ---
    def send_discord_alert(self, title, description):
        if not self.webhook_url: 
            self.log_message("⚠️ Error: Webhook URL not set.")
            return
        try:
            data = {"embeds": [{"title": title, "description": description, "color": 5814783}]}
            res = requests.post(self.webhook_url, json=data, timeout=5)
            if res.status_code == 204:
                self.log_message("✅ Discord Alert Sent Successfully.")
            else:
                self.log_message(f"❌ Discord Error: {res.status_code}")
        except Exception as e:
            self.log_message(f"❌ Connection Error: {str(e)}")

    def test_webhook(self):
        self.log_message("Testing Webhook...")
        self.send_discord_alert("🟢 CONNECTION TEST", "XAUUSD Commander is successfully linked to your Discord channel.")

    def fetch_online_data(self):
        curr = time.time()
        if (curr - self.last_api_pull) < 900: return self.cached_sentiment, self.cached_dxy_trend
        try:
            sia = SentimentIntensityAnalyzer(); mixed = []
            rss = feedparser.parse("https://www.fxstreet.com/rss/news")
            if rss.entries:
                for e in rss.entries[:3]: mixed.append(str(e.title).strip())
            tick = yf.Ticker("GC=F")
            if tick.news:
                for n in tick.news[:2]: mixed.append(str(n.get('title')).strip())
            self.latest_headlines = mixed if mixed else ["No headlines found."]
            self.log_news(self.latest_headlines)
            score = sum(sia.polarity_scores(h)['compound'] for h in self.latest_headlines) / len(self.latest_headlines)
            self.cached_sentiment = "BULLISH" if score > 0.05 else ("BEARISH" if score < -0.05 else "NEUTRAL")
            dxy = yf.Ticker("DX-Y.NYB").history(period="2d")
            if not dxy.empty: self.cached_dxy_trend = "STRONG" if dxy['Close'].iloc[-1] > dxy['Close'].iloc[-2] else "WEAK"
            self.last_api_pull = curr
        except: pass
        return self.cached_sentiment, self.cached_dxy_trend

    def calculate_indicators(self, df):
        if len(df) < 21: return None
        gain = df['close'].diff().where(df['close'].diff() > 0, 0).rolling(14).mean()
        loss = (-df['close'].diff().where(df['close'].diff() < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ATR'] = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1).rolling(14).mean()
        df['Is_Green'] = df['close'] > df['open']
        vol_sum = df['tick_volume'].shift(1) + df['tick_volume'].shift(2) + df['tick_volume'].shift(3)
        df['Vol_Cond'] = df['tick_volume'] > vol_sum
        df['Bullish_Engulfing'] = (df['Is_Green'] == True) & (df['Is_Green'].shift(1) == False) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)) & df['Vol_Cond']
        df['Bearish_Engulfing'] = (df['Is_Green'] == False) & (df['Is_Green'].shift(1) == True) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1)) & df['Vol_Cond']
        return df

    def main_loop(self):
        first_run = True
        while self.is_running:
            try:
                acc, pwd = self.acc_entry.get().strip(), self.pass_entry.get().strip()
                if not acc or not mt5.initialize(login=int(acc), server=self.selected_server, password=pwd):
                    time.sleep(1); continue
                
                mt5.symbol_select("XAUUSD", True); info = mt5.account_info()
                if info:
                    if first_run: self.session_start_balance = info.balance; self.log_message("🧠 SYSTEM ONLINE."); first_run = False
                    self.total_profit_lbl.configure(text=f"{(info.balance - self.session_start_balance) + info.profit:,.2f}")
                    self.balance_lbl.configure(text=f"{info.balance:,.2f}"); self.pl_lbl.configure(text=f"{info.profit:,.2f}", text_color="#28a745" if info.profit >= 0 else "#dc3545")
                    if self.auto_exit_var.get():
                        target = float(self.exit_val_entry.get())
                        if (self.exit_type.get() == "amount" and info.profit >= target) or (self.exit_type.get() == "percent" and (info.profit / info.balance) * 100 >= target):
                            self.log_message(f"🎯 Target Reached ({info.profit:.2f}). Neutralizing."); self.kill_all_trades()

                sentiment, dxy = self.fetch_online_data()
                r15, r5, r1 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 50), mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 50), mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, 2)
                tick = mt5.symbol_info_tick("XAUUSD")
                
                if r15 is not None and r5 is not None and r1 is not None and tick is not None:
                    df_m15, df_m5, df_m1 = self.calculate_indicators(pd.DataFrame(r15)), self.calculate_indicators(pd.DataFrame(r5)), pd.DataFrame(r1)
                    m1_green, m1_red = df_m1['close'].iloc[-1] > df_m1['open'].iloc[-1], df_m1['close'].iloc[-1] < df_m1['open'].iloc[-1]
                    
                    if df_m15 is not None and df_m5 is not None:
                        atr, m15_trend = df_m5['ATR'].iloc[-1], ("UPTREND" if df_m15['EMA9'].iloc[-1] > df_m15['EMA21'].iloc[-1] else "DOWNTREND")
                        self.live_atr, rsi_m5, bull, bear = atr, df_m5['RSI'].iloc[-1], df_m5['Bullish_Engulfing'].iloc[-2], df_m5['Bearish_Engulfing'].iloc[-2]
                        
                        self.macro_lbl.configure(text=m15_trend, text_color="#28a745" if m15_trend == "UPTREND" else "#dc3545")
                        self.micro_lbl.configure(text="Bull Engulf" if bull else ("Bear Engulf" if bear else "Neutral"), text_color="#FFD700" if (bull or bear) else "white")
                        self.atr_lbl.configure(text=f"{atr:.2f}"); self.dxy_lbl.configure(text=dxy, text_color="#dc3545" if "STRONG" in dxy else "#28a745")
                        live_spread = tick.ask - tick.bid
                        self.spread_lbl.configure(text=f"{live_spread:.0f} pts", text_color="red" if live_spread > float(self.max_spread_entry.get()) else "white")

                        # Reasoning Log
                        curr_t = time.time()
                        if curr_t - self.last_scan_log > 15:
                            mode = self.mode_var.get()
                            self.log_message("-------------------------------------")
                            self.log_message(f"🧠 {mode.upper()} AUDIT | NEWS: {sentiment}")
                            self.log_message(f"📊 Indicators: RSI {rsi_m5:.1f} | DXY {dxy} | Spread {live_spread:.0f}")
                            
                            if mode == "Scalper":
                                self.log_message(f"⚡ SCALPER Check: M1 {'Green (BUY)' if m1_green else 'Red (SELL)'}")
                            elif m15_trend == "UPTREND":
                                if mode == "Super Aggressive": self.log_message(f"⚡ Super Buy Check: M1 {'Green (PASS)' if m1_green else 'RED (WAIT)'}")
                                elif bull: self.log_message(f"✅ Aggro Buy Check: Engulfing Found | RSI {'PASS' if (mode=='Aggressive' or rsi_m5<70) else 'FAIL'}")
                                else: self.log_message("💤 Status: Waiting for Engulfing.")
                            else:
                                if mode == "Super Aggressive": self.log_message(f"⚡ Super Sell Check: M1 {'Red (PASS)' if m1_red else 'GREEN (WAIT)'}")
                                elif bear: self.log_message(f"✅ Aggro Sell Check: Engulfing Found | RSI {'PASS' if (mode=='Aggressive' or rsi_m5>30) else 'FAIL'}")
                                else: self.log_message("💤 Status: Waiting for Engulfing.")
                            self.last_scan_log = curr_t

                        pos = mt5.positions_get(symbol="XAUUSD")
                        open_count = len(pos) if pos else 0
                        if open_count < int(self.max_trades_entry.get()) and live_spread <= float(self.max_spread_entry.get()):
                            mode = self.mode_var.get()
                            if mode == "Scalper":
                                if m1_green: self.execute_trade("BUY", info.balance)
                                elif m1_red: self.execute_trade("SELL", info.balance)
                            elif m15_trend == "DOWNTREND":
                                if (mode == "Super Aggressive" and m1_red) or (bear and (mode == "Aggressive" or (rsi_m5 > 30 and "STRONG" in dxy))):
                                    self.execute_trade("SELL", info.balance)
                            elif m15_trend == "UPTREND":
                                if (mode == "Super Aggressive" and m1_green) or (bull and (mode == "Aggressive" or (rsi_m5 < 70 and "WEAK" in dxy))):
                                    self.execute_trade("BUY", info.balance)

                        if pos:
                            for p in pos:
                                if self.partial_var.get():
                                    dist = atr * float(self.atr_sl_entry.get())
                                    if (p.type == 0 and p.price_current >= (p.price_open + dist) and p.sl < p.price_open) or \
                                       (p.type == 1 and p.price_current <= (p.price_open - dist) and (p.sl > p.price_open or p.sl == 0.0)):
                                        mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "sl": p.price_open, "tp": p.tp})
                                        if p.volume > 0.01: mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": round(p.volume/2, 2), "type": 1 if p.type == 0 else 0, "position": p.ticket, "price": tick.bid if p.type == 0 else tick.ask, "deviation": 50})
                                if self.trail_var.get():
                                    t_dist = atr * float(self.trail_dist_entry.get())
                                    if p.type == 0 and (p.price_current - p.price_open) > t_dist:
                                        new_sl = p.price_current - (atr * 0.5)
                                        if new_sl > p.sl: mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "sl": new_sl, "tp": p.tp})
                                    elif p.type == 1 and (p.price_open - p.price_current) > t_dist:
                                        new_sl = p.price_current + (atr * 0.5)
                                        if new_sl < p.sl or p.sl == 0: mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "sl": new_sl, "tp": p.tp})
                mt5.shutdown()
                time.sleep(1) 
            except: time.sleep(1)

    def trigger_manual_trade(self, direction):
        if mt5.initialize(login=int(self.acc_entry.get()), server=self.selected_server, password=self.pass_entry.get()):
            self.execute_trade(direction, self.live_balance, is_manual=True); mt5.shutdown()
    def secure_profits(self):
        if mt5.initialize(login=int(self.acc_entry.get()), server=self.selected_server, password=self.pass_entry.get()):
            pos = mt5.positions_get(symbol="XAUUSD")
            if pos:
                for p in pos:
                    if p.profit > 0: mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "type": 1 if p.type == 0 else 0, "position": p.ticket, "price": mt5.symbol_info_tick(p.symbol).bid if p.type == 0 else mt5.symbol_info_tick(p.symbol).ask, "deviation": 50})
                    self.send_discord_alert("💰 SECURE PROFITS", f"Neutralized trade with {p.profit:.2f} PHP.")
            mt5.shutdown()
    def execute_trade(self, side, bal, is_manual=False):
        t = mt5.symbol_info_tick("XAUUSD")
        sl_d, tp_d = self.live_atr * float(self.atr_sl_entry.get()), self.live_atr * float(self.atr_tp_entry.get())
        lot = max(0.01, round((((bal / 58) * float(self.risk_pct_entry.get())) / 100) / (sl_d * 100), 2))
        price, sl, tp = (t.ask if side == "BUY" else t.bid), ( (t.ask-sl_d) if side == "BUY" else (t.bid+sl_d) ), ( (t.ask+tp_d) if side == "BUY" else (t.bid-tp_d) )
        res = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": "XAUUSD", "volume": lot, "type": 0 if side == "BUY" else 1, "price": price, "sl": sl, "tp": tp, "deviation": 50, "magic": 777777})
        if res and res.retcode == 10009: 
            self.log_message(f"🚀 {side} {lot} EXECUTED")
            self.send_discord_alert(f"🚀 {side} EXECUTED", f"Lot: {lot}\nPrice: {price}")

    def create_input(self, f, l, r, c, default=""):
        ctk.CTkLabel(f, text=l).grid(row=r, column=c, padx=5, pady=5)
        e = ctk.CTkEntry(f, width=80); e.insert(0, default); e.grid(row=r, column=c+1, padx=5, pady=5, sticky="ew"); return e
    def create_stat(self, f, l, v, r, c):
        ctk.CTkLabel(f, text=l, font=("Helvetica", 9, "bold")).grid(row=r*2, column=c)
        lbl = ctk.CTkLabel(f, text=v, font=("Consolas", 15, "bold")); lbl.grid(row=r*2+1, column=c); return lbl
    def start_bot(self): self.is_running = True; self.start_btn.configure(state="disabled"); self.stop_btn.configure(state="normal"); threading.Thread(target=self.main_loop, daemon=True).start()
    def stop_bot(self): self.is_running = False; self.start_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
    def kill_all_trades(self):
        if mt5.initialize(login=int(self.acc_entry.get()), server=self.selected_server, password=self.pass_entry.get()):
            pos = mt5.positions_get(symbol="XAUUSD")
            if pos:
                for p in pos: mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "type": 1 if p.type == 0 else 0, "position": p.ticket, "price": mt5.symbol_info_tick(p.symbol).bid if p.type == 0 else mt5.symbol_info_tick(p.symbol).ask, "deviation": 50})
            mt5.shutdown()
    def open_webhook_config(self):
        url = ctk.CTkInputDialog(text="Webhook URL:", title="Discord").get_input()
        if url: self.webhook_url = url.strip(); self.save_config()
    def toggle_on_top(self): self.attributes("-topmost", self.on_top_var.get())
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                c = json.load(f); self.selected_server, self.webhook_url = c.get("srv", "Exness-MT5Trial17"), c.get("webhook", ""); self.server_btn.configure(text=f" Server: {self.selected_server}")
    def save_config(self):
        with open(CONFIG_FILE, "w") as f: json.dump({"srv": self.selected_server, "webhook": self.webhook_url}, f)

if __name__ == "__main__":
    UltimateAutotrader().mainloop()