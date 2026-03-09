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

# --- MACHINE LEARNING MODULE ---
try:
    from sklearn.ensemble import RandomForestClassifier
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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
        self.callback(server); self.destroy()

class UltimateAutotrader(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("XAUUSD Commander v25.1 (MTF AI)")
        self.geometry("850x1650")
        self.is_running = False
        self.webhook_url = ""
        self.selected_server = "Exness-MT5Trial17"
        
        self.live_balance = 0.0
        self.session_start_balance = 0.0
        self.wins, self.losses = 0, 0
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
        
        threading.Thread(target=self.fetch_online_data, daemon=True).start()

    def setup_ui(self):
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # --- Sessions Clocks ---
        self.clock_frame = ctk.CTkFrame(self.main_container, fg_color="#111", border_width=1, border_color="#333")
        self.clock_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        self.clock_frame.grid_columnconfigure((0,1,2,3), weight=1)
        self.syd_title, self.syd_time = self.create_clock_widget(self.clock_frame, "SYDNEY", 0)
        self.tok_title, self.tok_time = self.create_clock_widget(self.clock_frame, "TOKYO", 1)
        self.lon_title, self.lon_time = self.create_clock_widget(self.clock_frame, "LONDON", 2)
        self.ny_title, self.ny_time = self.create_clock_widget(self.clock_frame, "NEW YORK", 3)
        self.update_clocks()

        # Performance Tracker
        self.tracker_frame = ctk.CTkFrame(self.main_container, fg_color="#1a1a1a", border_width=1, border_color="#00D2FF")
        self.tracker_frame.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        self.tracker_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.total_profit_lbl = self.create_stat(self.tracker_frame, "SESSION PROFIT", "0.00", 0, 0)
        self.total_profit_lbl.configure(text_color="#FFD700")
        self.wl_stat_lbl = self.create_stat(self.tracker_frame, "WINS / LOSSES", "0 / 0", 0, 1)
        self.winrate_lbl = self.create_stat(self.tracker_frame, "WIN RATE %", "0%", 0, 2)
        ai_txt = "WAITING" if ML_AVAILABLE else "SKLEARN MISSING"
        self.ai_lbl = self.create_stat(self.tracker_frame, "AI FORECAST (MTF)", ai_txt, 0, 3)

        # Auth & Password
        self.auth_frame = ctk.CTkFrame(self.main_container); self.auth_frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        self.acc_entry = self.create_input(self.auth_frame, "Account:", 0, 0)
        self.pass_entry = ctk.CTkEntry(self.auth_frame, show="*"); self.pass_entry.grid(row=0, column=3, padx=5, pady=5)
        self.show_pass_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.auth_frame, text="Show", variable=self.show_pass_var, command=self.toggle_password).grid(row=1, column=3)
        self.server_btn = ctk.CTkButton(self.auth_frame, text=f" Server: {self.selected_server}", command=self.open_server_picker)
        self.server_btn.grid(row=2, column=0, columnspan=4, pady=5, sticky="ew")

        # Dashboard Stats
        self.dash_frame = ctk.CTkFrame(self.main_container, fg_color="#111"); self.dash_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.dash_frame.grid_columnconfigure((0,1,2,3), weight=1)
        self.balance_lbl = self.create_stat(self.dash_frame, "BALANCE", "Wait", 0, 0)
        self.pl_lbl = self.create_stat(self.dash_frame, "LIVE P/L", "0.00", 0, 1)
        self.lot_calc_lbl = self.create_stat(self.dash_frame, "ADAPTIVE LOT", "0.00", 0, 2)
        self.spread_lbl = self.create_stat(self.dash_frame, "SPREAD", "Wait", 0, 3)
        self.macro_lbl = self.create_stat(self.dash_frame, "MACRO (M15)", "Wait", 1, 0)
        self.micro_lbl = self.create_stat(self.dash_frame, "MICRO (M5)", "Wait", 1, 1)
        self.atr_lbl = self.create_stat(self.dash_frame, "VOLATILITY", "Wait", 1, 2)
        self.dxy_lbl = self.create_stat(self.dash_frame, "DXY DOLLAR", "Wait", 1, 3)

        # Mode Selection
        self.mode_frame = ctk.CTkFrame(self.main_container, fg_color="transparent"); self.mode_frame.grid(row=4, column=0, padx=20, pady=5)
        self.mode_var = ctk.StringVar(value="Super Aggressive")
        for m in ["Normal", "Aggressive", "Super Aggressive", "Scalper"]:
            ctk.CTkRadioButton(self.mode_frame, text=m, variable=self.mode_var, value=m).pack(side="left", padx=10)

        # Settings Panel
        self.set_frame = ctk.CTkFrame(self.main_container); self.set_frame.grid(row=5, column=0, padx=20, pady=5, sticky="nsew")
        self.risk_pct_entry = self.create_input(self.set_frame, "Risk %:", 0, 0, "1.0")
        self.atr_sl_entry = self.create_input(self.set_frame, "SL Multi:", 0, 2, "1.5")
        self.atr_tp_entry = self.create_input(self.set_frame, "TP Multi:", 0, 4, "3.0")
        self.max_spread_entry = self.create_input(self.set_frame, "Max Spread:", 1, 0, "300")
        self.trail_dist_entry = self.create_input(self.set_frame, "Trail Start:", 1, 2, "1.0")
        self.max_trades_entry = self.create_input(self.set_frame, "Max Trades:", 1, 4, "3")

        self.guard_frame = ctk.CTkFrame(self.main_container, fg_color="transparent"); self.guard_frame.grid(row=6, column=0, padx=20, pady=5)
        self.trail_var = ctk.BooleanVar(value=True); ctk.CTkCheckBox(self.guard_frame, text="Trailing Stop", variable=self.trail_var).pack(side="left", padx=10)
        self.sc_var = ctk.BooleanVar(value=True); ctk.CTkCheckBox(self.guard_frame, text="Smart Close", variable=self.sc_var).pack(side="left", padx=10)
        self.partial_var = ctk.BooleanVar(value=True); ctk.CTkCheckBox(self.guard_frame, text="Partial Profits", variable=self.partial_var, text_color="#00D2FF").pack(side="left", padx=10)
        
        self.webhook_box = ctk.CTkFrame(self.main_container, fg_color="transparent"); self.webhook_box.grid(row=7, column=0, pady=5)
        ctk.CTkButton(self.webhook_box, text="Set Discord", fg_color="#5865F2", width=100, command=self.open_webhook_config).pack(side="left", padx=2)
        ctk.CTkButton(self.webhook_box, text="TEST", fg_color="#555", width=40, command=self.test_webhook).pack(side="left", padx=2)

        # Sensitivity Slider
        self.sense_frame = ctk.CTkFrame(self.main_container, border_width=1, border_color="#333")
        self.sense_frame.grid(row=8, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(self.sense_frame, text="CANDLE SENSITIVITY:", font=("Helvetica", 10, "bold")).pack(side="left", padx=10)
        self.sense_slider = ctk.CTkSlider(self.sense_frame, from_=0.1, to=2.0, number_of_steps=19, command=self.update_sense_label)
        self.sense_slider.set(1.0); self.sense_slider.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        self.sense_val_lbl = ctk.CTkLabel(self.sense_frame, text="1.0x", font=("Consolas", 14, "bold"), text_color="#00D2FF", width=50); self.sense_val_lbl.pack(side="right", padx=10)

        self.mom_frame = ctk.CTkFrame(self.main_container, border_width=1, border_color="#444")
        self.mom_frame.grid(row=9, column=0, padx=20, pady=5, sticky="ew")
        ctk.CTkLabel(self.mom_frame, text="MARKET MOMENTUM:", font=("Helvetica", 10, "bold")).pack(side="left", padx=10)
        self.mom_bar = ctk.CTkProgressBar(self.mom_frame, height=15)
        self.mom_bar.pack(side="left", fill="x", expand=True, padx=10)
        self.mom_bar.set(0.5)

        # Logs
        ctk.CTkLabel(self.main_container, text="LIVE MARKET INTELLIGENCE", font=("Helvetica", 10, "bold"), text_color="gray").grid(row=10, column=0, pady=(10,0))
        self.news_box = ctk.CTkTextbox(self.main_container, height=120, font=("Helvetica", 11), border_width=1); self.news_box.grid(row=11, column=0, padx=20, pady=5, sticky="nsew")
        self.log_box = ctk.CTkTextbox(self.main_container, height=350, font=("Consolas", 11), border_width=2); self.log_box.grid(row=12, column=0, padx=20, pady=10, sticky="nsew")
        self.news_box.configure(state="disabled"); self.log_box.configure(state="disabled")

        # Control Buttons
        self.btn_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.btn_frame.grid(row=13, column=0, pady=10)
        self.start_btn = ctk.CTkButton(self.btn_frame, text="START SCAN", fg_color="#28a745", command=self.start_bot); self.start_btn.pack(side="left", padx=10)
        ctk.CTkButton(self.btn_frame, text="KILL ALL", fg_color="#8b0000", command=self.kill_all_trades).pack(side="left", padx=10)
        self.stop_btn = ctk.CTkButton(self.btn_frame, text="STOP", fg_color="#555", state="disabled", command=self.stop_bot); self.stop_btn.pack(side="left", padx=10)

        self.force_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.force_frame.grid(row=14, column=0, pady=10)
        ctk.CTkButton(self.force_frame, text="FORCE BUY", fg_color="#28a745", width=100, command=lambda: self.trigger_manual_trade("BUY")).pack(side="left", padx=5)
        ctk.CTkButton(self.force_frame, text="FORCE SELL", fg_color="#dc3545", width=100, command=lambda: self.trigger_manual_trade("SELL")).pack(side="left", padx=5)
        ctk.CTkButton(self.force_frame, text="SECURE PROFITS", fg_color="#ffc107", text_color="black", width=120, command=self.secure_profits).pack(side="left", padx=5)

    def toggle_password(self): self.pass_entry.configure(show="" if self.show_pass_var.get() else "*")
    def update_sense_label(self, val): self.sense_val_lbl.configure(text=f"{val:.1f}x")
    def open_server_picker(self): ServerPicker(self, self.set_server)
    def set_server(self, s): self.selected_server = s; self.server_btn.configure(text=f" Server: {s}")
    def open_webhook_config(self): 
        url = ctk.CTkInputDialog(text="Discord Webhook URL:", title="Webhook").get_input()
        if url: self.webhook_url = url.strip(); self.save_config()
    def test_webhook(self): 
        if self.webhook_url: requests.post(self.webhook_url, json={"content": "✅ XAUUSD Commander Linked!"}, timeout=5)

    def log_message(self, msg):
        self.log_box.configure(state="normal"); self.log_box.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"); self.log_box.see("end"); self.log_box.configure(state="disabled")

    # --- ADVANCED AI TRAINING MODULE (Multi-Timeframe Integration) ---
    def train_ai_model(self):
        if not ML_AVAILABLE: return False
        self.log_message("⏳ TRAINING AI ON 5,000 CANDLES (M5/M15/M30/H1 ANALYSIS)...")
        r5 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 5000)
        
        if r5 is None:
            self.log_message("❌ AI TRAINING FAILED: Insufficient MT5 Data.")
            return False
            
        df = pd.DataFrame(r5)
        
        # Feature Engineering: Simulating MTF momentum directly on the M5 axis
        df['RSI'] = self.calc_rsi_series(df)
        df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
        
        df['body_M5'] = df['close'] - df['open']
        df['body_M15'] = df['close'] - df['open'].shift(2) # Last 3 M5 candles = 15m
        df['body_M30'] = df['close'] - df['open'].shift(5) # Last 6 M5 candles = 30m
        df['body_H1'] = df['close'] - df['open'].shift(11) # Last 12 M5 candles = 1h
        
        df['vol_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(12).mean() # Volume spike vs 1H average
        
        df['ema_dist_M5'] = df['close'].ewm(span=9).mean() - df['close'].ewm(span=21).mean()
        df['ema_dist_H1'] = df['close'].ewm(span=108).mean() - df['close'].ewm(span=252).mean() # H1 Trend Equivalent
        
        # Target: 1 if the next close is higher than current close (Bullish prediction)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)
        
        self.ml_features = ['RSI', 'ATR', 'body_M5', 'body_M15', 'body_M30', 'body_H1', 'vol_ratio', 'ema_dist_M5', 'ema_dist_H1']
        X = df[self.ml_features]
        y = df['target']
        
        self.rf_model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        self.rf_model.fit(X, y)
        self.log_message("✅ MULTI-TIMEFRAME DEEP LEARNING MODEL READY.")
        return True

    def calc_rsi_series(self, df):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        return 100 - (100 / (1 + (gain / (loss + 0.0001))))

    def fetch_online_data(self):
        curr = time.time()
        if (curr - self.last_api_pull) < 600 and self.latest_headlines: return self.cached_sentiment, self.cached_dxy_trend
        try:
            sia = SentimentIntensityAnalyzer(); headlines = []
            rss = feedparser.parse("https://www.fxstreet.com/rss/news")
            if rss.entries:
                for e in rss.entries[:3]: headlines.append(e.title)
            tick = yf.Ticker("GC=F")
            if tick.news:
                for n in tick.news[:3]: headlines.append(str(n.get('title')).strip())
            if not headlines: headlines = ["Market news unavailable at this time."]
            
            self.latest_headlines = headlines
            self.news_box.configure(state="normal"); self.news_box.delete("1.0", "end")
            for h in headlines: self.news_box.insert("end", f" • {h}\n")
            self.news_box.configure(state="disabled")
            
            score = sum(sia.polarity_scores(h)['compound'] for h in headlines) / (len(headlines) if headlines else 1)
            self.cached_sentiment = "BULLISH" if score > 0.05 else ("BEARISH" if score < -0.05 else "NEUTRAL")
            
            dxy = yf.Ticker("DX-Y.NYB").history(period="2d")
            if len(dxy) >= 2: self.cached_dxy_trend = "STRONG" if dxy['Close'].iloc[-1] > dxy['Close'].iloc[-2] else "WEAK"
            self.last_api_pull = curr
        except: pass
        return self.cached_sentiment, self.cached_dxy_trend

    def calculate_indicators(self, df, sense=1.0):
        if len(df) < 21: return None
        gain = df['close'].diff().where(df['close'].diff() > 0, 0).rolling(14).mean()
        loss = (-df['close'].diff().where(df['close'].diff() < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 0.0001))))
        df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ATR'] = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1).rolling(14).mean()
        df['Is_Green'] = df['close'] > df['open']
        vol_sum = df['tick_volume'].shift(1) + df['tick_volume'].shift(2) + df['tick_volume'].shift(3)
        df['Vol_Cond'] = df['tick_volume'] > (vol_sum / 3)
        curr_body = (df['close'] - df['open']).abs(); prev_body = (df['open'].shift(1) - df['close'].shift(1)).abs()
        df['Bullish_Engulfing'] = (df['Is_Green'] == True) & (df['Is_Green'].shift(1) == False) & (curr_body > (prev_body * sense)) & df['Vol_Cond']
        df['Bearish_Engulfing'] = (df['Is_Green'] == False) & (df['Is_Green'].shift(1) == True) & (curr_body > (prev_body * sense)) & df['Vol_Cond']
        return df

    def main_loop(self):
        first_run = True
        while self.is_running:
            try:
                acc, pwd = self.acc_entry.get().strip(), self.pass_entry.get().strip()
                if not mt5.initialize(login=int(acc), server=self.selected_server, password=pwd): time.sleep(1); continue
                mt5.symbol_select("XAUUSD", True); info = mt5.account_info(); tick = mt5.symbol_info_tick("XAUUSD")
                if info and tick:
                    if first_run: 
                        self.session_start_balance = info.balance; self.log_message("🧠 ENGINE ONLINE.")
                        if ML_AVAILABLE: self.train_ai_model()
                        first_run = False
                        
                    self.total_profit_lbl.configure(text=f"{(info.balance - self.session_start_balance) + info.profit:,.2f}")
                    self.balance_lbl.configure(text=f"{info.balance:,.2f}"); self.pl_lbl.configure(text=f"{info.profit:,.2f}")
                    self.spread_lbl.configure(text=f"{(tick.ask - tick.bid):.0f} pts")
                    
                    sentiment, dxy = self.fetch_online_data(); self.dxy_lbl.configure(text=dxy)
                    
                    # Increased M5 fetch size to support H1 EMA logic in live prediction
                    r15, r5 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 50), mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 300)
                    if r15 is not None and r5 is not None:
                        sense_val = self.sense_slider.get()
                        df_m15, df_m5 = self.calculate_indicators(pd.DataFrame(r15), sense_val), self.calculate_indicators(pd.DataFrame(r5), sense_val)
                        if df_m15 is not None and df_m5 is not None:
                            atr, trend = df_m5['ATR'].iloc[-1], ("UPTREND" if df_m15['EMA9'].iloc[-1] > df_m15['EMA21'].iloc[-1] else "DOWNTREND")
                            self.live_atr, rsi_m5, bull, bear = atr, df_m5['RSI'].iloc[-1], df_m5['Bullish_Engulfing'].iloc[-2], df_m5['Bearish_Engulfing'].iloc[-2]
                            self.macro_lbl.configure(text=trend, text_color="#28a745" if trend=="UPTREND" else "#dc3545")
                            self.micro_lbl.configure(text="Bull Engulf" if bull else ("Bear Engulf" if bear else "Neutral")); self.atr_lbl.configure(text=f"{atr:.2f}")

                            # Adaptive Lot Display
                            try:
                                sl_d = atr * float(self.atr_sl_entry.get() or "1.5")
                                disp_lot = max(0.01, round((((info.balance / 58) * float(self.risk_pct_entry.get() or "1.0")) / 100) / (sl_d * 100), 2))
                                self.lot_calc_lbl.configure(text=f"{disp_lot}")
                            except: pass

                            # Momentum Bar Logic
                            diff = tick.bid - df_m5['open'].iloc[-1]
                            m_val = max(0, min(1, (diff / (atr * 2)) + 0.5))
                            self.mom_bar.set(m_val)
                            self.mom_bar.configure(progress_color="#28a745" if m_val > 0.5 else "#dc3545")
                            
                            # --- LIVE AI PREDICTION (MTF) ---
                            ai_pred_dir = "NEUTRAL"
                            if hasattr(self, 'rf_model'):
                                body_M5 = df_m5['close'].iloc[-1] - df_m5['open'].iloc[-1]
                                body_M15 = df_m5['close'].iloc[-1] - df_m5['open'].iloc[-3]
                                body_M30 = df_m5['close'].iloc[-1] - df_m5['open'].iloc[-6]
                                body_H1 = df_m5['close'].iloc[-1] - df_m5['open'].iloc[-12]
                                vol_ratio = df_m5['tick_volume'].iloc[-1] / df_m5['tick_volume'].rolling(12).mean().iloc[-1]
                                ema_dist_M5 = df_m5['close'].ewm(span=9).mean().iloc[-1] - df_m5['close'].ewm(span=21).mean().iloc[-1]
                                ema_dist_H1 = df_m5['close'].ewm(span=108).mean().iloc[-1] - df_m5['close'].ewm(span=252).mean().iloc[-1]
                                
                                pred = self.rf_model.predict([[rsi_m5, atr, body_M5, body_M15, body_M30, body_H1, vol_ratio, ema_dist_M5, ema_dist_H1]])[0]
                                prob = self.rf_model.predict_proba([[rsi_m5, atr, body_M5, body_M15, body_M30, body_H1, vol_ratio, ema_dist_M5, ema_dist_H1]])[0][pred] * 100
                                ai_pred_dir = "BULLISH" if pred == 1 else "BEARISH"
                                self.ai_lbl.configure(text=f"{ai_pred_dir} ({prob:.1f}%)", text_color="#28a745" if pred==1 else "#dc3545")

                            mode = self.mode_var.get() # <--- MOVED HERE

                            # Exhaustive Logic Audit
                            if time.time() - self.last_scan_log > 15:
                                curr_body = abs(df_m5['close'].iloc[-1] - df_m5['open'].iloc[-1])
                                prev_body = abs(df_m5['close'].iloc[-2] - df_m5['open'].iloc[-2])
                                
                                self.log_message(f"--- {mode.upper()} LOGIC AUDIT ---")
                                self.log_message(f"AI FORECAST: {ai_pred_dir} | NEWS: {sentiment}")
                                self.log_message(f"TREND: {trend} | RSI: {rsi_m5:.1f} | DXY: {dxy}")
                                self.log_message(f"CANDLE: Size {curr_body:.2f} vs Required {(prev_body*sense_val):.2f}")
                                
                                if trend == "UPTREND":
                                    self.log_message(f"🐂 BULLISH CHECK: {'PASS' if bull else 'WAITING (Need larger green body)'}")
                                    if rsi_m5 > 70: self.log_message("⚠️ RSI BLOCK: Market Overbought.")
                                else:
                                    self.log_message(f"🐻 BEARISH CHECK: {'PASS' if bear else 'WAITING (Need larger red body)'}")
                                    if rsi_m5 < 30: self.log_message("⚠️ RSI BLOCK: Market Oversold.")
                                self.last_scan_log = time.time()

                            # Entry Logic
                            pos = mt5.positions_get(symbol="XAUUSD"); open_count = len(pos) if pos else 0
                            if open_count < int(self.max_trades_entry.get() or "3"):
                                if mode == "Scalper":
                                    r1 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, 1)
                                    if r1 is not None and (r1[0]['close'] > r1[0]['open']): self.execute_trade("BUY", info.balance)
                                    elif r1 is not None: self.execute_trade("SELL", info.balance)
                                elif trend == "UPTREND" and (mode == "Super Aggressive" or (bull and rsi_m5 < 70)): self.execute_trade("BUY", info.balance)
                                elif trend == "DOWNTREND" and (mode == "Super Aggressive" or (bear and rsi_m5 > 30)): self.execute_trade("SELL", info.balance)
                mt5.shutdown(); time.sleep(1)
            except Exception as e: self.log_message(f"Loop Error: {e}"); time.sleep(1)

    def trigger_manual_trade(self, side):
        if mt5.initialize(login=int(self.acc_entry.get()), server=self.selected_server, password=self.pass_entry.get()): self.execute_trade(side, self.live_balance); mt5.shutdown()
    def secure_profits(self):
        if mt5.initialize(login=int(self.acc_entry.get()), server=self.selected_server, password=self.pass_entry.get()):
            pos = mt5.positions_get(symbol="XAUUSD")
            if pos:
                for p in pos:
                    if p.profit > 0: mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "type": 1 if p.type == 0 else 0, "position": p.ticket, "price": mt5.symbol_info_tick(p.symbol).bid if p.type == 0 else mt5.symbol_info_tick(p.symbol).ask, "deviation": 50})
            mt5.shutdown()
    def execute_trade(self, side, bal):
        t = mt5.symbol_info_tick("XAUUSD"); sl_d = self.live_atr * float(self.atr_sl_entry.get() or "1.5"); tp_d = self.live_atr * float(self.atr_tp_entry.get() or "3.0")
        lot = max(0.01, round((((bal / 58) * float(self.risk_pct_entry.get() or "1.0")) / 100) / (sl_d * 100), 2))
        price, sl, tp = (t.ask if side == "BUY" else t.bid), ( (t.ask-sl_d) if side == "BUY" else (t.bid+sl_d) ), ( (t.ask+tp_d) if side == "BUY" else (t.bid-tp_d) )
        res = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": "XAUUSD", "volume": lot, "type": 0 if side == "BUY" else 1, "price": price, "sl": sl, "tp": tp, "magic": 777777})
        if res and res.retcode == 10009: 
            self.log_message(f"🚀 {side} {lot} EXECUTED"); 
            if self.webhook_url: requests.post(self.webhook_url, json={"content": f"🚀 {side} EXECUTED\nLot: {lot}\nPrice: {price}"}, timeout=5)

    def create_stat(self, f, l, v, r, c):
        ctk.CTkLabel(f, text=l, font=("Helvetica", 9, "bold")).grid(row=r*2, column=c); lbl = ctk.CTkLabel(f, text=v, font=("Consolas", 15, "bold")); lbl.grid(row=r*2+1, column=c); return lbl
    def create_input(self, f, l, r, c, d=""):
        ctk.CTkLabel(f, text=l).grid(row=r, column=c, padx=5); e = ctk.CTkEntry(f, width=80); e.insert(0, d); e.grid(row=r, column=c+1, padx=5); return e
    def start_bot(self): self.is_running = True; self.start_btn.configure(state="disabled"); self.stop_btn.configure(state="normal"); threading.Thread(target=self.main_loop, daemon=True).start()
    def stop_bot(self): self.is_running = False; self.start_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
    def kill_all_trades(self):
        if mt5.initialize(login=int(self.acc_entry.get()), server=self.selected_server, password=self.pass_entry.get()):
            pos = mt5.positions_get(symbol="XAUUSD")
            if pos:
                for p in pos: mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "type": 1 if p.type == 0 else 0, "position": p.ticket, "price": mt5.symbol_info_tick(p.symbol).bid if p.type == 0 else mt5.symbol_info_tick(p.symbol).ask, "deviation": 50})
            mt5.shutdown()
    def toggle_on_top(self): self.attributes("-topmost", self.on_top_var.get())
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                c = json.load(f); self.selected_server, self.webhook_url = c.get("srv", "Exness-MT5Trial17"), c.get("webhook", ""); self.server_btn.configure(text=f" Server: {self.selected_server}")
    def save_config(self):
        with open(CONFIG_FILE, "w") as f: json.dump({"srv": self.selected_server, "webhook": self.webhook_url}, f)
    
    def create_clock_widget(self, frame, name, col):
        title_lbl = ctk.CTkLabel(frame, text=name, font=("Helvetica", 9, "bold"))
        title_lbl.grid(row=0, column=col)
        time_lbl = ctk.CTkLabel(frame, text="00:00:00", font=("Consolas", 14, "bold"), text_color="gray")
        time_lbl.grid(row=1, column=col)
        return title_lbl, time_lbl

    def update_clocks(self):
        now = datetime.now(pytz.timezone('Asia/Manila'))
        h = now.hour; time_str = now.strftime("%H:%M:%S")
        syd_active = 5 <= h < 14
        tok_active = 7 <= h < 16
        lon_active = 15 <= h < 24
        ny_active = h >= 20 or h < 5
        
        for t_lbl, tm_lbl, is_active in [(self.syd_title, self.syd_time, syd_active), 
                                         (self.tok_title, self.tok_time, tok_active), 
                                         (self.lon_title, self.lon_time, lon_active), 
                                         (self.ny_title, self.ny_time, ny_active)]:
            color = "#28a745" if is_active else "gray"
            t_lbl.configure(text_color=color); tm_lbl.configure(text=time_str, text_color=color)
        self.after(1000, self.update_clocks)

if __name__ == "__main__":
    UltimateAutotrader().mainloop()