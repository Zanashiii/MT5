import customtkinter as ctk
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import yfinance as yf
import json, os, threading, time, requests
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
        
        trials = [f"Exness-MT5Trial{i}" for i in range(1, 31) if i != 13]
        reals = [f"Exness-MT5Real{i}" for i in range(1, 60)]
        for s in ["Exness-MT5Trial", "Exness-MT5Real"] + trials + reals:
            ctk.CTkButton(self.scroll_frame, text=s, fg_color="transparent", anchor="w",
                          command=lambda s=s: self.finish(s)).pack(fill="x", padx=5, pady=2)

    def finish(self, server):
        self.callback(server)
        self.destroy()

class UltimateAutotrader(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("XAUUSD MTF Commander v16.0")
        self.geometry("750x1200")
        self.is_running = False
        self.webhook_url = ""
        self.selected_server = "Exness-MT5Trial17"
        
        self.live_balance = 0.0
        self.live_atr = 0.0

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        ctk.CTkLabel(self.header_frame, text="MTF COMMAND CENTER", font=("Impact", 24), text_color="#00D2FF").pack(side="left")
        
        self.on_top_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.header_frame, text="Pin Window", variable=self.on_top_var, command=self.toggle_on_top).pack(side="right")

        self.auth_frame = ctk.CTkFrame(self)
        self.auth_frame.grid(row=1, column=0, padx=20, pady=5, sticky="nsew")
        self.acc_entry = self.create_input(self.auth_frame, "Account:", 0, 0)
        ctk.CTkLabel(self.auth_frame, text="Password:").grid(row=0, column=2, padx=5, pady=5)
        self.pass_entry = ctk.CTkEntry(self.auth_frame, show="*")
        self.pass_entry.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        self.show_pass_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.auth_frame, text="Show", variable=self.show_pass_var, command=self.toggle_password, font=("Helvetica", 11)).grid(row=1, column=3, sticky="w")
        self.server_btn = ctk.CTkButton(self.auth_frame, text=f"🌐 Server: {self.selected_server}", fg_color="#333", command=self.open_server_picker)
        self.server_btn.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        self.dash_frame = ctk.CTkFrame(self, fg_color="#111")
        self.dash_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.dash_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.balance_lbl = self.create_stat(self.dash_frame, "BALANCE (PHP)", "---", 0, 0)
        self.pl_lbl = self.create_stat(self.dash_frame, "P/L", "0.00", 0, 1)
        self.lot_calc_lbl = self.create_stat(self.dash_frame, "ADAPTIVE LOT", "0.00", 0, 2)
        
        self.macro_lbl = self.create_stat(self.dash_frame, "MACRO TREND (M15)", "Analyzing...", 1, 0)
        self.micro_lbl = self.create_stat(self.dash_frame, "MICRO CANDLE (M5)", "Analyzing...", 1, 1)
        self.atr_lbl = self.create_stat(self.dash_frame, "VOLATILITY (ATR)", "---", 1, 2)

        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.grid(row=3, column=0, padx=20, pady=5, sticky="nsew")
        self.risk_pct_entry = self.create_input(self.settings_frame, "Risk %:", 0, 0, default="1.0")
        self.atr_sl_entry = self.create_input(self.settings_frame, "SL ATR Multiplier:", 0, 2, default="1.0")
        self.atr_tp_entry = self.create_input(self.settings_frame, "TP ATR Multiplier:", 0, 4, default="2.0")
        
        # AI Logic Toggles
        self.breakeven_var = ctk.BooleanVar(value=True)
        self.be_check = ctk.CTkCheckBox(self.settings_frame, text="🛡️ Auto-Breakeven", variable=self.breakeven_var, text_color="#00FF7F")
        self.be_check.grid(row=1, column=0, pady=10, padx=10, sticky="w")

        self.smartclose_var = ctk.BooleanVar(value=True)
        self.sc_check = ctk.CTkCheckBox(self.settings_frame, text="🧠 Smart Close", variable=self.smartclose_var, text_color="#00D2FF")
        self.sc_check.grid(row=1, column=1, pady=10, padx=10, sticky="w")

        self.discord_btn = ctk.CTkButton(self.settings_frame, text="🔗 Configure Discord Webhook", fg_color="#5865F2", command=self.open_webhook_config)
        self.discord_btn.grid(row=1, column=2, columnspan=4, pady=10, padx=10, sticky="ew")

        self.log_box = ctk.CTkTextbox(self, font=("Consolas", 12), border_width=2)
        self.log_box.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        self.log_box.configure(state="disabled")

        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.grid(row=5, column=0, pady=10)
        self.start_btn = ctk.CTkButton(self.btn_frame, text="START MTF SCAN", fg_color="#28a745", command=self.start_bot)
        self.start_btn.pack(side="left", padx=10)
        self.kill_btn = ctk.CTkButton(self.btn_frame, text="KILL ALL (PANIC)", fg_color="#8b0000", command=self.kill_all_trades)
        self.kill_btn.pack(side="left", padx=10)
        self.stop_btn = ctk.CTkButton(self.btn_frame, text="STOP AI", fg_color="#555", state="disabled", command=self.stop_bot)
        self.stop_btn.pack(side="left", padx=10)

        self.force_frame = ctk.CTkFrame(self, fg_color="#2a2a2a", border_width=1, border_color="#555")
        self.force_frame.grid(row=6, column=0, pady=10, padx=20, sticky="ew")
        ctk.CTkLabel(self.force_frame, text="MANUAL COMMANDS", font=("Helvetica", 12, "bold"), text_color="gray").pack(pady=(5,0))
        
        self.force_btns = ctk.CTkFrame(self.force_frame, fg_color="transparent")
        self.force_btns.pack(pady=10)
        self.force_buy_btn = ctk.CTkButton(self.force_btns, text="⚡ FORCE BUY", width=100, fg_color="#28a745", hover_color="#1e7e34", command=lambda: self.trigger_manual_trade("BUY"))
        self.force_buy_btn.pack(side="left", padx=10)
        self.force_sell_btn = ctk.CTkButton(self.force_btns, text="⚡ FORCE SELL", width=100, fg_color="#dc3545", hover_color="#c82333", command=lambda: self.trigger_manual_trade("SELL"))
        self.force_sell_btn.pack(side="left", padx=10)
        
        self.secure_btn = ctk.CTkButton(self.force_btns, text="💰 SECURE PROFITS", width=120, fg_color="#ffc107", text_color="black", hover_color="#e0a800", command=self.secure_profits)
        self.secure_btn.pack(side="left", padx=10)

    def log_message(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_box.see("end"); self.log_box.configure(state="disabled")

    def kill_all_trades(self):
        self.log_message("⚠️ PANIC: Closing ALL positions (Winners & Losers)...")
        acc, pwd = self.acc_entry.get().strip(), self.pass_entry.get().strip()
        if mt5.initialize(login=int(acc), server=self.selected_server, password=pwd):
            positions = mt5.positions_get(symbol="XAUUSD")
            if positions:
                for p in positions:
                    t = mt5.symbol_info_tick(p.symbol)
                    mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume,
                                    "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                    "position": p.ticket, "price": t.bid if p.type == mt5.POSITION_TYPE_BUY else t.ask,
                                    "deviation": 50, "magic": 777777})
                self.log_message("✅ All open trades neutralized.")
            mt5.shutdown()

    def secure_profits(self):
        self.log_message("🔒 SECURE PROFITS: Hunting for winning trades...")
        acc, pwd = self.acc_entry.get().strip(), self.pass_entry.get().strip()
        if mt5.initialize(login=int(acc), server=self.selected_server, password=pwd):
            positions = mt5.positions_get(symbol="XAUUSD")
            closed_any = False
            if positions:
                for p in positions:
                    if p.profit > 0:  
                        t = mt5.symbol_info_tick(p.symbol)
                        res = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume,
                                        "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                        "position": p.ticket, "price": t.bid if p.type == mt5.POSITION_TYPE_BUY else t.ask,
                                        "deviation": 50, "magic": 777777})
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            self.log_message(f"✅ Secured Ticket #{p.ticket} for +${p.profit:.2f} profit!")
                            closed_any = True
                if not closed_any:
                    self.log_message("⚠️ No XAUUSD trades are currently in profit.")
            else:
                self.log_message("⚠️ No open trades found.")
            mt5.shutdown()

    def smart_close(self, position_type_to_close, reason):
        """Actively closes trades that are moving against the new market trend."""
        if not self.smartclose_var.get():
            return
            
        positions = mt5.positions_get(symbol="XAUUSD")
        if not positions: return

        for p in positions:
            if p.type == position_type_to_close:
                t = mt5.symbol_info_tick(p.symbol)
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": p.symbol,
                    "volume": p.volume,
                    "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": p.ticket,
                    "price": t.bid if p.type == mt5.POSITION_TYPE_BUY else t.ask,
                    "deviation": 50,
                    "magic": 777777
                }
                res = mt5.order_send(req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    pl_str = f"+${p.profit:.2f}" if p.profit >= 0 else f"-${abs(p.profit):.2f}"
                    self.log_message(f"🧠 SMART CLOSE: Killed Ticket #{p.ticket} ({pl_str}). Reason: {reason}")
                    if self.webhook_url: 
                        requests.post(self.webhook_url, json={"content": f"🧠 **Smart Close Triggered**\nTicket: #{p.ticket}\nResult: {pl_str}\nReason: {reason}"})

    def manage_breakeven(self):
        if not self.breakeven_var.get() or self.live_atr <= 0:
            return
            
        positions = mt5.positions_get(symbol="XAUUSD")
        if not positions: return

        activation_dist = self.live_atr * 1.0 

        for p in positions:
            if p.type == mt5.POSITION_TYPE_BUY:
                if (p.price_current - p.price_open) >= activation_dist:
                    if p.sl < p.price_open:  
                        req = {"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": "XAUUSD", "sl": p.price_open, "tp": p.tp}
                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            self.log_message(f"🛡️ AUTO-BREAKEVEN: Risk neutralized for BUY #{p.ticket}")
                            
            elif p.type == mt5.POSITION_TYPE_SELL:
                if (p.price_open - p.price_current) >= activation_dist:
                    if p.sl > p.price_open or p.sl == 0.0: 
                        req = {"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": "XAUUSD", "sl": p.price_open, "tp": p.tp}
                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            self.log_message(f"🛡️ AUTO-BREAKEVEN: Risk neutralized for SELL #{p.ticket}")

    def process_data(self, df):
        if df is None or df.empty or 'close' not in df.columns:
            return None

        df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['TR'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
        df['ATR'] = df['TR'].rolling(14).mean()
        
        df['Is_Green'] = df['close'] > df['open']
        df['Bullish_Engulfing'] = (df['Is_Green'] == True) & (df['Is_Green'].shift(1) == False) & \
                                  (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
        df['Bearish_Engulfing'] = (df['Is_Green'] == False) & (df['Is_Green'].shift(1) == True) & \
                                  (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
        return df

    def calculate_adaptive_lot(self, balance, sl_dist):
        try:
            risk_pct = float(self.risk_pct_entry.get()) / 100
            risk_usd = (balance / 58.0) * risk_pct
            sl_risk_per_lot = sl_dist * 100 
            lot = risk_usd / sl_risk_per_lot
            final_lot = max(0.01, round(lot, 2))
            self.lot_calc_lbl.configure(text=f"{final_lot}")
            return final_lot
        except:
            return 0.01

    def main_loop(self):
        while self.is_running:
            try:
                acc, pwd = self.acc_entry.get().strip(), self.pass_entry.get().strip()
                if mt5.initialize(login=int(acc), server=self.selected_server, password=pwd):
                    mt5.symbol_select("XAUUSD", True) 
                    info = mt5.account_info()
                    self.live_balance = info.balance  
                    
                    news = yf.Ticker("GC=F").news
                    sentiment = "NEUTRAL"
                    if news:
                        sia = SentimentIntensityAnalyzer()
                        score = sum(sia.polarity_scores(n.get('title',''))['compound'] for n in news[:5]) / 5
                        if score > 0.05: sentiment = "BULLISH"
                        elif score < -0.05: sentiment = "BEARISH"

                    rates_m15 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 100)
                    rates_m5 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 100)
                    
                    df_m15_raw = pd.DataFrame(rates_m15) if rates_m15 is not None else pd.DataFrame()
                    df_m5_raw = pd.DataFrame(rates_m5) if rates_m5 is not None else pd.DataFrame()

                    df_m15 = self.process_data(df_m15_raw)
                    df_m5 = self.process_data(df_m5_raw)
                    
                    if df_m15 is None or df_m5 is None:
                        self.log_message("⚠️ Data buffer empty. Waiting for broker to load candles...")
                        mt5.shutdown()
                        time.sleep(5)
                        continue
                    
                    m15_e9, m15_e21 = df_m15['EMA9'].iloc[-1], df_m15['EMA21'].iloc[-1]
                    macro_trend = "UPTREND" if m15_e9 > m15_e21 else "DOWNTREND"
                    
                    bull_engulf = df_m5['Bullish_Engulfing'].iloc[-1]
                    bear_engulf = df_m5['Bearish_Engulfing'].iloc[-1]
                    self.live_atr = df_m5['ATR'].iloc[-1] 
                    
                    candle_status = "Neutral"
                    if bull_engulf: candle_status = "Bull Engulfing"
                    if bear_engulf: candle_status = "Bear Engulfing"

                    self.balance_lbl.configure(text=f"{info.balance:,.2f}")
                    self.pl_lbl.configure(text=f"{info.profit:,.2f}", text_color="#28a745" if info.profit >= 0 else "#dc3545")
                    self.macro_lbl.configure(text=macro_trend, text_color="#28a745" if macro_trend == "UPTREND" else "#dc3545")
                    self.micro_lbl.configure(text=candle_status, text_color="#FFD700" if candle_status != "Neutral" else "white")
                    self.atr_lbl.configure(text=f"{self.live_atr:.2f}")

                    sl_dist = self.live_atr * float(self.atr_sl_entry.get())
                    tp_dist = self.live_atr * float(self.atr_tp_entry.get())
                    lot = self.calculate_adaptive_lot(info.balance, sl_dist)

                    self.log_message(f"--- 📡 MTF PULSE CHECK ---")
                    
                    # --- CORE MTF EXECUTION WITH SMART CLOSE ---
                    if bull_engulf and macro_trend == "UPTREND" and sentiment != "BEARISH":
                        self.smart_close(mt5.POSITION_TYPE_SELL, "Trend Reversed UP (Bullish Engulfing)")
                        self.log_message(f"🔥 MTF ALIGNMENT: M15 Uptrend + M5 Bullish Engulfing.")
                        self.execute_trade("BUY", lot, sl_dist, tp_dist, is_manual=False)
                        
                    elif bear_engulf and macro_trend == "DOWNTREND" and sentiment != "BULLISH":
                        self.smart_close(mt5.POSITION_TYPE_BUY, "Trend Reversed DOWN (Bearish Engulfing)")
                        self.log_message(f"🩸 MTF ALIGNMENT: M15 Downtrend + M5 Bearish Engulfing.")
                        self.execute_trade("SELL", lot, sl_dist, tp_dist, is_manual=False)
                        
                    else:
                        self.log_message(f"👀 M15: {macro_trend} | M5: {candle_status}. Waiting...")

                    self.manage_breakeven()
                    
                    mt5.shutdown()
                else: self.log_message(f"❌ MT5 Connection Error")
            except Exception as e: self.log_message(f"🚨 Scan Error: {e}")
            time.sleep(15)

    def trigger_manual_trade(self, direction):
        if self.live_balance <= 0 or self.live_atr <= 0:
            self.log_message("⚠️ Cannot Force Trade. Turn on 'START MTF SCAN' first to pull market data.")
            return
            
        acc, pwd = self.acc_entry.get().strip(), self.pass_entry.get().strip()
        if mt5.initialize(login=int(acc), server=self.selected_server, password=pwd):
            sl_dist = self.live_atr * float(self.atr_sl_entry.get())
            tp_dist = self.live_atr * float(self.atr_tp_entry.get())
            lot = self.calculate_adaptive_lot(self.live_balance, sl_dist)
            
            self.execute_trade(direction, lot, sl_dist, tp_dist, is_manual=True)
            mt5.shutdown()

    def execute_trade(self, type_str, lot, sl_dist, tp_dist, is_manual):
        mt5.symbol_select("XAUUSD", True)
        t = mt5.symbol_info_tick("XAUUSD")
        if not t:
            self.log_message("❌ Execution Failed: Could not fetch live price.")
            return

        price = t.ask if type_str == "BUY" else t.bid
        sl = price - sl_dist if type_str == "BUY" else price + sl_dist
        tp = price + tp_dist if type_str == "BUY" else price - tp_dist
        
        origin = "⚡ MANUAL OVERRIDE" if is_manual else "🤖 MTF AI EXECUTION"
        self.log_message(f"{origin}: {type_str} {lot} Lots at {price:.2f}")
        self.log_message(f"🛡️ Hard Stop Loss set to: {sl:.2f}")
        self.log_message(f"🎯 Take Profit set to: {tp:.2f}")

        req = {
            "action": mt5.TRADE_ACTION_DEAL, 
            "symbol": "XAUUSD", 
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if type_str == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price, 
            "sl": sl, 
            "tp": tp, 
            "deviation": 50,  
            "magic": 777777   
        }
        res = mt5.order_send(req)
        
        if res is None:
            self.log_message(f"❌ FAILED: Exness rejected the order format. Code: {mt5.last_error()}")
        elif res.retcode == mt5.TRADE_RETCODE_DONE:
            self.log_message(f"✅ SUCCESS: Broker accepted ticket #{res.order}")
            if self.webhook_url: 
                requests.post(self.webhook_url, json={"content": f"**{origin}**\nExecuted {type_str} {lot} Lots on XAUUSD.\nEntry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}"})
        else:
            self.log_message(f"❌ FAILED: Broker rejected trade. Code: {res.retcode}")

    def create_input(self, frame, label, r, c, default=""):
        ctk.CTkLabel(frame, text=label).grid(row=r, column=c, padx=5, pady=5)
        entry = ctk.CTkEntry(frame, width=90); entry.insert(0, default)
        entry.grid(row=r, column=c+1, padx=5, pady=5, sticky="ew")
        return entry

    def create_stat(self, frame, label, val, r, c):
        ctk.CTkLabel(frame, text=label, font=("Helvetica", 9, "bold")).grid(row=r*2, column=c)
        lbl = ctk.CTkLabel(frame, text=val, font=("Consolas", 15, "bold"))
        lbl.grid(row=r*2+1, column=c); return lbl

    def toggle_password(self): self.pass_entry.configure(show="" if self.show_pass_var.get() else "*")
    def toggle_on_top(self): self.attributes("-topmost", self.on_top_var.get())
    def open_server_picker(self): ServerPicker(self, self.set_server)
    def set_server(self, s): self.selected_server = s; self.server_btn.configure(text=f"🌐 Server: {s}")
    def open_webhook_config(self): 
        d = ctk.CTkInputDialog(text="Webhook:", title="Discord")
        url = d.get_input()
        if url: self.webhook_url = url.strip()

    def save_config(self):
        c = {"acc": self.acc_entry.get(), "srv": self.selected_server, "webhook": self.webhook_url}
        with open(CONFIG_FILE, "w") as f: json.dump(c, f)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                c = json.load(f); self.selected_server = c.get("srv", "Exness-MT5Trial17")
                self.webhook_url = c.get("webhook", "")
                self.server_btn.configure(text=f"🌐 Server: {self.selected_server}")

    def start_bot(self):
        self.save_config(); self.is_running = True
        self.start_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        self.log_message("🚀 MTF SCANNER WAKING UP...")
        threading.Thread(target=self.main_loop, daemon=True).start()

    def stop_bot(self):
        self.is_running = False; self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled"); self.log_message("🛑 MTF SCANNER STOPPED.")

if __name__ == "__main__":
    UltimateAutotrader().mainloop()