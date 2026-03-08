import customtkinter as ctk
import MetaTrader5 as mt5
import pandas as pd
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FullyAutomatedBot(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Fully Automated Exness Bot")
        self.geometry("600x680")
        self.resizable(False, False)

        self.title_label = ctk.CTkLabel(self, text="AI Autotrader (XAUUSD)", font=("Helvetica", 22, "bold"))
        self.title_label.pack(pady=(15, 5))

        # --- RISK MANAGEMENT SETTINGS PANEL ---
        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.pack(pady=10, padx=20, fill="x")
        
        # Lot Size
        self.lot_label = ctk.CTkLabel(self.settings_frame, text="Lot Size:")
        self.lot_label.grid(row=0, column=0, padx=10, pady=10)
        self.lot_entry = ctk.CTkEntry(self.settings_frame, width=80)
        self.lot_entry.insert(0, "0.01")
        self.lot_entry.grid(row=0, column=1, padx=10, pady=10)

        # Stop Loss ($ distance)
        self.sl_label = ctk.CTkLabel(self.settings_frame, text="Stop Loss ($ drop):")
        self.sl_label.grid(row=0, column=2, padx=10, pady=10)
        self.sl_entry = ctk.CTkEntry(self.settings_frame, width=80)
        self.sl_entry.insert(0, "5.00")
        self.sl_entry.grid(row=0, column=3, padx=10, pady=10)

        # Take Profit ($ distance)
        self.tp_label = ctk.CTkLabel(self.settings_frame, text="Take Profit ($ gain):")
        self.tp_label.grid(row=0, column=4, padx=10, pady=10)
        self.tp_entry = ctk.CTkEntry(self.settings_frame, width=80)
        self.tp_entry.insert(0, "10.00")
        self.tp_entry.grid(row=0, column=5, padx=10, pady=10)

        # --- DISPLAY BOX ---
        self.result_box = ctk.CTkTextbox(self, width=560, height=350, font=("Consolas", 13))
        self.result_box.pack(pady=10)
        self.result_box.insert("0.0", "System Ready.\nWarning: Executing this will place live trades.\nPress 'Scan & Auto-Trade'...")
        self.result_box.configure(state="disabled")

        # --- ACTION BUTTON ---
        self.analyze_btn = ctk.CTkButton(
            self, 
            text="Scan & Auto-Trade", 
            font=("Helvetica", 14, "bold"), 
            height=40,
            fg_color="#8B0000", # Dark red warning color
            hover_color="#600000",
            command=self.run_bot
        )
        self.analyze_btn.pack(pady=15)

    def analyze_news_sentiment(self):
        gold_ticker = yf.Ticker("GC=F")
        news_items = gold_ticker.news
        if not news_items: return "No news.", 0.0

        sia = SentimentIntensityAnalyzer()
        total_score = sum(sia.polarity_scores(item.get('title', ''))['compound'] for item in news_items[:5])
        avg_score = total_score / min(5, len(news_items))
        
        if avg_score > 0.05: return "BULLISH (Positive)", "BULLISH"
        elif avg_score < -0.05: return "BEARISH (Negative)", "BEARISH"
        return "NEUTRAL", "NEUTRAL"

    def execute_trade(self, order_type_string, current_price):
        """Builds and sends the trade request to Exness MT5"""
        symbol = "XAUUSD"
        lot = float(self.lot_entry.get())
        sl_dist = float(self.sl_entry.get())
        tp_dist = float(self.tp_entry.get())

        # Grab exact live broker prices
        tick = mt5.symbol_info_tick(symbol)
        
        if "BUY" in order_type_string:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = price + sl_dist
            tp = price - tp_dist

        # Build the exact order dictionary required by MT5
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20, # Allowed slippage in points
            "magic": 777777, # Custom ID for your bot's trades
            "comment": "AI Hybrid Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC, # Standard for Exness
        }

        # Send the order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return f"❌ ORDER FAILED: {result.comment}"
        else:
            return f"✅ SUCCESS: {order_type_string} executed at {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}"

    def run_bot(self):
        self.analyze_btn.configure(text="Processing...", state="disabled")
        self.update() 

        try:
            # 1. NLP News Brain
            news_display, sentiment = self.analyze_news_sentiment()

            # 2. MT5 Math Initialization
            if not mt5.initialize(): raise Exception("MT5 not connected.")
            
            rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 1000)
            if rates is None: raise Exception("Failed to pull XAUUSD data.")

            df = pd.DataFrame(rates)
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()

            current_fast = df['SMA_10'].iloc[-1]
            current_slow = df['SMA_50'].iloc[-1]
            prev_fast = df['SMA_10'].iloc[-2]
            prev_slow = df['SMA_50'].iloc[-2]
            current_price = df['close'].iloc[-1]

            # 3. Math Logic
            tech_signal = "HOLD"
            if prev_fast <= prev_slow and current_fast > current_slow: tech_signal = "BUY"
            elif prev_fast >= prev_slow and current_fast < current_slow: tech_signal = "SELL"

            # 4. Master Verdict & Execution Loop
            execution_log = "No trade criteria met."
            if tech_signal == "BUY" and sentiment == "BULLISH":
                master_verdict = "🔥 STRONG BUY DETECTED"
                execution_log = self.execute_trade("BUY", current_price)
            elif tech_signal == "SELL" and sentiment == "BEARISH":
                master_verdict = "🩸 STRONG SELL DETECTED"
                execution_log = self.execute_trade("SELL", current_price)
            else:
                master_verdict = f"⚠️ NO TRADE: Tech ({tech_signal}) & News ({sentiment}) do not match."

            # 5. Output Screen
            output = (
                f"=== EXNESS ALGO REPORT ===\n"
                f"Asset: Gold (XAUUSD) @ ${current_price:.2f}\n"
                f"Technical Signal: {tech_signal}\n"
                f"News Sentiment: {sentiment}\n\n"
                f"=== ALGO VERDICT ===\n"
                f"{master_verdict}\n\n"
                f"=== EXECUTION STATUS ===\n"
                f"{execution_log}\n"
            )

            self.result_box.configure(state="normal")
            self.result_box.delete("0.0", "end")
            self.result_box.insert("0.0", output)
            self.result_box.configure(state="disabled")

        except Exception as e:
            self.result_box.configure(state="normal")
            self.result_box.delete("0.0", "end")
            self.result_box.insert("0.0", f"SYSTEM ERROR:\n{e}")
            self.result_box.configure(state="disabled")
            
        finally:
            mt5.shutdown()
            self.analyze_btn.configure(text="Scan & Auto-Trade", state="normal")

if __name__ == "__main__":
    app = FullyAutomatedBot()
    app.mainloop()