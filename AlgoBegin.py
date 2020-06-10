import alpaca_trade_api as tradeapi
import json
import requests
import websocket
import os
import pandas as pd
import datetime

socket = "wss://alpaca.socket.polygon.io/stocks"
TICKERS = "AM.MSFT"
API_KEY = os.getenv('APCA_API_KEY_ID')
# HEADERS = {'APCA_API_KEY_ID': os.getenv("APCA_API_KEY_ID"), 'APCA_API_SECRET_KEY': os.getenv("APCA_API_SECRET_KEY")}


# "ev": "AM",             // Event Type ( A = Second Agg, AM = Minute Agg )
#     "sym": "MSFT",          // Symbol Ticker
#     "v": 10204,             // Tick Volume
#     "av": 200304,           // Accumulated Volume ( Today )
#     "op": 114.04,           // Todays official opening price
#     "vw": 114.4040,         // VWAP (Volume Weighted Average Price)
#     "o": 114.11,            // Tick Open Price
#     "c": 114.14,            // Tick Close Price
#     "h": 114.19,            // Tick High Price
#     "l": 114.09,            // Tick Low Price
#     "a": 114.1314,          // Tick Average / VWAP Price
#     "s": 1536036818784,     // Tick Start Timestamp ( Unix MS )
#     "e": 1536036818784,     // Tick End Timestamp ( Unix MS )
stocks = ["MSFT", "APPL", "GOOG"]
df = pd.DataFrame(columns=(['Time'] + stocks))


def on_open(webs):
    auth_data = {
        "action": "auth",
        "params": API_KEY
    }
    webs.send(json.dumps(auth_data))
    channel_data = {
        "action": "subscribe",
        "params": TICKERS
    }
    print("Connection opened.")
    webs.send(json.dumps(channel_data))


def on_message(webs, message):
    current_line = json.loads(message)
    time = current_line["time"]
    ticker = current_line["sym"]
    price = current_line["vw"]
    last_time = df['time'].iloc[-1]
    if last_time != time:
            df.append({"Time": time, ticker: price})
            if df.size < 60:
                df.drop(df.index[:1], inplace=True)
    else:
        df.loc[-1, ticker] = price
    print(df)


def on_close(webs):
    print("Connection closed.")


ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
ws.run_forever()
