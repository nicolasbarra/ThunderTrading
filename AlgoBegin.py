import alpaca_trade_api as tradeapi
import json
import requests
import websocket
import os

socket = "wss://alpaca.socket.polygon.io/stocks"
TICKERS = "AM.MSFT"
API_KEY = os.getenv('APCA_API_KEY_ID')
# HEADERS = {'APCA_API_KEY_ID': os.getenv("APCA_API_KEY_ID"), 'APCA_API_SECRET_KEY': os.getenv("APCA_API_SECRET_KEY")}


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
    print(current_line)


def on_close(webs):
    print("Connection closed.")


ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
ws.run_forever()
