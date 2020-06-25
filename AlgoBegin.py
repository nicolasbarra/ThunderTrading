import alpaca_trade_api as tradeapi
import json
import requests
import websocket
import os
import pandas as pd
import numpy as np
import datetime

socket = "wss://alpaca.socket.polygon.io/stocks"
TICKERS = "AM.MSFT, AM.AAPL, AM.F"
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
stocks = ["Time", "MSFT", "AAPL", "F"]
df = pd.DataFrame(columns=stocks)
print("this is df", df)


def _update_weight(window, weights, np_asset_prices, n_iterations, tau, time, number_of_assets, epsilon):
    """
    Predicts the next time's portfolio weight.

    :param time: (int) Current time period.
    :return: (np.array) Predicted weights.
    """
    # Until the relative time window, return original weights.
    if time < window - 1:
        return weights

    # Set the current predicted relatives value.
    current_prediction = _calculate_predicted_relatives(np_asset_prices, window, n_iterations, tau, time)

    # Set the deviation from the mean of current prediction.
    predicted_deviation = current_prediction - np.ones(number_of_assets) * np.mean(
        current_prediction)

    # Calculate alpha, the lagrangian multiplier.
    norm2 = np.linalg.norm(predicted_deviation, ord=1) ** 2

    # If norm2 is zero, return previous weights.
    if norm2 == 0:
        return weights
    alpha = np.minimum(0, (current_prediction * weights - epsilon) / norm2)

    # Update new weights.
    new_weights = weights - alpha * predicted_deviation

    # Project to simplex domain.
    new_weights = _simplex_projection(new_weights)

    return new_weights

def _simplex_projection(weight):
    """
    Calculates the simplex projection of weights.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

    :param weight: (np.array) Weight to be projected onto the simplex domain.
    :return: (np.array) Simplex projection of the original weight.
    """
    # Sort in descending order.
    _mu = np.sort(weight)[::-1]

    # Calculate adjusted sum.
    adjusted_sum = np.cumsum(_mu) - 1
    j = np.arange(len(weight)) + 1

    # Determine the conditions.
    cond = _mu - adjusted_sum / j > 0

    # If all conditions are false, return uniform weight.
    if not cond.any():
        uniform_weight = np.ones(len(weight)) / len(weight)
        return uniform_weight

    # Define max rho.
    rho = float(j[cond][-1])

    # Define theta.
    theta = adjusted_sum[cond][-1] / rho

    # Calculate new weight.
    new_weight = np.maximum(weight - theta, 0)
    return new_weight

def _calculate_predicted_relatives(np_asset_prices, window, n_iteration, tau, time):
    """
    Calculates the predicted relatives using l1 median.

    :param time: (int) Current time.
    :return: (np.array) Predicted relatives using l1 median.
    """
    # Calculate the L1 median of the price window.
    price_window = np_asset_prices[time - window + 1:time + 1]
    curr_prediction = np.median(price_window, axis=0)

    # Iterate until the maximum iteration allowed.
    for _ in range(n_iteration - 1):
        prev_prediction = curr_prediction
        # Transform mu according the Modified Weiszfeld Algorithm
        curr_prediction = _transform(curr_prediction, price_window)

        # If null value or condition is satisfied, break.
        if curr_prediction.size == 0 or np.linalg.norm(prev_prediction - curr_prediction, ord=1) \
                <= tau * np.linalg.norm(curr_prediction, ord=1):
            curr_prediction = prev_prediction
            break

    # Divide by the current time's price.
    predicted_relatives = curr_prediction / price_window[-1]

    return predicted_relatives


def _transform(old_mu, price_window):
    """
    Calculates L1 median approximation by using the Modified Weiszfeld Algorithm.

    :param old_mu: (np.array) Current value of the predicted median value.
    :param price_window: (np.array) A window of prices provided by the user.
    :return: (np.array) New updated l1 median approximation.
    """
    # Calculate the difference set.
    diff = price_window - old_mu

    # Remove rows with all zeros.
    non_mu = diff[~np.all(diff == 0, axis=1)]

    # Edge case for identical price windows.
    if non_mu.shape[0] == 0:
        return non_mu

    # Number of zeros.
    n_zero = diff.shape[0] - non_mu.shape[0]

    # Calculate eta.
    eta = 0 if n_zero == 0 else 1

    # Calculate l1 norm of non_mu.
    l1_norm = np.linalg.norm(non_mu, ord=1, axis=1)

    # Calculate tilde.
    tilde = 1 / np.sum(1 / l1_norm) * np.sum(np.divide(non_mu.T, l1_norm), axis=1)

    # Calculate gamma.
    gamma = np.linalg.norm(
        np.sum(np.apply_along_axis(lambda x: x / np.linalg.norm(x, ord=1), 1, non_mu), axis=0),
        ord=1)

    # Calculate next_mu value.
    with np.errstate(invalid='ignore'):
        next_mu = np.maximum(0, 1 - eta / gamma) * tilde + np.minimum(1, eta / gamma) * old_mu
    return tilde if eta == 0 else next_mu


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
    current_line = json.loads(message)[0]
    print(current_line)
    time = current_line['e']
    ticker = current_line['sym']
    price = current_line['vw']
    if df.size == 0:
        df.loc[0, "Time"] = time
        df.loc[0, ticker] = price
    else:
        last_time = df['Time'].iloc[-1]
        if last_time == time:
            df.loc[df.index.max(), ticker] = price
        else:
            new_index = df.index.max() + 1
            df.loc[new_index, "Time"] = time
            df.loc[new_index, ticker] = price
            if len(df.index) > 20:
                df.drop(df.index[0], inplace=True)

def on_close(webs):
    print("Connection closed.")
    print(df)

ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
ws.run_forever()
message_array = [
    {'ev': 'A', 'sym': 'AAPL', 'v': 1778, 'av': 37672744, 'op': 351.46, 'vw': 351.1188, 'o': 351.12, 'c': 351.11,
     'h': 351.125, 'l': 351.11, 'a': 350.7528, 'z': 127, 'n': 1, 's': 1592337499000, 'e': 1592337500000},
    {'ev': 'A', 'sym': 'F', 'v': 1001, 'av': 105498351, 'op': 6.87, 'vw': 6.5562, 'o': 6.5562, 'c': 6.5562, 'h': 6.5562,
     'l': 6.5562, 'a': 6.6447, 'z': 500, 'n': 1, 's': 1592337499000, 'e': 1592337500000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 1976, 'av': 35429110, 'op': 192.89, 'vw': 193.2415, 'o': 193.24, 'c': 193.24,
     'h': 193.25, 'l': 193.235, 'a': 193.4725, 'z': 65, 'n': 1, 's': 1592337499000, 'e': 1592337500000},
    {'ev': 'A', 'sym': 'F', 'v': 23023, 'av': 105521374, 'op': 6.87, 'vw': 6.5512, 'o': 6.555, 'c': 6.5572, 'h': 6.56,
     'l': 6.55, 'a': 6.6447, 'z': 1918, 'n': 1, 's': 1592337500000, 'e': 1592337501000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 596, 'av': 37673340, 'op': 351.46, 'vw': 351.0983, 'o': 351.11, 'c': 351.09,
     'h': 351.11, 'l': 351.08, 'a': 350.7528, 'z': 59, 'n': 1, 's': 1592337500000, 'e': 1592337501000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 2051, 'av': 35431161, 'op': 192.89, 'vw': 193.2267, 'o': 193.2384, 'c': 193.225,
     'h': 193.2384, 'l': 193.22, 'a': 193.4725, 'z': 82, 'n': 1, 's': 1592337500000, 'e': 1592337501000},
    {'ev': 'A', 'sym': 'F', 'v': 11001, 'av': 105532375, 'op': 6.87, 'vw': 6.5507, 'o': 6.56, 'c': 6.555, 'h': 6.56,
     'l': 6.55, 'a': 6.6447, 'z': 1833, 'n': 1, 's': 1592337501000, 'e': 1592337502000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 600, 'av': 37673940, 'op': 351.46, 'vw': 351.0879, 'o': 351.103, 'c': 351.07,
     'h': 351.103, 'l': 351.07, 'a': 350.7528, 'z': 150, 'n': 1, 's': 1592337501000, 'e': 1592337502000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 1803, 'av': 35432964, 'op': 192.89, 'vw': 193.2198, 'o': 193.22, 'c': 193.22,
     'h': 193.22, 'l': 193.21, 'a': 193.4725, 'z': 100, 'n': 1, 's': 1592337501000, 'e': 1592337502000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 8642, 'av': 35441606, 'op': 192.89, 'vw': 193.252, 'o': 193.22, 'c': 193.26,
     'h': 193.28, 'l': 193.22, 'a': 193.4724, 'z': 110, 'n': 1, 's': 1592337502000, 'e': 1592337503000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 7359, 'av': 37681299, 'op': 351.46, 'vw': 351.1323, 'o': 351.11, 'c': 351.165,
     'h': 351.165, 'l': 351.1, 'a': 350.7528, 'z': 75, 'n': 1, 's': 1592337502000, 'e': 1592337503000},
    {'ev': 'A', 'sym': 'F', 'v': 1301, 'av': 105533676, 'op': 6.87, 'vw': 6.5569, 'o': 6.555, 'c': 6.56, 'h': 6.56,
     'l': 6.555, 'a': 6.6447, 'z': 162, 'n': 1, 's': 1592337502000, 'e': 1592337503000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 3043, 'av': 37684342, 'op': 351.46, 'vw': 351.1758, 'o': 351.14, 'c': 351.15,
     'h': 351.21, 'l': 351.14, 'a': 350.7529, 'z': 64, 'n': 1, 's': 1592337503000, 'e': 1592337504000},
    {'ev': 'A', 'sym': 'F', 'v': 48460, 'av': 105582136, 'op': 6.87, 'vw': 6.559, 'o': 6.56, 'c': 6.56, 'h': 6.56,
     'l': 6.555, 'a': 6.6446, 'z': 637, 'n': 1, 's': 1592337503000, 'e': 1592337504000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 6008, 'av': 35447614, 'op': 192.89, 'vw': 193.2565, 'o': 193.25, 'c': 193.25,
     'h': 193.27, 'l': 193.25, 'a': 193.4724, 'z': 101, 'n': 1, 's': 1592337503000, 'e': 1592337504000},
    {'ev': 'A', 'sym': 'F', 'v': 10600, 'av': 105592736, 'op': 6.87, 'vw': 6.5594, 'o': 6.555, 'c': 6.555, 'h': 6.56,
     'l': 6.555, 'a': 6.6446, 'z': 757, 'n': 1, 's': 1592337504000, 'e': 1592337505000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 2938, 'av': 35450552, 'op': 192.89, 'vw': 193.2553, 'o': 193.24, 'c': 193.26,
     'h': 193.26, 'l': 193.24, 'a': 193.4724, 'z': 101, 'n': 1, 's': 1592337504000, 'e': 1592337505000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 1644, 'av': 37685986, 'op': 351.46, 'vw': 351.146, 'o': 351.15, 'c': 351.13,
     'h': 351.15, 'l': 351.13, 'a': 350.7529, 'z': 68, 'n': 1, 's': 1592337504000, 'e': 1592337505000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 6083, 'av': 37692069, 'op': 351.46, 'vw': 351.0884, 'o': 351.145, 'c': 351.07,
     'h': 351.145, 'l': 351.03, 'a': 350.753, 'z': 114, 'n': 1, 's': 1592337505000, 'e': 1592337506000},
    {'ev': 'A', 'sym': 'F', 'v': 753, 'av': 105593489, 'op': 6.87, 'vw': 6.5594, 'o': 6.5572, 'c': 6.56, 'h': 6.56,
     'l': 6.5572, 'a': 6.6446, 'z': 150, 'n': 1, 's': 1592337505000, 'e': 1592337506000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 9605, 'av': 35460157, 'op': 192.89, 'vw': 193.241, 'o': 193.25, 'c': 193.25,
     'h': 193.25, 'l': 193.23, 'a': 193.4723, 'z': 145, 'n': 1, 's': 1592337505000, 'e': 1592337506000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 355, 'av': 37692424, 'op': 351.46, 'vw': 351.0753, 'o': 351.075, 'c': 351.075,
     'h': 351.075, 'l': 351.075, 'a': 350.753, 'z': 59, 'n': 1, 's': 1592337506000, 'e': 1592337507000},
    {'ev': 'A', 'sym': 'F', 'v': 2000, 'av': 105595489, 'op': 6.87, 'vw': 6.56, 'o': 6.56, 'c': 6.56, 'h': 6.56,
     'l': 6.5594, 'a': 6.6446, 'z': 142, 'n': 1, 's': 1592337506000, 'e': 1592337507000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 2180, 'av': 35462337, 'op': 192.89, 'vw': 193.2662, 'o': 193.26, 'c': 193.27,
     'h': 193.27, 'l': 193.26, 'a': 193.4723, 'z': 77, 'n': 1, 's': 1592337506000, 'e': 1592337507000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 2696, 'av': 35465033, 'op': 192.89, 'vw': 193.2771, 'o': 193.28, 'c': 193.28,
     'h': 193.285, 'l': 193.27, 'a': 193.4723, 'z': 99, 'n': 1, 's': 1592337507000, 'e': 1592337508000},
    {'ev': 'A', 'sym': 'F', 'v': 2836, 'av': 105598325, 'op': 6.87, 'vw': 6.5569, 'o': 6.555, 'c': 6.555, 'h': 6.56,
     'l': 6.555, 'a': 6.6446, 'z': 202, 'n': 1, 's': 1592337507000, 'e': 1592337508000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 1511, 'av': 37693935, 'op': 351.46, 'vw': 351.0633, 'o': 351.085, 'c': 351.0602,
     'h': 351.085, 'l': 351.06, 'a': 350.753, 'z': 167, 'n': 1, 's': 1592337507000, 'e': 1592337508000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 5127, 'av': 35470160, 'op': 192.89, 'vw': 193.2826, 'o': 193.28, 'c': 193.29,
     'h': 193.3, 'l': 193.2793, 'a': 193.4723, 'z': 205, 'n': 1, 's': 1592337508000, 'e': 1592337509000},
    {'ev': 'A', 'sym': 'F', 'v': 1251, 'av': 105599576, 'op': 6.87, 'vw': 6.5595, 'o': 6.555, 'c': 6.56, 'h': 6.56,
     'l': 6.555, 'a': 6.6446, 'z': 139, 'n': 1, 's': 1592337508000, 'e': 1592337509000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 2739, 'av': 37696674, 'op': 351.46, 'vw': 351.1194, 'o': 351.09, 'c': 351.13,
     'h': 351.13, 'l': 351.09, 'a': 350.753, 'z': 101, 'n': 1, 's': 1592337508000, 'e': 1592337509000},
    {'ev': 'A', 'sym': 'F', 'v': 1285, 'av': 105600861, 'op': 6.87, 'vw': 6.5532, 'o': 6.5572, 'c': 6.55, 'h': 6.56,
     'l': 6.55, 'a': 6.6446, 'z': 257, 'n': 1, 's': 1592337509000, 'e': 1592337510000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 952, 'av': 35471112, 'op': 192.89, 'vw': 193.2995, 'o': 193.29, 'c': 193.3056,
     'h': 193.3056, 'l': 193.29, 'a': 193.4722, 'z': 79, 'n': 1, 's': 1592337509000, 'e': 1592337510000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 599, 'av': 37697273, 'op': 351.46, 'vw': 351.1405, 'o': 351.15, 'c': 351.135,
     'h': 351.15, 'l': 351.12, 'a': 350.753, 'z': 66, 'n': 1, 's': 1592337509000, 'e': 1592337510000},
    {'ev': 'A', 'sym': 'F', 'v': 4461, 'av': 105605322, 'op': 6.87, 'vw': 6.5581, 'o': 6.555, 'c': 6.555, 'h': 6.56,
     'l': 6.555, 'a': 6.6446, 'z': 278, 'n': 1, 's': 1592337510000, 'e': 1592337511000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 4744, 'av': 37702017, 'op': 351.46, 'vw': 351.154, 'o': 351.1374, 'c': 351.1962,
     'h': 351.2, 'l': 351.12, 'a': 350.7531, 'z': 89, 'n': 1, 's': 1592337510000, 'e': 1592337511000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 7190, 'av': 35478302, 'op': 192.89, 'vw': 193.3079, 'o': 193.31, 'c': 193.29,
     'h': 193.33, 'l': 193.29, 'a': 193.4722, 'z': 121, 'n': 1, 's': 1592337510000, 'e': 1592337511000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 2793, 'av': 37704810, 'op': 351.46, 'vw': 351.2325, 'o': 351.2104, 'c': 351.23,
     'h': 351.24, 'l': 351.2104, 'a': 350.7531, 'z': 96, 'n': 1, 's': 1592337511000, 'e': 1592337512000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 7587, 'av': 35485889, 'op': 192.89, 'vw': 193.3335, 'o': 193.3, 'c': 193.36,
     'h': 193.36, 'l': 193.295, 'a': 193.4722, 'z': 185, 'n': 1, 's': 1592337511000, 'e': 1592337512000},
    {'ev': 'A', 'sym': 'F', 'v': 5518, 'av': 105610840, 'op': 6.87, 'vw': 6.5572, 'o': 6.555, 'c': 6.56, 'h': 6.56,
     'l': 6.555, 'a': 6.6446, 'z': 306, 'n': 1, 's': 1592337511000, 'e': 1592337512000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 11399, 'av': 37716209, 'op': 351.46, 'vw': 351.2747, 'o': 351.225, 'c': 351.25,
     'h': 351.37, 'l': 351.21, 'a': 350.7532, 'z': 61, 'n': 1, 's': 1592337512000, 'e': 1592337513000},
    {'ev': 'A', 'sym': 'F', 'v': 557, 'av': 105611397, 'op': 6.87, 'vw': 6.5595, 'o': 6.56, 'c': 6.56, 'h': 6.56,
     'l': 6.56, 'a': 6.6446, 'z': 92, 'n': 1, 's': 1592337512000, 'e': 1592337513000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 14161, 'av': 35500050, 'op': 192.89, 'vw': 193.3583, 'o': 193.35, 'c': 193.36,
     'h': 193.41, 'l': 193.34, 'a': 193.4721, 'z': 117, 'n': 1, 's': 1592337512000, 'e': 1592337513000},
    {'ev': 'A', 'sym': 'AAPL', 'v': 5308, 'av': 37721517, 'op': 351.46, 'vw': 351.2547, 'o': 351.27, 'c': 351.2,
     'h': 351.28, 'l': 351.2, 'a': 350.7533, 'z': 120, 'n': 1, 's': 1592337513000, 'e': 1592337514000},
    {'ev': 'A', 'sym': 'F', 'v': 3000, 'av': 105614397, 'op': 6.87, 'vw': 6.555, 'o': 6.555, 'c': 6.555, 'h': 6.555,
     'l': 6.555, 'a': 6.6446, 'z': 3000, 'n': 1, 's': 1592337513000, 'e': 1592337514000},
    {'ev': 'A', 'sym': 'MSFT', 'v': 1196, 'av': 35501246, 'op': 192.89, 'vw': 193.3669, 'o': 193.36, 'c': 193.38,
     'h': 193.38, 'l': 193.36, 'a': 193.4721, 'z': 56, 'n': 1, 's': 1592337513000, 'e': 1592337514000}]

# for current_line in message_array:
#     time = current_line['e']
#     ticker = current_line['sym']
#     price = current_line['vw']
#     print("ticker", ticker)
#     if df.size == 0:
#         df.loc[0, "Time"] = time
#         df.loc[0, ticker] = price
#     else:
#         last_time = df['Time'].iloc[-1]
#         if last_time == time:
#             df.loc[df.index.max(), ticker] = price
#         else:
#             new_index = df.index.max() + 1
#             df.loc[new_index, "Time"] = time
#             df.loc[new_index, ticker] = price
#             if len(df.index) > 10:
#                 df.drop(df.index[0], inplace=True)
#
# print(df)
# print(df.columns)
