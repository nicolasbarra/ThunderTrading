import alpaca_trade_api as tradeapi
import json
import websocket
import os
import pandas as pd
import numpy as np


socket = "wss://alpaca.socket.polygon.io/stocks"
TICKERS = "AM.MSFT, AM.AAPL, AM.F"
stock_weights = [(1/4), (1/4), (2/4)]
stocks = ["Time", "MSFT", "AAPL", "F"]
df = pd.DataFrame(columns=stocks)


def _update_weight(window, weights, np_asset_prices, n_iterations, tau, number_of_assets, epsilon):
    """
    Predicts the next time's portfolio weight.

    :return: (np.array) Predicted weights.
    """

    # Set the current predicted relatives value.
    current_prediction = _calculate_predicted_relatives(np_asset_prices, window, n_iterations, tau)

    # Set the deviation from the mean of current prediction.

    current_prediction = current_prediction.iloc[1:]

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
    # print(new_weights)
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


def _calculate_predicted_relatives(np_asset_prices, window, n_iteration, tau):
    """
    Calculates the predicted relatives using l1 median.

    :return: (np.array) Predicted relatives using l1 median.
    """
    # Calculate the L1 median of the price window.
    price_window = np_asset_prices
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
    predicted_relatives = curr_prediction / price_window.iloc[-1]

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
    API_KEY = os.getenv('APCA_API_KEY_ID')
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
    if len(df.index) > 0 and not df.iloc[-1].isnull().values.any():
        global stock_weights
        stock_weights = _update_weight(tau=0.001, window=7, n_iterations=200, epsilon=20, weights=stock_weights,
                                       np_asset_prices=df,
                                       number_of_assets=len(stock_weights))
        print("=========")


def on_close(webs):
    print("Connection closed.")
    print(df)


ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
ws.run_forever()
