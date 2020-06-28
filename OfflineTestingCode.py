from RMRStrat import df
from RMRStrat import _update_weight
from TestingDataMessage import message_array
from TestingDataMinute import minute_data

for current_line in minute_data:
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
        stock_weights = _update_weight(tau=0.001, window=7, n_iterations=200, epsilon=20, weights=stock_weights,
                                       np_asset_prices=df,
                                       number_of_assets=len(stock_weights))
        print("=========")

print(stock_weights)
