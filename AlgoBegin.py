import alpaca_trade_api as tradeapi

api = tradeapi.REST()

aapl = api.polygon.historic_agg_v2('AAPL', 1, 'day', _from='2019-01-01', to='2019-02-01').df

# aapl = api.polygon.daily_open_close('AAPL', '2020-06-01')

print(aapl)

#printing