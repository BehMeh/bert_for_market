import yfinance as yf
start = '1990-01-01'
end = '2023-01-04'
symbol = 'IWM'
data = yf.Ticker(symbol).history(start=start, end=end)
data.index = data.index.tz_localize(None)
data.drop(['Dividends',	'Stock Splits', 'Capital Gains'], axis =1, inplace= True)
data.columns=data.columns.str.lower()
data.to_excel("data_set/yfinance/"+symbol+".xlsx")

