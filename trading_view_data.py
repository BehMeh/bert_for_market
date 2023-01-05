import pandas as pd
import glob

DATA_SOURCES = {0: "Tradingview",
                1: "yfinance"}


class TradingViewData:
    def __init__(self, start, end, datasource):
        self.start = start
        self.end = end
        self.dir = "data_set/" + datasource + "/"
        self.datasource = datasource

    def list_all_available_data(self):
        return glob.glob(self.dir + "*.csv")

    def main_columns(self):
        return ['open', 'high', 'low', 'close', 'volume']

    def load_data(self, symbol, load_full_data_set=True):
        print("symbol: ", symbol)
        if self.datasource == 'Tradingview':
            file_name = self.dir + symbol + ".csv"
            print("file_name: ", file_name)
            data = pd.read_csv(file_name, header=0, index_col=0)
        elif self.datasource == 'yfinance':
            file_name = self.dir + symbol + ".xlsx"
            print("file_name: ", file_name)
            data = pd.read_excel(file_name, header=0, index_col=0)

        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        data.index = [i.strftime('%Y-%m-%d') for i in data.index]
        data.columns = data.columns.str.lower()
        data.index = pd.to_datetime(data.index)
        if not load_full_data_set:
            data = data.loc[self.start:self.end]
        print("Loading ", len(data), " rows of data for ", symbol)
        # print(data.iloc[0])

        return data
