# This class converts bar patterns to IDs to be used by Bert
# The ID's start at 2,000 since BERT vocabulary start at 1,996
# This version uses 2 candles and their relationship through high/low
import pandas as pd
import os
from trading_view_data import TradingViewData
import numpy as np

START = 2000

def round_off(x, prec=2, base=0.3):
    return round(base * round(x / base), prec)


def generate_hlc_info(df):
    # High
    conditions = [
        (df['high'] < df['low'].shift()),
        (df['high'] > df['low'].shift()) & (df['high'] < df['high'].shift()),
        (df['high'] > df['high'].shift())]
    choices = [-1, 0, 1]
    df['h2h1'] = np.select(conditions, choices, default=np.nan)

    # Low
    conditionsl2l1 = [
        (df['low'] < df['low'].shift()),
        (df['low'] > df['low'].shift()) & (df['low'] < df['high'].shift()),
        (df['low'] > df['high'].shift())]
    choicesl2l1 = [-1, 0, 1]
    df['l2l1'] = np.select(conditionsl2l1, choicesl2l1, default=np.nan)

    # Close
    conditionsclose = [
        (df['close'] < df['open']) & (df['close'].shift() < df['open'].shift()),
        (df['close'] > df['open']) & (df['close'].shift() > df['open'].shift()),
        (df['close'] > df['open']) | (df['close'].shift() < df['open'].shift()),
        (df['close'] < df['open']) | (df['close'].shift() > df['open'].shift())]
    choicesclose = [-2, 1, 0, -1]
    df['c'] = np.select(conditionsclose, choicesclose, default=np.nan)

    df['h2h1_prev'] = df['h2h1'].shift(2)
    df['l2l1_prev'] = df['l2l1'].shift(2)

    conditionso3 = [
        (df['open'].shift(-1) < df['low']),
        (df['open'].shift(-1) > df['low']) & (df['open'].shift(-1) < df['high']),
        (df['open'].shift(-1) > df['high'])]
    choiceso3 = [-1, 0, 1]

    df['o_next'] = np.select(conditionso3, choiceso3, default=np.nan)

    conditionsvol = [
        (df['volume'] + df['volume'].shift(1) + df['volume'].shift(2) - (3 * df['volume'].shift(3))) < 0,
        (df['volume'] + df['volume'].shift(1) + df['volume'].shift(2) - (3 * df['volume'].shift(3))) > 0
    ]
    choicesvol = [0, 1]

    df['vol_cond'] = np.select(conditionsvol, choicesvol, default=np.nan)
    vocabulary_columns = ['h2h1_prev', 'l2l1_prev', 'h2h1', 'l2l1', 'c', 'o_next', 'vol_cond']
    return vocabulary_columns, df[vocabulary_columns]





class TwoCandleBar2BertConverter:
    def __init__(self, overlap: bool, pnl_calculation_length):
        self.token_id_start = START
        self.saving_path = 'tokenized_data/'
        self.overlap = overlap
        self.all_symbols_for_vocabulary = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE',
                                           'DIA', 'TLT']  # ,'YM=F', 'NQ=F',]
        self.vocabulary = self.load_vocabulary()
        self.main_columns = ['h2h1_prev', 'l2l1_prev', 'h2h1', 'l2l1', 'c', 'o_next', 'vol_cond']
        self.pnl_calculation_length = pnl_calculation_length

    def add_index_close_pnl(self, data, original_data):
        data.index = original_data.index
        data['close_price'] = original_data.close
        # data['open_price'] = original_data.open
        pnl_distance = self.pnl_calculation_length
        # We can use this method if we want to have a different pnl for no overlap tokenization
        # if not self.overlap:
        #     pnl_distance = 3
        data['pnl'] = (original_data['close'].shift(-pnl_distance) -
                       original_data['close']) / original_data['close']

        data['green'] = data['pnl'] > 0

        return data

    def load_vocabulary(self):
        print("---- Loading Vocabulary File ----")
        return pd.read_excel(self.saving_path + 'vocabulary.xlsx')

    def generate_vocabulary(self):
        """
        This function generates vocabulary table for 2 candle pattern
        Once it generates 2 candles, then it creates a 4 candle pattern + next open
        it autosaves the output
        :return: Vocabulary file
        """
        print("---- Generating Vocabulary File ----")
        start_date = '1999-01-01'
        end_date = '2020-12-05'

        data_generator = TradingViewData(start_date, end_date)
        all_data = pd.DataFrame()
        for symbol in self.all_symbols_for_vocabulary:
            data = data_generator.load_data(symbol, True)
            all_data = pd.concat((all_data, data))

        all_data.reset_index(inplace=True)

        col_names, vocabulary = generate_hlc_info(all_data)

        vocabulary_table = pd.DataFrame(vocabulary.dropna().value_counts(), columns=['value_count'])
        vocabulary_table.reset_index(inplace=True)
        vocabulary_table['token_id'] = vocabulary_table.index + self.token_id_start

        print("Saving Vocabulary file")
        vocabulary_table.to_excel(self.saving_path + 'vocabulary.xlsx', merge_cells=False)

        return vocabulary_table

    def tokenize(self, data, symbol_name, save_output=True):
        """
        This method will take the converted data and split it into batches of length size
        :param data: main dataframe to be tokenized
        :param symbol_name: Used for saving the output file
        :param save_output:
        :return: tokenized df
        """

        # 1: generate words
        col_names, converted_data = generate_hlc_info(data)
        # print("-------------")
        # print(converted_data.head(5))
        # print("-------------")
        # 3: merge to obtain token id for all rows
        # print("------------")
        # print("self.vocabulary.columns: ", self.vocabulary.columns[1:])
        # print("------------")
        converted_data = pd.merge(converted_data, self.vocabulary[self.vocabulary.columns[1:]], "left")

        converted_data = self.add_index_close_pnl(converted_data, data)

        converted_data.dropna(inplace=True)
        converted_data['token_id'] = converted_data['token_id'].astype(int)

        if not self.overlap:
            converted_data = converted_data.iloc[::4]
        if save_output:
            self.save_output(converted_data, symbol_name)

        return converted_data

    def _get_file_name(self, symbol_name):
        name = "Tokenized " + symbol_name
        if self.overlap:
            name = name + " With Overlap"
        else:
            name = name + " No Overlap"

        return name + ".xlsx"

    def save_output(self, df, symbol_name):
        self._check_folders()
        file_name = self._get_file_name(symbol_name)
        print("Saving", file_name)
        df.to_excel(self.saving_path + file_name, merge_cells=False)

    def _check_folders(self):
        if not os.path.exists(self.saving_path):
            print("Creating folder to save tokenized data")
            os.makedirs(self.saving_path)
