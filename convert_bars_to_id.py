# This class converts bar patterns to IDs to be used by Bert
# The ID's start at 2,000 since BERT vocabulary start at 1,996
# This version uses normalized candles
import pandas as pd
import os

START = 2000

POLICIES = {0: "Only Close",
            1: "Close and Open"}


def normalize_single_bar(d_f):
    return (d_f[['open', 'high', 'low', 'close']].
            sub(d_f[['open', 'high', 'low', 'close']].
                mean(axis=1), axis=0).div(d_f.loc[:, ['open', 'high', 'low', 'close']].
                                          std(axis=1), axis=0)
            )


def round_off(x, prec=2, base=0.3):
    return round(base * round(x / base), prec)


class Bar2BertConverter:
    def __init__(self, policy, length, data, symbol_name,overlap: bool):
        self.token_id_start = START
        self.policy = policy
        self.length = length
        self.original_data = data
        self.normalized_data = self._convert_data().apply(round_off)
        self.saving_path = 'tokenized_data/'
        self.symbol_name = symbol_name
        self.overlap = overlap

    def _convert_data(self):
        """
        :param d_f1:the original data
        :param policy: type of policy used to convert the data
        :return: df converted using the policy
        """

        # Assert policy is in POLICIES

        print("Converting Data Using Policy: ", POLICIES[self.policy])
        normalized_df = normalize_single_bar(self.original_data)
        if self.policy == 0:
            normalized_df.drop(['open', 'high', 'low'], axis=1, inplace=True)
        if self.policy == 1:
            normalized_df.drop(['high', 'low'], axis=1, inplace=True)

        return normalized_df

    def get_columns(self):
        if self.policy == 0:
            return ['close']
        if self.policy == 1:
            return ['open', 'close']

    def add_index_close_pnl(self, data):
        data.index = self.original_data.index
        data['close_price'] = self.original_data.close
        data['pnl'] = 100 * (self.original_data['close'].shift(-self.length) -
                             self.original_data['close']) / self.original_data['close']

        data['green'] = data['pnl'] > 0

        return data

    def generate_words(self):
        columns = self.get_columns()
        modified = self.normalized_data
        for i in range(1, self.length):
            for col in columns:
                modified[col + "_" + str(i)] = self.normalized_data[col].shift(i)
        return modified

    def tokenize(self, save_output=True):
        """
        This method will take the converted data and split it into batches of length size
        :param save_output:
        :param overlap: if the system is going to use overlap bars or not. if this is false
        then the size of df returned is going to be data_to_be_tokenized/length
        :return: tokenized df
        """

        # 1: generate words
        # shift previous length number of bars forward from data_to_be_tokenized
        normalize = self.generate_words()

        # 1: create a value_counts() from data_to_be_tokenized
        value_counts = pd.DataFrame(normalize.value_counts(), columns=['value_count']).reset_index()
        # 2: Add a col as token_id to this column
        value_counts['token_id'] = value_counts.index + self.token_id_start
        value_counts.drop('value_count', axis=1, inplace=True)

        # 3: merge to obtain token id for all rows
        normalize = pd.merge(normalize, value_counts, "left")

        normalize = self.add_index_close_pnl(normalize)

        normalize.dropna(inplace=True)
        normalize['token_id'] = normalize['token_id'].astype(int)

        if not self.overlap:
            normalize = normalize.iloc[::self.length]
        if save_output:
            self.save_output(normalize)

        return normalize

    def _get_file_name(self):
        name = "Tokenized " + self.symbol_name + " - Length " + str(self.length)
        if self.overlap:
            name = name + " With Overlap"
        else:
            name = name + " No Overlap"

        return name+".xlsx"

    def save_output(self, df):
        self._check_folders()
        file_name = self._get_file_name()
        print("Saving", file_name)
        df.to_excel(self.saving_path + file_name, merge_cells=False)

    def _check_folders(self):
        if not os.path.exists(self.saving_path):
            print("Creating folder to save tokenized data")
            os.makedirs(self.saving_path)