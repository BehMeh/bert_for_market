from typing import Optional, Union, List
from two_candles_convert_to_id import TwoCandleBar2BertConverter

# from gym.core import RenderFrame

from trading_view_data import TradingViewData

import numpy as np
from gym import spaces
import gym
import pandas as pd
from transformers import BertModel
# import random
import torch
import datetime

import logging
from sklearn.preprocessing import normalize


def generate_BERT_sequence(sequence):
    # [CLS] = 101
    # [SEP] = 102
    sequence = np.append([101], sequence)
    sequence = np.append(sequence, [102])
    return sequence


SENTENCE_SIZE = 32

INITIAL_CAPITAL = 100000
ACTIONS = {0: "Short",
           1: "Long"}

MODES = {0: "Train",
         1: "Test"}
# the system will print a report of the traded after every 100 trades
REPORT_COUNTER = 100
# BERT_OUTPUT_SIZE = 3072
BERT_TYPES = {0: "Sum (768 Columns)",
              1: "Concatenate (3,072 Columns)"}


class BertGym(gym.Env):

    def __init__(self, overlap, start_date, end_date, mode, pnl_calculation_length=1, test_symbol=['SPY'], bert_type=0,
                 datasource='Tradingview', take_shorts=True):
        self.bert_type = bert_type
        self.bert_output_size = 768
        lower_end = -60
        if bert_type == 1:
            self.bert_output_size = 3072
            lower_end = -30
        self.highest_high = INITIAL_CAPITAL
        self.mode = mode
        self.buy_and_hold_portfolio_balance = INITIAL_CAPITAL
        self.portfolio_balance = INITIAL_CAPITAL
        self.buy_and_hold_number_of_shares = 0.0
        self.max_dd = 0.0
        self.buy_and_hold_max_dd = 0.0
        self.buy_and_hold_highest_high = INITIAL_CAPITAL
        self.model = None
        self.load_model()
        self.data_generator = TradingViewData(start_date, end_date, datasource)
        self.all_symbols = test_symbol
        if mode == 'train':
            self.all_symbols = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE', 'DIA',
                                'TLT']
        self.symbol_counter = -1
        self.current_symbol = self.all_symbols[0]
        self.counter = 0
        self.current_data = None
        self.observation_space = spaces.Box(low=-1, high=1, shape=(SENTENCE_SIZE * self.bert_output_size,),
                                            dtype=np.float32)
        # print("self.observation_space.sample().dtype: ", self.observation_space.sample().shape)
        self.action_space = spaces.Discrete(2)
        self.total_reward = 0
        self.token_generator = TwoCandleBar2BertConverter(overlap, pnl_calculation_length)
        self.round = 4  # number of digits to round
        # this is the row that contains the pnl information. The system sees SENTENCE_SIZE bars and then takes an action
        self.pnl_location = self.counter + SENTENCE_SIZE - 1

        self.number_of_short_trades = 0
        # to avoid division by 0 when reporting
        if take_shorts == False:
            self.number_of_short_trades = 1
        self.number_of_long_trades = 0
        self.number_of_won_short_trades = 0
        self.number_of_won_long_trades = 0

        self.number_of_long_trades_per_report = 0

        self.number_of_short_trades_per_report = 0
        # to avoid division by 0 when reporting
        if take_shorts == False:
            self.number_of_short_trades_per_report = 0
        self.number_of_won_long_trades_per_report = 0
        self.number_of_won_short_trades_per_report = 0

        # now we will Create and configure logger
        logging.basicConfig(filename="bert.log",
                            format='%(asctime)s %(message)s',
                            filemode='w')

        # Let us Create an object
        self.logger = logging.getLogger()
        # Now we are going to Set the threshold of logger to DEBUG
        self.logger.setLevel(logging.DEBUG)
        self.symbol_result = pd.DataFrame(columns=['symbol', 'number_of_long_trades', 'number_of_long_trades_won',
                                                   'long_trade_win_ratio',
                                                   'number_of_short_trades', 'number_of_short_trades_won',
                                                   'short_trade_win_ratio'])
        self.all_symbol_results = pd.DataFrame()
        self.take_shorts = take_shorts
        # this parameter calculates how many days in the future the system has to look for profit
        # it also trades only every other number of days based on this value
        self.pnl_calculation_length = pnl_calculation_length

    def load_model(self):
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, )

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

    def load_data(self):
        raw_data = self.data_generator.load_data(self.all_symbols[self.symbol_counter], False)
        tokenized_data = self.token_generator.tokenize(raw_data, self.all_symbols[self.symbol_counter], True)
        return tokenized_data

    def _get_info(self):

        return {"Portfolio Balance: $": round(self.portfolio_balance, self.round),
                "Portfolio Draw Down: %": round(100 * self.max_dd, self.round),
                "Buy and Hold Balance: $": round(self.buy_and_hold_portfolio_balance, self.round),
                "Buy and Hold Draw Down: %": round(100 * self.buy_and_hold_max_dd, self.round),
                "Total Reward *100: %": round(100 * self.total_reward, self.round)}

    def get_obs(self):
        sentence = self.generate_BERT_sentence()
        # print("len(sentence): ", len(sentence))
        indexed_tokens = generate_BERT_sequence(sentence)
        # print("len(indexed_tokens): ", len(indexed_tokens))
        segments_ids = [1] * len(indexed_tokens)

        tokens_tensor = torch.tensor(np.array([indexed_tokens]))
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)

        # print("first token_embeddings.size(): ", token_embeddings.size())

        # Let's get rid of the "batches" dimension since we don't need it.
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # print("token_embeddings.size(): ", token_embeddings.size())
        # Finally, we can switch around the "layers" and "tokens" dimensions with permute.

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # print(token_embeddings.size())

        # Concatinage last 4 layers
        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor
            # Concatenate the vectors (that is, append them together) from the last
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            if self.bert_type == 1:
                vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            elif self.bert_type == 0:
                vec = torch.sum(token[-4:], dim=0)

            token_vecs.append(vec)
        # print(pd.DataFrame(token_vecs_cat).astype("float"))
        # print("token_vecs_cat.type: ", token_vecs_cat.type)

        # print(pd.DataFrame(token_vecs).astype(np.float32))

        final_obs = (pd.DataFrame(token_vecs).astype(np.float32)).to_numpy().reshape(-1)
        # print("type(final_obs[3089]): ", type(final_obs[3089]))
        # print("Min: ", final_obs.min())
        # print("Max: ", final_obs.max())
        output = final_obs[self.bert_output_size:-self.bert_output_size]
        # print("output.shape: ", output.shape)

        normalized_output = normalize([output]).ravel().astype(np.float32)
        # print("normalized_output.shape: ", normalized_output.shape)
        # print(normalized_output)

        assert self.observation_space.contains(normalized_output), (
            "The observation method does not match the given observation space")

        return normalized_output

    def calculate_reward(self, action):
        # 2: Long
        # 1: short
        # 0: Hold
        pnl = self.current_data.pnl.iloc[self.pnl_location]
        # print("pnl for ", self.current_data.index.values[self.pnl_location], " is : %", pnl)
        rew = 0
        if action == 1:
            rew = pnl
            self.number_of_long_trades_per_report += 1
            self.number_of_long_trades += 1
            if rew > 0:
                self.number_of_won_long_trades_per_report += 1
                self.number_of_won_long_trades += 1
                print("Long Trade Won")
            else:
                print("Long Trade Lost")

        elif action == 0 and self.take_shorts:
            rew = -pnl
            self.number_of_short_trades_per_report += 1
            self.number_of_short_trades += 1
            if rew > 0:
                self.number_of_won_short_trades_per_report += 1
                self.number_of_won_short_trades += 1
                print("Short Trade Won")
            else:
                print("Short Trade Lost")
        # elif action == 0:
        #     rew = -abs(pnl) * 0.01

        self.total_reward = self.total_reward + rew

        self.portfolio_balance = (self.portfolio_balance * rew) + self.portfolio_balance
        self.buy_and_hold_portfolio_balance = self.buy_and_hold_number_of_shares * \
                                              self.current_data.iloc[self.pnl_location]['close_price']
        # print("['close_price']: ", self.current_data.iloc[self.pnl_location]['close_price'])
        self._calculate_dd()

        return rew

    def _calculate_dd(self):
        """
        This function calculates draw down and highest high watermark
        :return: Noting
        """

        dif = (self.portfolio_balance - self.highest_high) / self.highest_high

        if dif >= 0:
            self.highest_high = round(self.portfolio_balance, self.round)

            # self.highest_high_bar_position = self.data_counter
            # self.last_highest_high_data_counter = self.data_counter

        if dif < self.max_dd:
            self.max_dd = round(dif, self.round)

        buy_and_hold_dif = (self.buy_and_hold_portfolio_balance - self.buy_and_hold_highest_high) / \
                           self.buy_and_hold_highest_high

        if buy_and_hold_dif >= 0:
            self.buy_and_hold_highest_high = round(self.buy_and_hold_portfolio_balance, self.round)

        if buy_and_hold_dif < self.buy_and_hold_max_dd:
            self.buy_and_hold_max_dd = round(buy_and_hold_dif, self.round)

    def generate_BERT_sentence(self):
        # last_index = len(data) - max_length
        # sentence_length = random.randint(min_length, max_length)
        # start = random.randint(min_length, last_index - 1)
        # print(self.current_data.columns)
        # print(self.current_data.index.values[self.counter:self.counter + SENTENCE_SIZE])
        return self.current_data.token_id[self.counter:self.counter + SENTENCE_SIZE]

    def step(self, action):

        # print(self.current_symbol, " -- Step ", self.counter, " [",
        #       pd.to_datetime(self.current_data.index.values[self.counter]).strftime("%Y-%m-%d")
        #       , "] -- Action ->", ACTIONS[action])
        print(self.current_symbol, " -- Step ", self.counter, " [",
              pd.to_datetime(self.current_data.index.values[self.counter]).strftime("%Y-%m-%d"), "]")

        reward = self.calculate_reward(action)

        self.counter += self.pnl_calculation_length
        obs = self.get_obs()

        done = False
        if len(self.current_data) < (self.counter + SENTENCE_SIZE + 2):
            done = True
            self._generate_full_symbol_report()
        # if self.counter >11:
        #     done = True

        self.pnl_location = self.counter + SENTENCE_SIZE - 1

        # print a report of the system after every 100 trades
        if self.counter % REPORT_COUNTER == 0:
            print("--------------- Report for ", self.current_symbol, " ---------------")
            print(self._get_info())
            print("----------------------------------------------")
            print("Number of Long: ", self.number_of_long_trades_per_report)
            print("Number of Won Long Trades: ", self.number_of_won_long_trades_per_report)
            print("Long Win Ratio, ",
                  round(100 * self.number_of_won_long_trades_per_report / self.number_of_long_trades_per_report, 2))
            print("Number of short: ", self.number_of_short_trades_per_report)
            print("Number of Won Short Trades: ", self.number_of_won_short_trades_per_report)

            short_win_ratio = 0
            if self.take_shorts:
                print("Short Win Ratio, ",
                      round(100 * self.number_of_won_short_trades_per_report / self.number_of_short_trades_per_report,
                            2))
                short_win_ratio = round(
                    100 * self.number_of_won_short_trades_per_report / self.number_of_short_trades_per_report, 2)

            long_win_ratio = round(
                100 * self.number_of_won_long_trades_per_report / self.number_of_long_trades_per_report, 2)
            self.logger.info("--------------- Report %s", str(self.counter // REPORT_COUNTER))
            self.logger.info("--------------- %s", self.current_symbol)
            self.logger.info(self._get_info())
            self.logger.info("----------------------------------------------")
            self.logger.info("Number of Long: %s", self.number_of_long_trades_per_report)
            self.logger.info("Number of Won Long Trades: %a", self.number_of_won_long_trades_per_report)
            self.logger.info("Long Win Ratio, %f", long_win_ratio)
            self.logger.info("Number of short: %s", self.number_of_short_trades_per_report)
            self.logger.info("Number of Won Short Trades: %s", self.number_of_won_short_trades_per_report)
            self.logger.info("Short Win Ratio, %f", short_win_ratio)
            self.logger.info("=====================================================================")

            new_row = {'symbol': self.current_symbol,
                       'result_date': pd.to_datetime(self.current_data.index.values[self.counter]).strftime("%Y-%m-%d"),
                       'number_of_long_trades': self.number_of_long_trades_per_report,
                       'number_of_long_trades_won': self.number_of_won_long_trades_per_report,
                       'long_trade_win_ratio': long_win_ratio,
                       'number_of_short_trades': self.number_of_short_trades_per_report,
                       'number_of_short_trades_won': self.number_of_won_short_trades_per_report,
                       'short_trade_win_ratio': short_win_ratio, 'get_info': self._get_info()}
            self.symbol_result = self.symbol_result.append(new_row, ignore_index=True)

            if self.mode == 'test':
                shorts = ' With'
                if not self.take_shorts:
                    shorts = shorts + " No"
                shorts = shorts + " Shorts"
                self.symbol_result.to_excel('results/' + self.current_symbol + shorts + '_100_runs profit every ' +
                                            str(self.pnl_calculation_length) + ' bars.xlsx')

            self.number_of_long_trades_per_report = 0
            self.number_of_short_trades_per_report = 0
            self.number_of_won_long_trades_per_report = 0
            self.number_of_won_short_trades_per_report = 0
            if not self.take_shorts:
                self.number_of_short_trades_per_report = 1

        return obs, reward, done, self._get_info()

    def render(self):
        pass

    def _generate_full_symbol_report(self):
        print("--------------- Reset Report for ", self.current_symbol, " ---------------")
        print(self._get_info())
        print("------------------------------------------------------")
        print("Total Number of long Trades: ", self.number_of_long_trades)
        print("Number of Won Long Trades: ", self.number_of_won_long_trades)
        print("Long Win Ratio, %", round(100 * self.number_of_won_long_trades / self.number_of_long_trades, 2))
        print("Total Number of Short Trades: ", self.number_of_short_trades)
        print("Number of Won Short Trades: ", self.number_of_won_short_trades)

        if self.take_shorts:
            print("Short Win Ratio, %", round(100 * self.number_of_won_short_trades / self.number_of_short_trades, 2))
        print("=============================================")

        self.logger.info("Reset Report for %s ", self.current_symbol)
        self.logger.info("-----------------------------------------")
        self.logger.info(self._get_info())
        self.logger.info("------------------------------------------------------")
        self.logger.info("Total Number of long Trades: %s", self.number_of_long_trades)
        self.logger.info("Number of Won Long Trades: %s", self.number_of_won_long_trades)
        self.logger.info("Long Win Ratio, %f",
                         round(100 * self.number_of_won_long_trades / self.number_of_long_trades, 2))
        self.logger.info("Total Number of Short Trades: %s", self.number_of_short_trades)
        self.logger.info("Number of Won Short Trades: %s", self.number_of_won_short_trades)

        if self.take_shorts:
            self.logger.info("Short Win Ratio, %f",
                             round(100 * self.number_of_won_short_trades / self.number_of_short_trades, 2))
        self.logger.info("=============================================")

        info = self._get_info()
        info['symbol'] = self.current_symbol
        info['Total Number of long Trades'] = self.number_of_long_trades
        info["Number of Won Long Trades"] = self.number_of_won_long_trades
        info["Long Win Ratio"] = round(self.number_of_won_long_trades / self.number_of_long_trades, 2)
        info["Total Number of Short Trades"] = self.number_of_short_trades
        info["Number of Won Short Trades"] = self.number_of_won_short_trades

        if self.take_shorts:
            info["Short Win Ratio"] = round(self.number_of_won_short_trades / self.number_of_short_trades, 2)

        self.all_symbol_results = self.all_symbol_results.append(info, ignore_index=True)
        shorts = 'With'
        if not self.take_shorts:
            shorts = shorts + " No"
        shorts = shorts + " Shorts"
        self.all_symbol_results.to_excel(
            'results/' + shorts + ' full_results profit every ' + str(self.pnl_calculation_length) + ' bars.xlsx')

    def reset(self):
        # Only print all this if the reset is called after a done flag
        if self.counter != 0:
            self._generate_full_symbol_report()

            # self.symbol_result = pd.DataFrame()

        self.symbol_counter += 1
        self.symbol_counter = self.symbol_counter % len(self.all_symbols)
        self.counter = 0
        self.current_symbol = self.all_symbols[self.symbol_counter]
        self.current_data = self.load_data()
        self.pnl_location = self.counter + SENTENCE_SIZE - 1
        # print(self.current_data)
        # print('self.current_data.columns:', self.current_data.columns)
        self.buy_and_hold_number_of_shares = INITIAL_CAPITAL / self.current_data.iloc[self.pnl_location]['close_price']
        self.buy_and_hold_portfolio_balance = INITIAL_CAPITAL
        self.portfolio_balance = INITIAL_CAPITAL
        self.highest_high = INITIAL_CAPITAL
        self.max_dd = 0.0
        self.buy_and_hold_max_dd = 0.0
        self.buy_and_hold_highest_high = INITIAL_CAPITAL

        # obs = self.get_obs()
        # print("RESET FUNCTION: obs.shape ", obs.shape)

        self.total_reward = 0.0

        self.number_of_won_short_trades = 0
        self.number_of_short_trades = 0
        self.number_of_won_long_trades = 0
        self.number_of_long_trades = 0

        if self.take_shorts == False:
            self.number_of_short_trades = 1

        return self.get_obs()
