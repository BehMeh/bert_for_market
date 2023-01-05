from convert_bars_to_id import Bar2BertConverter
from trading_view_data import TradingViewData
from two_candles_convert_to_id import TwoCandleBar2BertConverter

from bert_gym import BertGym
import numpy as np
import pandas as pd
import random
import torch
from transformers import BertModel
import stable_baselines3.common.env_checker as sb3
from stable_baselines3 import PPO
import os


# @misc{huggingfacecourse,
#   author = {Hugging Face},
#   title = {The Hugging Face Course, 2022},
#   howpublished = "\url{https://huggingface.co/course}",
#   year = {2022},
#   note = "[Online; accessed <today>]"
# }


def generate_BERT_sequence(sequence):
    # [CLS] = 101
    # [SEP] = 102
    sequence = np.append([101], sequence)
    sequence = np.append(sequence, [102])
    return sequence


def generate_BERT_sentence(data, min_length, max_length):
    last_index = len(data) - max_length
    sentence_length = random.randint(min_length, max_length)
    start = random.randint(min_length, last_index - 1)

    return data[start:start + sentence_length], start, sentence_length


DATA_SOURCES = {0: "Tradingview",
                1: "yfinance"}

if __name__ == '__main__':

    # start_date = '1999-01-01'
    # end_date = '2021-01-01'
    start_date = '2020-01-01'

    # start_date = '2022-10-16'
    end_date = '2023-01-04'
    test_symbol = ['IWM']
    datasource = 'yfinance'
    mode = 'test'
    pnl_calculation_length = 1
    # mygym = BertGym(True, start_date, end_date, 0)
    my_gym_spy = BertGym(True, start_date, end_date, mode, pnl_calculation_length=pnl_calculation_length, test_symbol=test_symbol,
                         datasource=datasource, take_shorts=False)
    my_gym_spy.training = False
    # mygym.reset()
    # for i in range(22):
    #     action = i % 2
    #     obs, rew, done, info = mygym.step(action)
    #     if done:
    #         mygym.reset()
    #
    # print(shit)
    # print(After this point it will save over the last model)

    model_name = "PPO_32_768"
    models_dir = "models/" + model_name
    log_dir = "logs/" + model_name

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # model = PPO("MlpPolicy", mygym, verbose=1, tensorboard_log=log_dir)
    # TIMESTEPS = 1000
    # for i in range(100):
    #     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
    #     model.save(f"{models_dir}/{TIMESTEPS * i}")

    model = PPO.load(models_dir + "/99000.zip", env=my_gym_spy)

    episode_reward = 0
    obs = my_gym_spy.reset()
    # for _ in range(510):
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # print("action: ", action)
        # print("action.type: ", action.dtype)
        obs, reward, done, info = my_gym_spy.step(action)
        # my_gym_spy.render()
        episode_reward += reward
        # if done or info.get("is_success", False):
        #     print("Reward:", episode_reward, "Success?", info.get("is_success", False))
        #     episode_reward = 0.0
        #     obs = my_gym_spy.reset()

    print(lllll)
    #
    policy_generator = TwoCandleBar2BertConverter(False)
    # policy_generator.generate_vocabulary()

    symbols = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE', 'DIA', 'TLT']
    data_generator = TradingViewData(start_date, end_date)

    # print(keepkhh)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    for symbol in symbols:
        symbol_data = data_generator.load_data(symbol, False)
        symbol_tokenized = policy_generator.tokenize(symbol_data, symbol)
        # print(symbol_tokenized.columns)

        sentence, starting_position_of_sentence, length_of_sentence = generate_BERT_sentence(
            np.array(symbol_tokenized['token_id']), 100, 150)
        pnl = symbol_tokenized.pnl.iloc[starting_position_of_sentence + length_of_sentence - 1]
        indexed_tokens = generate_BERT_sequence(sentence)

        segments_ids = [1] * len(indexed_tokens)

        tokens_tensor = torch.tensor(np.array([indexed_tokens]))
        segments_tensors = torch.tensor([segments_ids])

        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers.

        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

        # Let's combine the layers to make this one whole big tensor.
        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        # token_embeddings.size()

        # Let's get rid of the "batches" dimension since we don't need it.
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings.size()
        # Finally, we can switch around the "layers" and "tokens" dimensions with permute.

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # token_embeddings.size()

        # Concatinage last 4 layers
        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor

            # Concatenate the vectors (that is, append them together) from the last
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)

        print('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
        print(pd.DataFrame(token_vecs_cat).astype("float").to_numpy())
        print("Min: ", pd.DataFrame(token_vecs_cat).astype("float").min())
        print("Max: ", pd.DataFrame(token_vecs_cat).astype("float").max())
        break
