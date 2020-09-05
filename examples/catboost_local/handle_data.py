import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd
from pathlib import Path

import catboost

sys.path.append("../../scorer/")
sys.path.append("../../scorer/orderbook_fast")
import orderbook_fast as ob_fast

from my_orderbook import MyOrderBook, action_handler
catboost_myob = MyOrderBook()

def collect_dataset(data_path):
    '''
        Собираем датасет
    '''

    global catboost_myob
    event_player = ob_fast.EventPlayer(data_path)
    orderbook_fast = ob_fast.OrderBook()

    X = []
    Y = []

    for ev in tqdm(event_player.iget_events(),
                    total=len(event_player),
                    desc=f"collecting dataset {data_path}"):

        if not ev.is_snapshot:
            catboost_myob.__getattribute__(action_handler[ev.action])

            orderbook_fast.apply_event(ev)
            if ev.need_prediction:
                features = catboost_myob.get_features(ev, orderbook_fast)

                X.append(features)
                Y.append(ev.Y)

    print(f"Dataset collected: len(X) = {len(X)}")
    return pd.DataFrame(X), pd.DataFrame(Y)


# X_train_very_small_A, Y_train_very_small_A = collect_dataset("../../data/very_small_A.npz")
# X_train_very_small_A.to_pickle("../../data/X_train_very_small_A_130.pickle")
# Y_train_very_small_A.to_pickle("../../data/Y_train_very_small_A_130.pickle")

# X_test_very_small_B, Y_test_very_small_B = collect_dataset("../../data/very_small_B.npz")
# X_test_very_small_B.to_pickle("../../data/X_test_very_small_B_130.pickle")
# Y_test_very_small_B.to_pickle("../../data/Y_test_very_small_B_130.pickle")

if __name__ == "__main__":
    print(f"curdir: {Path.cwd()}")


    X_train80, Y_train80 = collect_dataset("../../data/train80.npz")
    X_train80.to_pickle("../../data/X_train80_130.pickle")
    Y_train80.to_pickle("../../data/Y_train80_130.pickle")

    X_test20, Y_test20 = collect_dataset("../../data/test20.npz")
    X_test20.to_pickle("../../data/X_test20_130.pickle")
    Y_test20.to_pickle("../../data/Y_test20_130.pickle")

    print("X_train_very_small_A, Y_train_very_small_A", X_train80, Y_train80)
    print("X_test_very_small_B, Y_test_very_small_B", X_test20, Y_test20)

    print( "********* FINISH dataset collection ***********************")