# cython: language_level=3

import sys
import pandas as pd
import numpy as np

sys.path.append("../../scorer")
from orderbook_fast import Event, OrderBook

SIDE_BID = 0
SIDE_ASK = 1

STOPPER = 0

class MyOrderBook:
    """
    Для сохранения переменных между вызовами функций - обработчиков
    """

    def __init__(self):
        self.clear()

    def clear(self):
        """
        Reset all class variables in the beginning of every chunk action=10
        :return: None
        """
        global STOPPER

        if 2 < STOPPER < 5:
            print(f" {STOPPER} clear ")
            print(self._last_deals)
        STOPPER += 1

        self._last_deals = pd.DataFrame(np.full(shape=[30, 7], fill_value=np.nan, dtype=np.int_),
            columns=["time", "cat_action", "cat_type", "cat_side", "price",
                     "amount", "cat_is_snapshot",])

    def set_last_deal(self, last_deal):
        self._event_to_queue_forward(event=last_deal, queue=self._last_deals)

    def _event_to_np(self, ev):
        result = np.array([ev.time, ev.action, ev.type,
                           ev.side, ev.price, ev.amount,
                           ev.is_snapshot,
                           ], dtype=np.int_)
        return result
    def _event_to_queue_forward(self, event, queue):
        """
        Метод управляет очередь, новые события в конец датафрейма
        :return:
        """

        queue.iloc[0] = self._event_to_np(event)
        ind = list(queue.index - 1)
        ind[0] = len(ind) - 1
        queue.index = ind
        queue.sort_index(inplace=True)

    def _event_to_queue_backward(self, event, queue):
        """
        Метод управляет очередь, новые события в начало датафрейма
        :return:
        """
        queue.iloc[-1] = self._event_to_np(event)
        ind = list(queue.index + 1)
        ind[-1] = 0
        queue.index = ind
        queue.sort_index(inplace=True)

    def get_features_from_orderbook(self, orderbook, max_index=13):
        '''
            Считаем простые фичи по ордербуку:
        '''


        spread = orderbook.get_best_price(SIDE_ASK) - orderbook.get_best_price(SIDE_BID)
        features = [spread]
        for side in (SIDE_BID, SIDE_ASK):
            for ix in range(max_index):
                price_level = orderbook.get_price_level_at_ix(side, ix)
                if price_level is None:
                    features += [-1, -1]
                else:
                    features += [price_level.get_volume(), price_level.get_num_orders()]

        for com in np.linspace(start=0, stop=200, num=5):
            features.append(self._last_deals.price.ewm(com=com))

        return features

    def get_simple_deals_features(self, last_deals, orderbook):
        '''
            Считаем простые фичи по последним сделкам:
        '''
        cur_mean_price = orderbook.get_mean_price()
        cur_time = orderbook.get_time()

        features = []
        for side in (SIDE_BID, SIDE_ASK):
            deal_event = last_deals[side]
            if deal_event is None:
                features += [-1e9, -1e9, -1e9]
            else:
                features += [cur_mean_price - deal_event.price,
                             cur_time - deal_event.time,
                             deal_event.amount]
        return features

    def _get_cat_features_from_event(self, ):
        pass

