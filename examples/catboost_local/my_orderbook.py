# cython: language_level=3

import sys
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

sys.path.append("../../scorer")
from orderbook_fast import Event, OrderBook

ACTION_DELETE = 0
ACTION_ADD = 1
ACTION_MODIFY_AMOUNT = 2
ACTION_DEAL = 3
ACTION_NEW_CHUNK = 10

SIDE_BID = 0
SIDE_ASK = 1


STOPPER = 0

class MyOrderBook:
    """
    Для сохранения переменных между вызовами функций - обработчиков
    """

    def __init__(self):
        self.clear()

    def clear(self, _=None):
        """
        Reset all class variables in the beginning of every chunk action=10
        :return: None
        """

        # global STOPPER

        # if 2 < STOPPER < 5:
        #     print(f" {STOPPER} clear ")
        #     print(self._last_deals)
        # STOPPER += 1

        self._last_deals = pd.DataFrame(np.full(shape=[130, 24], fill_value=np.nan))



    def set_action_delete_order(self, deleted_order):
        pass
        # self._event_to_queue_forward(event=deleted_order, queue=self._deleted_orders)
    def set_action_add_order(self, added_order):
        added_order.action = 0
        self._event_to_queue_forward(event=added_order, queue=self._last_deals)
    def set_action_modify_amount(self, modified_order):
        pass
        # self._event_to_queue_forward(event=modified_order, queue=self._modified_orders)
    def set_action_deal(self, last_deal):
        last_deal.action = 1
        self._event_to_queue_forward(event=last_deal, queue=self._last_deals)

    def _event_to_np(self, ev):
        """
        Главный формирователь очереди
        :param ev: event для расчета индекса, куда вставлять  result
        :param val: значение, которое вставляем
        :return:
        """
        offset = ev.action + ev.type*2 + ev.side*4
        # block_columns_starts = [0, 8, 16]
        ind_price = 0 + offset
        ind_amount = 8 + offset
        ind_price_amount = 16 + offset

        result = np.full(24, fill_value=np.nan)
        result[ind_price] = ev.price
        result[ind_amount] = ev.amount
        result[ind_price_amount] = ev.amount * ev.price

        return result

    def _event_to_queue_forward(self, event, queue):
        """
        Метод управляет очередью, новые события в конец датафрейма
        :return:
        """

        queue.iloc[0] = self._event_to_np(event)
        ind = list(queue.index - 1)
        ind[0] = len(ind) - 1
        queue.index = ind
        queue.sort_index(inplace=True)

    def _event_to_queue_backward(self, event, queue):
        """
        Метод управляет очередью, новые события в начало датафрейма
        :return:
        """
        queue.iloc[-1] = self._event_to_np(event)
        ind = list(queue.index + 1)
        ind[-1] = 0
        queue.index = ind
        queue.sort_index(inplace=True)

    def get_features(self, event):
        '''
            Считаем фичи :
        '''

        features = self._my_get_simple_deals_features(event)
        features += self._get_ewm_from_queue(self._last_deals)

        return features


    def _get_ewm_from_queue(self, queue):
        features = []
        for com in [0.2, .4, .8, 1.2, 1.5, 2, 4, 6, 10]:
            # features.append(queue.ewm(com=com, ignore_na=True).mean())
            features += list(queue.ewm(com=com, ignore_na=True).mean().mean().values.ravel())



        return features

    def _my_get_simple_deals_features(self, deal):
        '''
            Считаем простые фичи по последним сделкам:
        '''

        price = [None, None]
        price[deal.side] = deal.price

        amount = [None, None]
        amount[deal.side] = deal.amount

        price_amount = [None, None]
        price_amount[deal.side] = deal.amount * deal.price

        result = [
            *price,
            *amount,
            *price_amount,
        ]

        return result

    def _get_simple_features_from_orderbook(self, oredrbook_fast, max_index=13):
        '''
            Считаем простые фичи по ордербуку:
        '''
        spread = oredrbook_fast.get_best_price(SIDE_ASK) - oredrbook_fast.get_best_price(SIDE_BID)
        features = [spread]
        for side in (SIDE_BID, SIDE_ASK):
            for ix in range(max_index):
                price_level = oredrbook_fast.get_price_level_at_ix(side, ix)
                if price_level is None:
                    features += [-1, -1]
                else:
                    features += [price_level.get_volume(),
                                 price_level.get_num_orders()]
        return features



action_handler = {
    ACTION_DELETE: "set_action_delete_order",
    ACTION_ADD: "set_action_add_order",
    ACTION_MODIFY_AMOUNT: "set_action_modify_amount",
    ACTION_DEAL: "set_action_deal",
    ACTION_NEW_CHUNK: "clear",
}




