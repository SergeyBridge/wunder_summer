# cython: language_level=3

import sys
import pandas as pd
import numpy as np

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

        self._cat_features = []

        # global STOPPER

        # if 2 < STOPPER < 5:
        #     print(f" {STOPPER} clear ")
        #     print(self._last_deals)
        # STOPPER += 1

        queue_df = pd.DataFrame(np.full(shape=[130, 7], fill_value=np.nan),
            columns=["time", "action", "type", "side", "price",
                     "amount", "is_snapshot",], )
        self._last_deals = queue_df.copy()
        self._added_orders = queue_df.copy()
        self._deleted_orders = queue_df.copy()
        self._modified_orders = queue_df.copy()

        self.last_deals_reference = [None, None]

    def set_action_delete_order(self, deleted_order):
        self._event_to_queue_forward(event=deleted_order, queue=self._deleted_orders)
    def set_action_add_order(self, added_order):
        self._event_to_queue_forward(event=added_order, queue=self._added_orders)
    def set_action_modify_amount(self, modified_order):
        self._event_to_queue_forward(event=modified_order, queue=self._modified_orders)
    def set_action_deal(self, last_deal):
        self._event_to_queue_forward(event=last_deal, queue=self._last_deals)

    def _event_to_np(self, ev):
        result = np.array([ev.time, ev.action, ev.type,
                           ev.side, ev.price, ev.amount,
                           ev.is_snapshot,
                           ], dtype=np.int_)
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

    def get_features(self, event, oredrbook_fast, max_index=13):
        '''
            Считаем фичи :
        '''

        features = self._my_get_simple_deals_features(event, oredrbook_fast)

        spread = oredrbook_fast.get_best_price(SIDE_ASK) - oredrbook_fast.get_best_price(SIDE_BID)
        features += [spread]

        for side in (SIDE_BID, SIDE_ASK):
            for ix in range(max_index):
                price_level = oredrbook_fast.get_price_level_at_ix(side, ix)
                if price_level is None:
                    features += [-1, -1]
                else:
                    features += [price_level.get_volume(), price_level.get_num_orders()]

        features += self._get_ewm_from_queue(self._last_deals)
        features += self._get_ewm_from_queue(self._added_orders)
        features += self._get_ewm_from_queue(self._deleted_orders)
        features += self._get_ewm_from_queue(self._modified_orders)

        # if event.action == ACTION_DEAL:
        #     self.last_deals_reference[event.side] = event
        #     features += self._get_simple_deals_features(event, oredrbook_fast)

        features +=self._get_simple_features_from_orderbook(oredrbook_fast)



        return features


    def _get_ewm_from_queue(self, queue):
        features = []
        for com in [0.2, .4, .8, 1.2, 1.5, 2, 4, 6, 10]:
            features.append(queue.price.ewm(com=com, ignore_na=True).mean().mean())
            features.append(queue.amount.ewm(com=com, ignore_na=True).mean().mean())
            price_mult_amount = queue.price * queue.amount
            features.append(price_mult_amount.ewm(com=com, ignore_na=True).mean().mean())
            features.append(np.log(price_mult_amount).ewm(com=com, ignore_na=True).mean().mean())

            bids_volume = sum(price_mult_amount[self._last_deals.side == 0])
            ask_volume = sum(price_mult_amount[self._last_deals.side == 1])

            if ask_volume == 0:
                ask_volume = 0.5
            features += [bids_volume / ask_volume]

            bids_count = sum(self._last_deals.side == 0)
            ask_count = sum(self._last_deals.side == 1)
            if ask_count == 0:
                ask_count = 0.5
            features += [bids_count / ask_count]

        return features
        
    # def get_simple_deals_features(self, last_deals, orderbook):
    #     '''
    #         Считаем простые фичи по последним сделкам:
    #     '''
    #     cur_mean_price = orderbook.get_mean_price()
    #     cur_time = orderbook.get_time()
    #     features = []
    #     for side in (SIDE_BID, SIDE_ASK):
    #         deal_event = last_deals[side]
    #         if deal_event is None:
    #             features += [-1e9, -1e9, -1e9]
    #         else:
    #             features += [cur_mean_price - deal_event.price,
    #                          cur_time - deal_event.time,
    #                          deal_event.amount]
    #     return features

    def _my_get_simple_deals_features(self, deal, orderbook):
        '''
            Считаем простые фичи по последним сделкам:
        '''
        cur_mean_price = orderbook.get_mean_price()
        cur_time = orderbook.get_time()

        result = [int(deal.action),
                  int(deal.type),
                  int(deal.side),
                  int(deal.is_snapshot),
                  cur_mean_price - deal.price,
                  cur_time - deal.time,
                  deal.amount
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

    @property
    def cat_features(self):
        return [0, 1, 2, 3]


action_handler = {
    ACTION_DELETE: "set_action_delete_order",
    ACTION_ADD: "set_action_add_order",
    ACTION_MODIFY_AMOUNT: "set_action_modify_amount",
    ACTION_DEAL: "set_action_deal",
    ACTION_NEW_CHUNK: "clear",
}




