import numpy as np
from sortedcontainers import SortedDict

from common import Event

class Side:
    BID = 0
    ASK = 1

class Action:
    DELETE = 0
    ADD = 1
    MODIFY_AMOUNT = 2
    DEAL = 3
    NEW_CHUNK = 10

MIN_PRICE = 0
MAX_PRICE = 10**9

DEBUG = False

class Order:
    def __init__(self, event):
        self.order_id = event.id
        self.side = event.side
        self.price = event.price
        self.type = event.type
        self.amount = event.amount
        self.initial_amount = event.amount
        self.time = event.time


class PriceLevel:
    def __init__(self, side, price):
        self.side = side
        self.price = price
        self.orders = []
        self.num_orders = 0
        self.total_amount = 0
        self.id_to_order = {}

    def add_order(self, order):
        self.orders.append(order)
        self.num_orders += 1
        self.total_amount += order.amount
        self.id_to_order[order.order_id] = order
        self._check_level()

    def _get_order(self, order_id):
        return self.id_to_order[order_id]

    def delete_order(self, order_id):
        order = self._get_order(order_id)

        self.num_orders -= 1
        self.total_amount -= order.amount
        self.orders.remove(order)
        del self.id_to_order[order_id]
        self._check_level()

    def modify_order(self, order_id, new_amount):
        order = self._get_order(order_id)
        change_amount = order.amount - new_amount
        assert change_amount > 0
        order.amount -= change_amount
        self.total_amount -= change_amount
        self._check_level()

    def get_volume(self):
        return self.total_amount

    def get_num_orders(self):
        return self.num_orders

    def get_orders(self):
        return self.orders

    def _check_level(self):
        if not DEBUG:
            return
        assert self.num_orders == len(self.orders)
        assert sum([o.amount for o in self.orders]) == self.total_amount
        


class OrderBook:
    def __init__(self):
        self._clear()

    def _clear(self):
        self.best_price = [MIN_PRICE, MAX_PRICE]
        self.price_levels = [SortedDict(), SortedDict()]
        self.time = 0
        self.events = []

    def _get_empty_price(self, side):
        return MIN_PRICE if side == Side.BID else MAX_PRICE

    def _get_updated_best_price(self, side):
        if len(self.price_levels[side]) == 0:
            return self._get_empty_price(side)
        if side == Side.BID:
            return self.price_levels[side].peekitem(-1)[0]  # max_key
        return self.price_levels[side].peekitem(0)[0]  # min_key

    def get_price_level(self, side, price):
        if price not in self.price_levels[side]:
            self.price_levels[side][price] = PriceLevel(side, price)
        return self.price_levels[side][price]

    def add_order(self, order):
        side, price = order.side, order.price
        self.get_price_level(side, price).add_order(order)
        self.best_price[side] = self._get_updated_best_price(side)

    def _del_if_empty(self, side, price):
        if self.price_levels[side][price].get_num_orders() == 0:
            del self.price_levels[side][price]
            if price == self.get_best_price(side):
                self.best_price[side] = self._get_updated_best_price(side)

    def delete_order(self, side, price, order_id):
        self.price_levels[side][price].delete_order(order_id)
        self._del_if_empty(side, price)
        
    def modify_order(self, side, price, order_id, new_amount):
        self.price_levels[side][price].modify_order(order_id, new_amount)
        self._del_if_empty(side, price)

    def apply_event(self, event):
        self.time = event.time
        self.events.append(event)

        if event.action == Action.DELETE:  # deletion
            self.delete_order(event.side, event.price, event.id)
        elif event.action == Action.MODIFY_AMOUNT:  # modifying
            self.modify_order(event.side, event.price, event.id, event.amount)
        elif event.action == Action.ADD:  # adding new order
            self.add_order(Order(event=event))
        elif event.action == Action.NEW_CHUNK: 
            self._clear()
        else:
            # print("Strange event: ", event)
            pass
        assert self.best_price[1] > self.best_price[0]

    def get_best_price(self, side):
        return self.best_price[side]

    def get_mean_price(self):
        return (self.best_price[Side.BID] + self.best_price[Side.ASK]) / 2.0

    def get_time(self):
        return self.time

    def get_events(self):
        return self.events

    def get_price_at_ix(self, side, index):
        return self.best_price[side] - index * (1 - 2 * side)

    def get_ix_at_price(self, side, price):
        return (self.best_price[side] - price) * (1 - 2 * side)

    def get_price_level_at_ix(self, side, ix):
        price = self.get_price_at_ix(side, ix)
        return self.price_levels[side].get(price, None)


class EventPlayer:
    def __init__(self, filename):
        self.events = np.load(filename)["events"]

    def iget_events(self):
        for event in self.events:
            yield Event(*event)

    def __len__(self):
        return len(self.events)

if __name__ == "__main__":
    player = EventPlayer("../data/train_small_A.npz")
    orderbook = OrderBook()

    from tqdm import tqdm
    for event in tqdm(player.iget_events(), total=len(player)):
        # print(event)
        orderbook.apply_event(event)