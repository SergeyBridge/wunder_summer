import numpy as np
cimport numpy as np
from sortedcontainers import SortedDict

DEF SIDE_BID = 0
DEF SIDE_ASK = 1

DEF ACTION_DELETE = 0
DEF ACTION_ADD = 1
DEF ACTION_MODIFY_AMOUNT = 2
DEF ACTION_DEAL = 3
DEF ACTION_NEW_CHUNK = 10

DEF MIN_PRICE = 0
DEF MAX_PRICE = 10**9

cdef class Event:
    cdef public:
        long long time
        int id
        int action
        int type
        int side
        int price
        int amount
        int is_snapshot
        int Y
        int need_prediction
    def __init__(self, np.ndarray np_event):
        self.time = np_event[0]
        self.id = np_event[1]
        self.action = np_event[2]
        self.type = np_event[3]
        self.side = np_event[4]
        self.price = np_event[5]
        self.amount = np_event[6]
        self.is_snapshot = np_event[7]
        self.Y = np_event[8]
        self.need_prediction = self.Y >= 0


cdef class Order:
    cdef public:
        int order_id
        int side
        int price
        int type
        int amount
        int initial_amount
        long long time

    def __init__(self, Event event):
        self.order_id = event.id
        self.side = event.side
        self.price = event.price
        self.type = event.type
        self.amount = event.amount
        self.initial_amount = event.amount
        self.time = event.time


cdef class PriceLevel:
    cdef public:
        int side
        int price
        list orders
        int num_orders 
        int total_amount
        dict id_to_order

    def __init__(self, int side, int price):
        self.side = side
        self.price = price
        self.orders = []
        self.num_orders = 0
        self.total_amount = 0
        self.id_to_order = {}

    def add_order(self, Order order):
        self.orders.append(order)
        self.num_orders += 1
        self.total_amount += order.amount
        self.id_to_order[order.order_id] = order

    cdef Order _get_order(self, int order_id):
        return self.id_to_order[order_id]

    def delete_order(self, int order_id):
        order = self._get_order(order_id)

        self.num_orders -= 1
        self.total_amount -= order.amount
        self.orders.remove(order)
        del self.id_to_order[order_id]

    def modify_order(self, int order_id, int new_amount):
        order = self._get_order(order_id)
        change_amount = order.amount - new_amount
        if change_amount <= 0:
            raise Exception("change_amount <= 0!")
        order.amount -= change_amount
        self.total_amount -= change_amount

    def get_volume(self):
        return self.total_amount

    def get_num_orders(self):
        return self.num_orders

    def get_orders(self):
        return self.orders


cdef class OrderBook:
    cdef public:
        int best_price[2]
        list events
        list price_levels
        long long time

    def __init__(self):
        self._clear()

    def _clear(self):
        self.best_price[:] = [MIN_PRICE, MAX_PRICE]
        self.price_levels = [SortedDict(), SortedDict()]
        self.time = 0
        self.events = []

    cdef int _get_empty_price(self, int side):
        return MIN_PRICE if side == SIDE_BID else MAX_PRICE

    cdef int _get_updated_best_price(self, side):
        if len(self.price_levels[side]) == 0:
            return self._get_empty_price(side)
        if side == SIDE_BID:
            return self.price_levels[side].peekitem(-1)[0]  # max_key
        return self.price_levels[side].peekitem(0)[0]  # min_key
 

    cdef PriceLevel _get_price_level(self, int side, int price):
        if price not in self.price_levels[side]:
            self.price_levels[side][price] = PriceLevel(side, price)
        return self.price_levels[side][price]

    def add_order(self, Order order):
        side, price = order.side, order.price
        self._get_price_level(side, price).add_order(order)
        self.best_price[side] = self._get_updated_best_price(side)

    cdef void _del_if_empty(self, int side, int price):
        if self.price_levels[side][price].get_num_orders() == 0:
            del self.price_levels[side][price]
            if price == self.get_best_price(side):
                self.best_price[side] = self._get_updated_best_price(side)

    def delete_order(self, int side, int price, int order_id):
        self.price_levels[side][price].delete_order(order_id)
        self._del_if_empty(side, price)
        
    def modify_order(self, int side, int price, int order_id, int new_amount):
        self.price_levels[side][price].modify_order(order_id, new_amount)
        self._del_if_empty(side, price)

    def apply_event(self, Event event):
        self.time = event.time
        self.events.append(event)

        if event.action == ACTION_DELETE:  # deletion
            self.delete_order(event.side, event.price, event.id)
        elif event.action == ACTION_MODIFY_AMOUNT:  # modifying
            self.modify_order(event.side, event.price, event.id, event.amount)
        elif event.action == ACTION_ADD:  # adding new order
            self.add_order(Order(event=event))
        elif event.action == ACTION_NEW_CHUNK: 
            self._clear()
        else:
            # print("Strange event: ", event)
            pass

    def get_best_price(self, int side):
        return self.best_price[side]

    def get_mean_price(self):
        return (self.best_price[SIDE_BID] + self.best_price[SIDE_ASK]) / 2.0

    def get_time(self):
        return self.time

    def get_events(self):
        return self.events

    def get_price_at_ix(self, int side, int index):
        return self.best_price[side] - index * (1 - 2 * side)

    def get_ix_at_price(self, int side, int price):
        return (self.best_price[side] - price) * (1 - 2 * side)

    def get_price_level_at_ix(self, int side, int ix):
        cdef int price = self.get_price_at_ix(side, ix)
        return self.price_levels[side].get(price, None)


class EventPlayer:
    def __init__(self, filename):
        self.events = np.load(filename)["events"]

    def iget_events(self):
        for event in self.events:
            yield Event(event)

    def __len__(self):
        return len(self.events)


#  ---  Not used ---
class Side:
    BID = 0
    ASK = 1

class Action:
    DELETE = 0
    ADD = 1
    MODIFY_AMOUNT = 2
    DEAL = 3
    NEW_CHUNK = 10
# --- END ---