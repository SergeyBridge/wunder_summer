from collections import namedtuple

EventBody = namedtuple('EventBody', [
                       'time', 'id', 'action', 'type', 'side', 'price', 'amount', 'is_snapshot', 'Y'])

class Event(EventBody):
    @property
    def need_prediction(self):
        return self.Y >= 0
