cimport numpy as np

cdef class Event:
    cdef public:
        np.ndarray np_event
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
        self.np_event = np_event
