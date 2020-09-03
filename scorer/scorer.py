import sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from orderbook import OrderBook, EventPlayer

# Чтобы использовать быстрый ордербук раскомментируйте строку:
# from orderbook_fast import OrderBook, EventPlayer

class Scorer:
    def __init__(self, fname):
        self.fname = fname
        self.event_player = EventPlayer(fname)

    def _check_return_value(self, event, pred_proba):
        if event.need_prediction:
            if pred_proba is None:
                raise ValueError("You should return probability if event.need_prediction == True")
            if not (0 <= pred_proba <= 1):
                raise ValueError("Predicted probability is not in [0, 1] range")
        else:
            if pred_proba is not None:
                raise ValueError("Return probability should be None if event.need_prediction == False")

    def score(self, process_event_func):
        Ys = []
        pred_probas = []
        orderbook = OrderBook()

        for event in tqdm(self.event_player.iget_events(), 
                            total=len(self.event_player),
                            desc="scoring"):        
            orderbook.apply_event(event)
            pred_proba = process_event_func(event, orderbook)
            self._check_return_value(event, pred_proba)

            if not event.need_prediction:
                continue

            Ys.append(event.Y)
            pred_probas.append(pred_proba)

        roc_auc = roc_auc_score(Ys, pred_probas)
        print(f"\nroc_auc_score = {roc_auc:.3f}")
        return roc_auc, (Ys, pred_probas)


# if __name__ == "__main__":
#   file_to_score = sys.argv[1]
#   scorer = Scorer(file_to_score)

#   roc_auc, (Ys, Ps) = scorer.score(process_event_and_predict_proba)
#   print("roc_auc = {:.3f}, len(Ys) = {}".format(roc_auc, len(Ys)))
