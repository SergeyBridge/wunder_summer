
import sys
import catboost

sys.path.append("../../scorer")
import orderbook_fast as ob

# import my_orderbook
from my_orderbook import MyOrderBook, action_handler
catboost_myob = MyOrderBook()


clf = catboost.CatBoostClassifier()
clf.load_model(fname="wunder.model")

def process_event_and_predict_proba(event, orderbook):


    catboost_myob.__getattribute__(action_handler[event.action])

    if not event.need_prediction:
        return None

    features = catboost_myob.get_features(event, orderbook, max_index=13)

    proba = clf.predict_proba([features])[0, 1]

    return proba

if __name__ == "__main__":
    from scorer import Scorer
    scoring = Scorer("../../data/train_small_A.npz")
    roc_auc, (true_ys, pred_probas) = scoring.score(process_event_and_predict_proba)

