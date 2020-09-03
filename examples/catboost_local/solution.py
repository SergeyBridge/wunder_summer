
import sys
import catboost

sys.path.append("../../scorer")
import orderbook_fast as ob


from my_orderbook import MyOrderBook

# SIDE_BID = 0
# SIDE_ASK = 1

catboost_myob = MyOrderBook()


clf = catboost.CatBoostClassifier()
clf.load_model(fname="wunder.model")

def process_event_and_predict_proba(event, orderbook):

    if event.action == ob.Action.DEAL:
        catboost_myob.set_last_deal(event)
    elif event.action == ob.Action.NEW_CHUNK:
        catboost_myob.clear()

    if not event.need_prediction:
        return None
    
    features = catboost_myob.get_features(orderbook, max_index=13)

    proba = clf.predict([features])[0]

    return proba

if __name__ == "__main__":
    from scorer import Scorer
    scoring = Scorer("../../data/train_small_A.npz")
    roc_auc, (true_ys, pred_probas) = scoring.score(process_event_and_predict_proba)

