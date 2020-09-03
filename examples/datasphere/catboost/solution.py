
import sys
import lightgbm as lg

sys.path.append("../../scorer")
import orderbook as ob

SIDE_BID = 0 
SIDE_ASK = 1

def get_simple_features_from_orderbook(orderbook, max_index=2):
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
    return features


def get_simple_deals_features(last_deals, orderbook):
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
            features += [cur_mean_price - deal_event.price, cur_time - deal_event.time, deal_event.amount]
    return features


clf = lg.Booster(model_file="wunder.model")
last_deals = [None, None]

def process_event_and_predict_proba(event, orderbook):
    if event.action == ob.Action.DEAL:
        last_deals[event.side] = event
    elif event.action == ob.Action.NEW_CHUNK:
        last_deals[:] = [None, None]
        
    if not event.need_prediction:
        return None
    
    features = get_simple_features_from_orderbook(orderbook)
    features += get_simple_deals_features(last_deals, orderbook)    
    proba = clf.predict([features])[0]
    return proba

if __name__ == "__main__":
    from scorer import Scorer
    scoring = Scorer("../../data/train_small_A.npz")
    roc_auc, (true_ys, pred_probas) = scoring.score(process_event_and_predict_proba)

