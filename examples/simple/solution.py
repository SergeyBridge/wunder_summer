SIDE_BID = 0
SIDE_ASK = 1

def process_event_and_predict_proba(event, orderbook):
    '''Функция которую реализовать в файле solution.py для отправки решения'''
    if not event.need_prediction:
        # Для моментов event.need_prediction == False не нужно ничего предсказывать
        return None

    # Пример получения лучших цен на покупку и продажу
    buy_price = orderbook.get_best_price(SIDE_BID)
    sell_price = orderbook.get_best_price(SIDE_ASK)

    # 0 - это индекс лучшей цены на покупку / продажу
    # Чем больше индекс - тем дальше от лучших цен уровень
    best_ix = 0

    # Получаем объемы на лучших ценах (best_ix = 0)
    best_buy_volume = orderbook.get_price_level_at_ix(SIDE_BID, best_ix).get_volume()
    best_sell_volume = orderbook.get_price_level_at_ix(SIDE_ASK, best_ix).get_volume()

    # Делаем эвристическое предсказание вероятности
    predict_proba = 0.75 if best_buy_volume > best_sell_volume else 0.25
    return predict_proba
    

if __name__ == "__main__":
    import sys
    sys.path.append("../../scorer")
    from scorer import Scorer
    # Загружаем данные в скорер
    scoring = Scorer("../../data/train_small_A.npz")
    # Оцениваем наше решение
    roc_auc, (true_ys, pred_probas) = scoring.score(process_event_and_predict_proba)

    '''Попробуйте оправить архив с файлом содержашим solution.py чтобы протестировать механику отправки'''
