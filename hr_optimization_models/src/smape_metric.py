import numpy as np

def smape(y_true, y_pred):
    '''
    Функция для расчета метрики SMAPE: абсолютная разность между наблюдаемым и
    предсказанным значениями деленая на полусумму их модулей.

    В отличие от MAPE, SMAPE симметрична: одинаково относится к ошибкам переоценки 
    и недооценки, деля абсолютную разницу на среднее арифметическое.

    Чем меньше значение SMAPE, тем точнее прогноз.
    '''
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = 100 * np.mean(numerator/denominator)
    return smape
