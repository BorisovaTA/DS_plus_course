import pandas as pd

def data_info(data: pd.DataFrame):
    '''
    Общая информация о таблице. Разведывательный анализ данных.

    Параметры
    ---------
    data : pandas.DataFrame
        Датасет с признаками.
    '''
    print('Размер таблицы', data.shape)
    display(data.head(2))
    data.info()
    display(data.describe().T)
    display(data.isna().sum().sort_values(ascending=False))
    print('Количество явных дубликатов:', data.duplicated().sum())
