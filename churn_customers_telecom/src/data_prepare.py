import pandas as pd

def snake_columns_rename(data: pd.DataFrame):
    '''
    Преобразовываем имена колонок в snake_case.

    Параметры
    ---------
    data : pandas.DataFrame
        Датасет с признаками.
    '''
    data.columns = (data.columns
        .str.replace(r"([a-z0-9])([A-Z])", r"\1 \2", regex=True) 
        .str.replace(" ", "_")                                   
        .str.lower()                                             
        .str.replace(r"_+", "_", regex=True)                     
        .str.strip("_") 
    )
    return data

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
    display(data.isna().sum())
    print('Количество уникальных ID:', len(data['customer_id'].unique()))
    print('Количество явных дубликатов:', data.duplicated().sum())