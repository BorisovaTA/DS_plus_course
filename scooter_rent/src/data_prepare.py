import pandas as pd

def data_info(data, detailed=True):
    '''
    Функция для вывода описательных статистик и первичной общей информации о датасете.
    -------
    Параметры:
    detailed=True - выводит доп.информацию о кол-ве пропущенных значений и дубликатов при наличии.
    '''
    print('=' * 80)
    print(f'ОБЗОР ДАТАСЕТА')
    print('=' * 80)

    print(f'Основная информация:')
    print(f'Размер данных: {data.shape[0]} строк × {data.shape[1]} столбцов')
    display(data.head(2))
    data.info()
    display(data.describe().T)

    if detailed:
        print('=' * 80)
        print('Пропущенные значения:')
        missing_cols = data.isna().sum()
        missing_cols = missing_cols[missing_cols > 0]
        
        if len(missing_cols) > 0:
            print('Столбцы с пропусками:')
            for col, count in missing_cols.sort_values(ascending=False).items():
                print(f'{col}: {count:,} ({count / data.shape[0] * 100:.2f}%)')
        else:
            print('Пропущенных значений нет.')

        print('=' * 80)
        print('Наличие дубликатов:')
        duplicates = data.duplicated().sum()
        if duplicates == 0:
            print('Явных дубликатов нет')
        else:
            print(f'Найдено дубликатов: {duplicates:,} ({duplicates/data.shape[0]*100:.2f}%)')

def monthly_income(data, 
                  subscription='Наличие подписки',
                  rides='Количество поездок', 
                  time='Общее время поездок'):
    '''Функция для расчета месячной выручки по формуле:
    стоимость старта поездки × количество поездок + \
     + стоимость одной минуты поездки × общая продолжительность всех поездок в минутах + \
     + стоимость подписки.
    '''
    data_free = data.loc[data[subscription] == "free"].copy()
    data_free['Месячная выручка'] = 50 * data_free[rides] + 8 * data_free[time]
    data_ultra = data.loc[data[subscription] == "ultra"].copy()
    data_ultra['Месячная выручка'] = 6 * data_ultra[time] + 199
    data = pd.concat([data_free, data_ultra])
    return data
