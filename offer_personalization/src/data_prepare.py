import re

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

        print('=' * 80)
        print('Уникальность id')
        if data.index.is_unique:
            print('Все значения в поле с идентификатором уникальны')
        else:
            print('В поле с идентификатором есть дубликаты')

def snake_case(data):
        '''
        Функция для приведения названий столбцов к "змеиному_регистру".
        '''
        data.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', i).replace(' ', '_').lower() for i in data.columns]
        print(data.columns)
        return data
