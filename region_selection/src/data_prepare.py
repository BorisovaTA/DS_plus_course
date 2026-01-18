import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
        if data['id'].is_unique:
            print('Все значения в поле с идентификатором уникальны')
        else:
            print('В поле с идентификатором есть дубликаты')

def data_prepare(data, RANDOM_STATE):
    '''Удаление лишних колонок, выделение таргета'''
    X = data.drop(['id', 'product'], axis=1)
    y = data['product']
    
    '''Разбиение данных на выборки'''
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    '''Масштабирование количественных признаков'''
    num_columns = X_train.select_dtypes(include='number').columns.tolist()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[num_columns])
    X_val_scaled = scaler.transform(X_val[num_columns])
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=num_columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=num_columns)
    display(X_train_scaled.head(2)), display(X_val_scaled.head(2))
    return X_train_scaled, X_val_scaled, y_train, y_val


