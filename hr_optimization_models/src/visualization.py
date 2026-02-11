import matplotlib.pyplot as plt
import seaborn as sns

def category_graph(data, columns):
    '''
    Функция для отрисовки стобчатых диаграмм в зависимости от количества 
    уникальных значений в категориальных признаках.
    '''
    for column in columns:
        if len(data[column].unique()) <= 2:
            values = data.value_counts(column)
            ax = values.plot.pie(
                autopct=lambda x: f'{x:.1f}%\n({(x * sum(values) / 100).round(0)})',
                startangle=90,
                ylabel='',       
                title=f'Соотношение значений в поле "{column}"'
            )
            plt.show()
        else:
            ax = data.value_counts(column).plot.barh()
            plt.title(f'Столбчатая диаграмма распределения в поле "{column}"')
            plt.xlabel('Количество сотрудников')
            plt.ylabel('Значение')
            plt.show()

def numeric_graph(data, num_columns, exception_columns):
    '''
    Функция для построения гистограмм.
    '''
    for column in num_columns:
        if data[column].dtype == 'float' or [column] == exception_columns:
            data[column].hist()
            plt.title(f'Гистограмма распределения в поле "{column}"')
            plt.xlabel('Значение')
            plt.ylabel('Количество сотрудников')
            plt.show()   
        else:
            data[column].value_counts().sort_index().plot(kind='bar') 
            plt.title(f'Гистограмма распределения в поле "{column}"')
            plt.xlabel('Значение')
            plt.ylabel('Количество сотрудников')
            plt.show()
        
        data[column].plot(kind='box')
        plt.title(f'Разброс значений признаков в поле "{column}"')
        plt.grid(True)
        plt.show()

def compare_cat_features(data_high_satisfaction, data_low_satisfaction):
    '''
    Функция для построения сравнительного графика распределения категориальных признаков
    среди сотрудников с высоким и низким уровнем удовлетворенности работой.
    '''
    columns = data_high_satisfaction.select_dtypes(exclude='number').columns.tolist()
    for column in columns:
        fig, ax = plt.subplots(figsize=(8,5))
        ax = (data_high_satisfaction.value_counts(column).plot(kind='bar', 
                                                               label='Высокий уровень удовлетворенности',
                                                               ax=ax,
                                                               legend=True)
            )
        
        (data_low_satisfaction.value_counts(column).plot(kind='bar', 
                                                         label='Низкий уровень удовлетворенности',
                                                         ax=ax, 
                                                         alpha=0.6,
                                                         grid=True, 
                                                         legend=True)
        )
        plt.title(f'Гистограмма распределения в поле "{column}"')
        plt.xlabel('Значение')
        plt.ylabel('Количество сотрудников')
        plt.show();

def category_graph_normalize(data, data_part, cat_columns):
    '''
    Функция для отрисовки стобчатых диаграмм с учетом неравномерности распределения данных по категориям. 
    Количество наблюдений нормализовано относительно исследуемого признака. 
    '''
    for column in cat_columns:
        value_counts = data[column].value_counts()
    
        if len(value_counts) <= 2:
            values = data_part[column].value_counts(normalize=True)
            ax = values.plot.pie(
                autopct=lambda x: f'{x:.1f}%\n({(x * sum(value_counts) / 100).round(0)})',
                startangle=90,
                ylabel='',       
                title=f'Соотношение значений в поле "{column}"'
            )
            plt.show()
        else:
            quit_ratio = data.groupby(column)['quit'].value_counts(normalize=True).unstack()['yes']
            quit_ratio = quit_ratio.fillna(0).sort_values()

            ax = quit_ratio.plot.barh()
            plt.title(f'Доля ушедших сотрудников в поле "{column}"')
            plt.xlabel('Доля ушедших')
            plt.ylabel(column)
            plt.show()

def numeric_graph_normalize(data, data_part, num_columns, exception_columns):
    '''
    Функция для отрисовки гистограмм с учетом неравномерности распределения данных по категориям. 
    Количество наблюдений нормализовано относительно исследуемого признака. 
    '''

    for column in num_columns:
        if data_part[column].dtype == 'float' or column in exception_columns:
            data_part[column].hist(density=True)
            plt.title(f'Гистограмма плотности распределения в поле "{column}" среди уволившихся')
            plt.xlabel('Значение')
            plt.ylabel('Плотность')
            plt.show()   
        else:
            quit_ratio = data.groupby(column)['quit'].value_counts(normalize=True).unstack().fillna(0)['yes']
            quit_ratio = quit_ratio.sort_values()
            ax = quit_ratio.sort_index().plot(kind='bar') 
            plt.title(f'Доля ушедших сотрудников в поле "{column}"')
            plt.xlabel(column)
            plt.ylabel('Доля ушедших')
            plt.show()
        
        # Ящик с усами (оставляем без изменений)
        data_part[column].plot(kind='box')
        plt.title(f'Разброс значений признаков в поле "{column}" среди уволившихся')
        plt.grid(True)
        plt.show()

def density_hist(data, column, split_col):
    '''
    Функция для отрисовки совместных нормализованных гистограммы плотности для несбалансированных групп.
    '''        
    hue_order = data[split_col].unique()

    ax = sns.histplot(
                data=data,
                x=column,
                bins=50,
                hue=split_col,
                hue_order=hue_order,
                stat="density",  
                common_norm=False, 
                kde=True,
                edgecolor='white', 
                alpha=0.7,
                palette="Set2" 
            )

    plt.title(f'Нормализованное распределение по полю "{column}"\n(учитывает дисбаланс в целевой переменной)', 
            fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(f'Значение в поле "{column}"', fontsize=12)
    plt.ylabel('Плотность вероятности', fontsize=12)

    counts = data[split_col].value_counts()
    new_labels = [f'{label} (n={counts.get(label,0):,})' for label in hue_order]

    ax.legend(handles=ax.legend_.legend_handles,
              labels=new_labels,
              title=split_col,
              fontsize=10)

    plt.tight_layout()
    plt.show();
