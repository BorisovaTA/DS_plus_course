import matplotlib.pyplot as plt

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
            values = data[column].value_counts()
            ax = values.plot.barh()

            xlim_max = values.max() * 1.3
            ax.set_xlim(right=xlim_max)

            for i, (count, label) in enumerate(zip(values, values.index)):
                ax.text(count + values.max() * 0.01, i, str(count), va='center')

            plt.title(f'Столбчатая диаграмма распределения в поле "{column}"')
            plt.xlabel('Количество')
            plt.ylabel('Значение')
            plt.tight_layout()
            plt.show();

def numeric_graph(data, num_columns, exception_columns, period=None):
    '''
    Функция для отрисовки стобчатых диаграмм или гистограмм для количественных признаков 
    в зависимости от типа данных (непрерывные значения или целочисленные).
    '''
    for column in num_columns:
        if data[column].dtype == 'float' or [column] == exception_columns:
            data[column].hist()
            if period is not None:
                plt.title(f'Гистограмма распределения в поле "{column}" за {period}')
            else:
                plt.title(f'Гистограмма распределения в поле "{column}"')
            plt.xlabel('Значение')
            plt.ylabel('Количество клиентов')
            plt.show()   
        else:
            data[column].value_counts().sort_index().plot.bar()
            if period is not None:
                plt.title(f'Столбчатая диаграмма данных в поле "{column}" за {period}')
            else:
                plt.title(f'Столбчатая диаграмма данных в поле "{column}"')
            plt.xlabel('Значение')
            plt.ylabel('Количество клиентов')
            plt.show()
        
        data[column].plot(kind='box')
        plt.title(f'Разброс значений признаков в поле "{column}"')
        plt.grid(True)
        plt.show();

def hist_with_median(data, col, period=None, **hist_params):
    '''
    Функция для отрисовки гистограмм для количественных признаков 
    с отображением медианы и ее значения.
    '''
    median_value = data[col].median()

    data[col].hist(**hist_params)

    if period is not None:
        plt.title(f'Распределение в поле "{col}" за {period}')
    else:
        plt.title(f'Распределение в поле "{col}"')

    plt.xlabel(f'Значения в поле "{col}"')
    plt.ylabel('Количество клиентов')

    plt.axvline(median_value, color='red', linestyle='--', linewidth=2, 
            label=f'Медиана = {median_value:.2f}')

    plt.legend()
    plt.show();

