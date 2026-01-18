import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def boxplots(data, col):   
    '''
    Функция для отрисовки графика "ящик с усами".
    ''' 
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='star_type', y=col, data=data, palette='Set1')
    plt.title(f'Разброс значений признаков в поле {col} по типу звезды')
    plt.grid(True)
    plt.show();

def category_graph(data, columns):
    '''
    Функция для отрисовки столбчатых диаграмм для категориальных признаков.
    '''
    for column in columns:
        values = data[~data[column].isna()].value_counts(column)
        ax = values.plot.barh()

        xlim_max = values.max() * 1.3
        ax.set_xlim(right=xlim_max)

        for i, (count, label) in enumerate(zip(values, values.index)):
                        ax.text(count + values.max() * 0.01, i, str(count), va='center')

        plt.title(f'Столбчатая диаграмма распределения в поле "{column}"')
        plt.xlabel('Количество звезд')
        plt.ylabel('Значение')
        plt.show();

def numeric_graph(data, num_columns):
    '''
    Функция для отрисовки гистограммы и "ящика с усами" для количественных признаков.
    '''
    for column in num_columns:
        data[column].hist()
        plt.title(f'Гистограмма распределения в поле "{column}"')
        plt.xlabel('Значение')
        plt.ylabel('Количество звезд')
        plt.show()   
        
        data[column].plot(kind='box')
        plt.title(f'Разброс значений признаков в поле "{column}"')
        plt.grid(True)
        plt.show();

def plot_temp_bar(y_test, y_pred):
    '''
    Функция для отрисовки графика «Факт — Прогноз». 
    '''
    y_test_np = y_test.detach().numpy().flatten()
    y_pred_np = y_pred.detach().numpy().flatten()

    indices = np.arange(len(y_test_np))
    bar_width = 0.35

    plt.figure(figsize=(14, 8))
    plt.bar(indices, y_test_np, width=bar_width, label='Факт', color='skyblue')
    plt.bar(indices + bar_width, y_pred_np, width=bar_width, label='Прогноз', color='orange')
    plt.xlabel('Номер звезды в тестовой выборке')
    plt.ylabel('Температура звезды (K)')
    plt.legend()
    plt.tight_layout()
    plt.show();






