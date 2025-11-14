from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

@dataclass
class PlotCfg:
    '''
    Настройки для отрисовки графиков в разделе EDA.

    target_col: str
        Имя столбца с целевой переменной.
    churn_value: int
        Значение целевой переменной, соответствующей "ушедшим" клиентам.
    nan_label: str
        Ярлык для пропущенных значений в категориальных переменных. - ???
    percent_decimals: int
        Кол-во знаков после запятой при выводе процентов.
    '''
    target_col: str = 'target'
    churn_value: int = 1
    nan_label: str = 'Nan/Пусто'
    percent_decimals: int = 1

def category_graph(
        data: pd.DataFrame,
        columns: List[str],
        cfg: PlotCfg = PlotCfg()):
    
    '''
    Функция для компактного вывода графиков (в сетке) для категориальных переменных:
    1. Бинарные признаки - круговая диаграмма (pie-chart)
    2. Остальные - столбчатая диаграмма

    Параметры
    ---------
    data : pandas.DataFrame
        Датасет с признаками.
    columns : List of str
        Имена категориальных столбцов для отрисовки.
    '''
    
    # --- подготовка сетки для компактного отображения
    n = len(columns)
    nrows = (n + 1) // 2  
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows))
    
    for idx, column in enumerate(columns):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col] if nrows > 1 else axes[col]
        
        values = data.value_counts(column)
        percentages = (values / len(data) * 100).round(cfg.percent_decimals)

        # --- выбор типа графика
        if len(data[column].unique()) <= 2:
            # Круговая диаграмма
            ax.pie(values.values, 
                   autopct=lambda x: f'{x:.{cfg.percent_decimals}f}%\n({int(round(x* sum(values) / 100))})',
                   startangle=90,
                   labels=values.index)
            ax.set_title(f'Соотношение значений в поле "{column}"')
            
        else:
            # Вертикальная столбчатая диаграмма
            ax.bar(range(len(values)), values.values)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=45, ha='right')
            ax.set_title(f'Распределение в поле "{column}"')
            ax.set_ylabel('Количество абонентов')

            # Увеличиваем верхний предел для корректного отображения подписей
            max_value = max(values.values)
            ax.set_ylim(0, max_value * 1.2)  

            # Добавляем подписи
            for i, (count, pct) in enumerate(zip(values.values, percentages)):
                ax.text(i, count + max_value * 0.02,
                        f'{count}\n({pct}%)', 
                        va='bottom', ha='center', fontsize=9)
    
    # Удаляем лишние ячейки сетки
    if n % 2 != 0:
        if nrows > 1:
            axes[-1, -1].axis('off')
        else:
            axes[-1].axis('off')
    
    plt.tight_layout()
    plt.show()

def numeric_graph(
        data: pd.DataFrame,
        num_cols: List[str],
        cfg: PlotCfg = PlotCfg()):
    '''
    Функция для вывода гистограммы и ящика с усами для количественных переменных.

    Параметры
    ---------
    data : pandas.DataFrame
        Датасет с признаками.
    num_cols : List of str
        Имена количественных столбцов для отрисовки.
    '''
    for column in num_cols:
        data[column].hist(bins=30)
        plt.title(f'Гистограмма распределения в поле "{column}"')
        plt.xlabel('Значение')
        plt.ylabel('Количество абонентов')
        plt.show()   

        data[column].plot(kind='box')
        plt.title(f'Разброс значений признаков в поле "{column}"')
        plt.grid(True)
        plt.show()

def category_graph_compare(
        data: pd.DataFrame,
        cat_columns: List[str],
        cfg: PlotCfg = PlotCfg()):
    
    '''
    Сравнивает распределения категорий между группами в процентах (ушедшие/оставшиеся).

    Параметры
    ---------
    data : pandas.DataFrame
        Датасет с признаками.
    columns : List of str
        Имена категориальных столбцов для отрисовки.
    '''

    churned_data = data.query(f'{cfg.target_col} == @cfg.churn_value')
    retained_data = data.query(f'{cfg.target_col} != @cfg.churn_value')

    for column in cat_columns:
        # --- считаем % по категориям
        churned_counts = churned_data[column].value_counts()
        churned_total = len(churned_data)
        
        retained_counts = retained_data[column].value_counts()
        retained_total = len(retained_data)
        
        churned_pct = (churned_counts / churned_total * 100).round(cfg.percent_decimals)
        retained_pct = (retained_counts / retained_total * 100).round(cfg.percent_decimals)

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(churned_pct))
        bar_width = 0.35
        
        bars1 = ax.bar(x_pos - bar_width/2, churned_pct.values, 
                      bar_width, label='Ушедшие клиенты', alpha=0.8)

        bars2 = ax.bar(x_pos + bar_width/2, retained_pct.values, 
                      bar_width, label='Оставшиеся клиенты', alpha=0.6)
        
        ax.set_ylim(0, 105)  

        # --- формирование подписей над колонками
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_x() + 1,  
                   f'{bar1.get_height():.{cfg.percent_decimals}f}%\n({churned_counts.iloc[i]})', 
                   ha='center', va='bottom', fontsize=8)

            ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_x() + 1,  
                   f'{bar2.get_height():.{cfg.percent_decimals}f}%\n({retained_counts.iloc[i]})', 
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Значение')
        ax.set_ylabel('Доля, %')
        ax.set_title(f'Сравнение распределения в поле "{column}"')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(churned_pct.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def numeric_graph_compare(
        data: pd.DataFrame,
        num_cols: List[str],
        cfg: PlotCfg = PlotCfg()):
    '''
    Сравнивает гистограммы для числовых признаков (ушедшие/оставшиеся).

    Параметры
    ---------
    data : pandas.DataFrame
        Датасет с признаками.
    num_cols : List of str
        Имена количественных столбцов для отрисовки.
    '''
    churned_data = data.query(f'{cfg.target_col} == @cfg.churn_value')
    retained_data = data.query(f'{cfg.target_col} != @cfg.churn_value')

    for column in num_cols:
        ax = churned_data.plot(kind='hist', 
                       y=column, 
                       histtype='step',
                       linewidth=5, 
                       alpha=0.7, 
                       label='Ушедшие абоненты',
                       density=True 
                      )

        retained_data.plot(kind='hist', 
                  y=column, 
                  histtype='step',
                  linewidth=5, 
                  alpha=0.7, 
                  label='Оставшиеся абоненты',
                  ax=ax, 
                  grid=True, 
                  legend=True,
                  density=True 
                )
        plt.title(f'Распределение доли среди ушедших и оставшихся абонентов в поле {column}')
        plt.xlabel(f'{column}')
        plt.ylabel('Количество абонентов')
        plt.show()