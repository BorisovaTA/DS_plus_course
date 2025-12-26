import matplotlib.pyplot as plt
import seaborn as sns

def bar_plot(data, x_col, y_col, title="", xlabel="", ylabel="",
              xticks=None, annotate=False):
    '''
    Функция для отрисовки столбчатых диаграмм для исследовательского анализа.
    '''
    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.bar(data[x_col], data[y_col])

    if xticks is not None:
        ax.set_xticks(list(xticks))
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)

    if annotate:
        for b, v in zip(bars, data[y_col].values):
            ax.text(b.get_x()+b.get_width()/2, v+0.8, f'{v:.2f}%', ha='center')

    plt.tight_layout()
    plt.show();

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

def numeric_graph(data, num_columns):
    '''
    Функция дял построения гистограмм.
    '''
    for column in num_columns:
        data[column].hist(bins=100)
        plt.title(f'Гистограмма распределения в поле "{column}"')
        plt.xlabel('Значение')
        plt.ylabel('Количество покупок')
        plt.show()   
        
        data[column].plot(kind='box')
        plt.title(f'Разброс значений признаков в поле "{column}"')
        plt.grid(True)
        plt.show();

def plot_confusion_matrix(data):
    '''
    Функция для отрисовки матрицы ошибок
    '''
    plt.figure(figsize=(6,5))
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Не виновен (0)', 'Виновен (1)'],
                yticklabels=['Не виновен (0)', 'Виновен (1)'])
    plt.xlabel('Предсказание')
    plt.ylabel('Фактическое значение')
    plt.title('Матрица ошибок')
    plt.show();

def plot_metrics(precision, recall):
    '''
    Функция для визуализации значений и соотношения метрик: точность/полнота.
    '''
    plt.figure(figsize=(6,4))
    sns.barplot(x=['Точность (Precision)', 'Полнота (Recall)'], 
                y=[precision, recall], palette='Set2')
    plt.ylim(0,1)
    plt.title('Метрики качества на тестовой выборке')
    for i, v in enumerate([precision, recall]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.show();


