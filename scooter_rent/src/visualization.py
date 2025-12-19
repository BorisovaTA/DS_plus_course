import matplotlib.pyplot as plt
import seaborn as sns

def barh_value_counts(data, column):
    '''
    Функция для отрисовки горизонтальной столбчатой диаграммы.
    '''

    col_counts = data.value_counts(column).sort_values(ascending=True)
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(col_counts))

    ax = sns.barplot(
        x=col_counts.values, 
        y=col_counts.index, 
        hue=col_counts.index,  
        palette=colors, 
        orient='h'
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3)

    plt.title(f'Распределение пользователей по полю "{column}"', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Количество пользователей')
    plt.ylabel(f'Значения в поле "{column}"')

    plt.show();

def pie_value_counts(data, column):
    
    '''
    Функция для отрисовки круговой диаграммы с выводом долей в процентном соотношении.
    '''

    col_counts = data.value_counts(column)
    colors = sns.color_palette('Set2', len(col_counts))

    wedges, texts, autotexts = plt.pie(
                                col_counts.values,
                                labels=col_counts.index,
                                autopct='%1.1f%%',
                                startangle=90,
                                colors=colors,
                                textprops={'fontsize': 12, 'fontweight': 'bold'},
                                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                                explode=[0.05] * len(col_counts)  
                            )

    plt.legend(
                wedges,
                [f'{label}: {count:,}' for label, count in col_counts.items()],
                loc='upper right'
            )

    plt.title(f'Соотношение пользователей в поле "{column}"', 
                fontsize=16, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.show();

def hist(data, column, **hist_params):
    data[column].hist(**hist_params)
    plt.title(f'Распределение пользователей в поле "{column}"')
    plt.xlabel(f'Значения в поле "{column}"')
    plt.ylabel('Количество пользователей')
    plt.show();


def density_hist(data, column, split_col):
    '''
    Функция для отрисовки нормализованной гистограммы плотности для несбалансированных групп.
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

    plt.title(f'Нормализованное распределение поездок по полю "{column}"\n(учитывает разное количество поездок в группах)', 
            fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(f'Значение в поле "{column}"', fontsize=12)
    plt.ylabel('Плотность вероятности', fontsize=12)

    counts = data[split_col].value_counts()
    new_labels = [f'{label} (n={counts.get(label,0):,})' for label in hue_order]

    ax.legend(handles=ax.legend_.legend_handles,
              labels=new_labels,
              title='Тип подписки',
              fontsize=10)

    plt.tight_layout()
    plt.show();
