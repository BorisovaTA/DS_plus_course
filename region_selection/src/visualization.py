import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch

def numeric_graph(data):
    '''
    Функция для отрисовки гистограмм для количественных признаков.
    '''
    for column in data.columns:
        if data[column].dtype == 'float':
            data[column].hist(bins=100)
            plt.title(f'Гистограмма распределения в поле "{column}"')
            plt.xlabel('Значение')
            plt.ylabel('Количество')
            plt.show()   
        else:
            continue

def corr_matrix(data, method='pearson'):
    '''
    Функция для отрисовки матрицы корреляций.
    '''
    num_columns = data.select_dtypes(include='number').columns.tolist()
    plt.figure(figsize=(12, 8))
    sns.heatmap(data[num_columns].corr(method=method), annot=True, cmap='cividis') 
    plt.show();

def check_scaler(data):
    '''
    Функция для визуализации результатов масштабирования количественных признаков.
    '''
    num_columns = data.select_dtypes(include='number').columns.tolist()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    data[num_columns].plot(kind='hist', bins=10, ax=axes[0])
    axes[0].set_title('Гистограмма распределения')
    data[num_columns].plot(kind='box', ax=axes[1], rot=45)
    axes[1].set_title('Разброс значений признаков')
    plt.show();

def compare_regions_profits(regions_data, region_names):
    '''
    Функция для отрисовки графиков сравнения прибыли по регионам:
    - график "ящик с усами" для распределения расчетных прибылей скважин в регионе
    - сравнительный график по средней прибыли и риску убытков для каждого региона
    '''
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # "Ящик с усами"
    ax1 = axes[0]

    all_profits = []
    all_labels = []

    for i, (profits, _, _) in enumerate(regions_data):
        all_profits.append(profits)
        all_labels.append(f'{region_names[i]}\n(n={len(profits)})')

    boxplot = ax1.boxplot(all_profits, patch_artist=True, vert=True)
    ax1.set_xticklabels(all_labels, fontsize=11)

    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(boxplot['boxes'], colors[:len(all_profits)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

    ax1.set_title('Сравнение распределения прибыли по регионам',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Прибыль, млн.руб.', fontsize=12)
    ax1.set_xlabel('Регион', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Средняя прибыль и риск по регионам
    ax2 = axes[1]

    means = [data[1] for data in regions_data]
    risks = [data[2] * 100 for data in regions_data]

    x = np.arange(len(region_names))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2, means, width,
        color='skyblue', edgecolor='black', zorder=3
    )

    ax2_twin = ax2.twinx()

    ax2.set_zorder(2)
    ax2_twin.set_zorder(1)
    ax2.patch.set_visible(False)

    bars2 = ax2_twin.bar(
        x + width / 2, risks, width,
        color='lightcoral', edgecolor='black', alpha=0.7, zorder=3
    )

    ax2.bar_label(bars1, padding=3, fmt='%.1f', fontsize=10)
    ax2_twin.bar_label(bars2, padding=3, fmt='%.1f%%', fontsize=10)

    ax2.set_title('Сравнение ключевых метрик по регионам',
                  fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel('Регион', fontsize=12)
    ax2.set_ylabel('Средняя прибыль, млн.руб.', fontsize=12)
    ax2_twin.set_ylabel('Риск убытков, %', fontsize=12)

    ax2.set_xticks(x)
    ax2.set_xticklabels(region_names, fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)

    ax2.set_ylim(top=ax2.get_ylim()[1] * 1.12)
    ax2_twin.set_ylim(top=ax2_twin.get_ylim()[1] * 1.12)

    legend_handles = [
        Patch(facecolor='skyblue', edgecolor='black', alpha=1,
              label='Средняя прибыль, млн.руб.'),
        Patch(facecolor='lightcoral', edgecolor='black', alpha=1,
              label='Риск убытков, %')
    ]

    leg = ax2.legend(
        handles=legend_handles,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.02),
        framealpha=1,
        facecolor='white',
        edgecolor='black'
    )
    leg.set_zorder(100) 

    fig.suptitle('Сравнительный анализ прибыльности регионов',
                 fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show();

