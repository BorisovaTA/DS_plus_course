import pandas as pd
import matplotlib.pyplot as plt

def hist_with_median(data, col, **hist_params):
    median_value = data[col].median()

    data[col].hist(**hist_params)
    plt.title(f'Распределение количества квартир в поле {col}')
    plt.xlabel(f'Значения в поле {col}')
    plt.ylabel('Количество квартир')

    plt.axvline(median_value, color='red', linestyle='--', linewidth=2, 
            label=f'Медиана = {median_value:.2f}')

    plt.legend()
    plt.show();
