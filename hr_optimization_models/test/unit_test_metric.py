import numpy as np
import sys
from pathlib import Path

root = Path(__file__).parent.parent 
sys.path.insert(0, str(root))

from src.smape_metric import smape

def test_smape_equal() -> None:
    '''
    Проверка корректности расчета метрики SMAPE при нулевой ошибке.
    '''
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    assert smape(y_true, y_pred) == 0

def test_smape() -> None:
    '''
    Проверка корректности расчета метрики SMAPE.
    '''
    y_true = np.array([1, 1])
    y_pred = np.array([1.5, 1.857])
    assert abs(smape(y_true, y_pred) - 50) < 1e-2

