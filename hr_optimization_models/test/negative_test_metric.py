import numpy as np
import sys
from pathlib import Path

root = Path(__file__).parent.parent 
sys.path.insert(0, str(root))

from src.smape_metric import smape

def test_zero_arrays():
    '''Тест на нулевые значения в обоих массивах (деление 0/0)'''
    try:
        result = smape(np.array([0, 0, 0]), np.array([0, 0, 0]))
        if np.isnan(result):
            pass  
    except Exception:
        raise AssertionError('Zero arrays not handled')

def test_different_lengths():
    '''Тест на массивы разной длины'''
    try:
        smape(np.array([1, 2, 3]), np.array([1, 2]))
    except (ValueError, TypeError):
        pass
    else: 
        raise AssertionError('Arrays with different lengths not handled')
    
def test_non_numeric_input():
    '''Тест на некорректные типы значений в массивах'''
    try:
        smape([1, 2, 'text'], [1, 2, 3])
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError('Non-numeric input not handled')

def test_none_input():
    '''Тест на наличие путых значений в массивах'''
    try:
        smape([1, None, 3], [1, 2, 3])
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError('None input not handled')

def test_inf_values():
    try:
        smape([1, np.inf, 3], [1, 2, 3])
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError('Inf values not handled')
