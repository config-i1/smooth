"""Tests the my_linalg library."""
import numpy as np

from smooth.my_linalg import add, dot_product, array_sum, arma_dot_product

arr1 = np.array([1, 2])
arr2 = np.array([3, 4])


def test_add():
    """Test the add function."""
    assert add(1, 2) == 3


def test_dot_product():
    """Test the dot_product function"""
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])
    assert dot_product(arr1, arr2) == 11


def test_arma_dot_product():
    """Tests the arma_dot_product function"""
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])
    assert arma_dot_product(arr1, arr2) == 11


def test_array_sum():
    """Tests the array_sum function"""
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])
    assert np.all(array_sum(arr1, arr2) == np.array([4, 6]))
