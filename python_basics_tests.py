import pytest

from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,
)

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)


def test_count_vowels():
    assert count_vowels(TextInput(value="hello")) == 2
    assert count_vowels(TextInput(value="rhythm")) == 0
    assert count_vowels(TextInput(value="")) == 0


def test_has_unique_characters():
    assert has_unique_characters(TextInput(value="abc")) is True
    assert has_unique_characters(TextInput(value="hello")) is False
    assert has_unique_characters(TextInput(value="")) is True


def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(value=1)) == 1
    assert count_one_bits(PositiveIntegerInput(value=7)) == 3
    assert count_one_bits(PositiveIntegerInput(value=255)) == 8


def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(value=4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(value=39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(value=999)) == 4


def test_mse():
    data = VectorPairInput(predicted=[1, 2, 3], expected=[1, 2, 3])
    assert mse(data) == pytest.approx(0.0)

    data = VectorPairInput(predicted=[1, 2, 3], expected=[2, 3, 4])
    assert mse(data) == pytest.approx(1.0)

    data = VectorPairInput(predicted=[1, 3], expected=[2, 1])
    assert mse(data) == pytest.approx(2.5)


def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(value=1)) == ""
    assert prime_factorization(PositiveIntegerInput(value=12)) == "(2**2)(3)"
    assert prime_factorization(PositiveIntegerInput(value=97)) == "(97)"


def test_pyramid():
    assert pyramid(PositiveIntegerInput(value=1)) == 1
    assert pyramid(PositiveIntegerInput(value=14)) == 3
    assert pyramid(PositiveIntegerInput(value=2)) == "It is impossible"


def test_is_balanced_number():
    assert is_balanced_number(PositiveIntegerInput(value=121)) is True
    assert is_balanced_number(PositiveIntegerInput(value=123)) is False
    assert is_balanced_number(PositiveIntegerInput(value=123321)) is True
