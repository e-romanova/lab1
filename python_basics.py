from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    return sum(text.count(vowel) for vowel in "aeiou")


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    for i in text:
        for j in text[text.index(i) + 1::]:
            if i == j:
                return False
    return True


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    bin_number = bin(number)[2::]
    return bin_number.count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    n = 0
    while len(str(number)) > 1:
        new_number = 1
        for i in str(number):
            new_number *= int(i)
        number = new_number
        n +=1
    return n

def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    summa = 0
    for i in range(len(predicted)):
        summa += (predicted[i] - expected[i]) ** 2
    return summa / len(predicted)

def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    result = ""
    divisor = 2

    while divisor * divisor <= number:
        power = 0

        while number % divisor == 0:
            number //= divisor
            power += 1

        if power == 1:
            result += f"({divisor})"
        elif power > 1:
            result += f"({divisor}**{power})"

        divisor += 1

    if number > 1:
        result += f"({number})"

    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    summa = 0
    k = 0

    while summa < cube_count:
        k += 1
        summa += k ** 2

    if summa == cube_count:
        return k
    else:
        return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = str(data.value)
    middle = len(number) // 2

    left = number[:middle]
    right = number[(len(number) + 1) // 2:]

    left_sum = 0
    right_sum = 0

    for digit in left:
        left_sum += int(digit)

    for digit in right:
        right_sum += int(digit)

    return left_sum == right_sum
