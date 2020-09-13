# coding=utf-8
from typing import Iterable
from random import random
from math import sqrt
import time

times = {}


def memoize(function):
    memo = {}

    def wrapper(*args):
        if len(args[0]) > 1:
            key = tuple(tuple(x) for x in args[0])
        else:
            key = args[0][0]
        if key in memo:
            return memo[key]
        else:
            rv = function(*args)
            memo[key] = rv
            return rv

    return wrapper


def benchmark(func):
    global times
    """
    Декоратор, выводящий время, которое заняло
    выполнение декорируемой функции.
    """

    def wrapper(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        if func.__name__ in times:
            times[func.__name__] += time.time() - t
        else:
            times[func.__name__] = time.time() - t
        print(func.__name__, time.time() - t)
        return res

    return wrapper


class Matrix(list):
    """
    2D matrix
    List of tuples where tuples are rows of matrix.
    """

    def __init__(self, data=None):
        if data is None:
            data = []
        elif type(data) in (int, float):
            super(Matrix, self).__init__(list(data))
        else:
            self._validate_matrix(data)
            super(Matrix, self).__init__(list([MatrixRow(row) for row in data]))

    # def __str__(self):
    # TODO: Normal view for Matrix.
    # pass

    @staticmethod
    def _validate_matrix(matrix):
        if not isinstance(matrix, Iterable):
            raise TypeError('Matrix should be iterable by rows')

    def copy(self):
        return Matrix([row.copy() for row in self])

    def create_step_version(self, vector=None):
        if len(vector) != len(self):
            raise VectorLengthError('Vector must be the same size with matrix: ' + str(vector))

        new_matrix = self.copy()
        for row_num in range(1, len(new_matrix)):
            for row_number in range(row_num, len(new_matrix)):
                coefficient = - new_matrix[row_number][row_num - 1] / new_matrix[row_num - 1][row_num - 1]
                new_matrix[row_number] = MatrixRow(new_matrix[row_number] + coefficient * new_matrix[row_num - 1])
                vector[row_number] = vector[row_number] + coefficient * vector[row_num - 1]

        return Matrix(new_matrix), vector


class MatrixRow(list):
    """
    Row of 2D matrix.
    Can be added/subbed to other matrix row or multied/divided on other value.
    """

    def __init__(self, data=None):
        if data is None:
            data = []
        else:
            self._validate_row(data)
        super(MatrixRow, self).__init__(data)

    @staticmethod
    def _validate_row(row):
        if type(row) in (float, int):
            return
        if not isinstance(row, Iterable):
            raise TypeError('Row should be iterable by elements')
        else:
            for item in row:
                if type(item) not in (int, float):
                    raise TypeError('Row elements must be numbers: ' + str(item))

    @staticmethod
    def _validate_row_for_addition(row1, row2):
        if len(row1) != len(row2):
            raise RowsLengthError("Rows must have equal lengths")

    def __add__(self, row):
        row = MatrixRow(row)
        self._validate_row_for_addition(self, row)
        new_row = [self[i] + row[i] for i in range(len(row))]
        return new_row

    def __sub__(self, row):
        row = MatrixRow(row)
        self._validate_row_for_addition(self, row)
        new_row = [self[i] + row[i] for i in range(len(row))]
        return new_row

    def __mul__(self, value):
        if type(value) not in (int, float):
            raise TypeError('Value must be number')
        return MatrixRow([item * value for item in self])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, value):
        if type(value) not in (int, float):
            raise TypeError('Value must be number')
        return MatrixRow([item / value for item in self])


class SquareMatrix(Matrix):
    def __init__(self, data=None):
        if type(data) in (int, float):
            super(Matrix, self).__init__([data])
        else:
            super(SquareMatrix, self).__init__(data.copy())

    @staticmethod
    def _validate_matrix(matrix):
        if not isinstance(matrix, Iterable):
            raise TypeError('Matrix must be iterable by rows')
        for row in matrix:
            if len(MatrixRow(row)) != len(matrix):
                raise TypeError('Matrix must be squared')

    def copy(self):
        return SquareMatrix([row.copy() for row in self])

    @memoize
    @benchmark
    def get_determinant(self):
        # import numpy as np
        # a = np.array(self)
        # return np.linalg.det(a)

        if len(self) == 1:
            return self[0][0]
        if len(self) == 2:
            return self[0][0] * self[1][1] - self[1][0] * self[0][1]
        else:
            result = 0
            for current_row_number in range(len(self)):
                matrix_with_null = self.make_first_column_null(self)
                # small_matrix = self.get_minor(current_row_number)
                # result += (-1) ** (current_row_number + 2) * self[current_row_number][0] * SquareMatrix(
                #     small_matrix).get_determinant()

                small_matrix = self.get_minor(current_row_number)
                result += (-1) ** (current_row_number + 2) * self[current_row_number][0] * SquareMatrix(
                    small_matrix).get_determinant()


            return result

    @benchmark
    def get_minor(self, i):
        minor = self.copy()
        del minor[i]
        for k in range(len(minor[0]) - 1):
            del minor[k][i]
        return minor

    @staticmethod
    def make_first_column_null(matrix):
        new_matrix = matrix.copy()
        for row_num in range(1, len(new_matrix)):
            coefficient = - new_matrix[row_num][0] / new_matrix[0][0]
            new_matrix[row_num] = MatrixRow(new_matrix[row_num] + coefficient * new_matrix[0)
        return Matrix(new_matrix)

    @benchmark
    def kramer(self, vector) -> list:
        """
        :param vector: vector of free arguments.
        :return: vector of solutions.
        """
        result_vector = list()

        main_matrix_determinant = self.get_determinant()
        if main_matrix_determinant == 0:
            return [None for _ in vector]

        for index in range(len(self)):
            replaced_matrix = self.replace_matrix_column_on_vector(self, index, vector)
            replaced_matrix_determinant = replaced_matrix.get_determinant()
            result_vector.append(replaced_matrix_determinant / main_matrix_determinant)
        return result_vector

    @staticmethod
    @benchmark
    def replace_matrix_column_on_vector(matrix: Matrix, column_index: int, vector: list) -> Matrix:
        if len(matrix) != len(vector):
            raise VectorLengthError('Vector must have the same length as matrix: ' + str(vector))

        new_matrix = matrix.copy()
        for row_number in range(len(new_matrix)):
            new_matrix[row_number][column_index] = vector[row_number]
        return SquareMatrix(new_matrix)


@benchmark
def generate_values(n: int, left_border: float, right_border: float, eps_len=5) -> list:
    """
    :param n: size of future Matrix
    :param left_border: Left border of random values
    :param right_border: right border of random values
    :return: generate n*(n+1) values
    """
    array_size = n * (n + 1)
    if right_border < left_border:
        left_border, right_border = right_border, left_border

    range_between_borders = right_border - left_border
    result_array = list()
    t = time.time()
    for _ in range(array_size):
        result_array.append(round(random() + right_border - range_between_borders, eps_len))
    return result_array


@benchmark
def create_file_with_values(values: Iterable, file_name: str, separator=';'):
    values_in_string = [str(value) for value in values]
    result_string = separator.join(values_in_string)
    with open(file_name, 'w') as file:
        file.write(result_string)


class GeneratedMatrixWithVector(SquareMatrix):
    """
    Generated Matrix from file.
    file contains n*(n+1) random values where n - size of squared matrix.
    the last n values - vector of values for the equation.
    Vector will stored in special variable of GeneratedMatrix object (vector).
    """

    def __init__(self, file_name: str, sep=';'):
        matrix_data, vector_data = self.get_matrix_data_and_vector_from_file(file_name, sep)
        self.vector = vector_data
        super().__init__(matrix_data)

    @benchmark
    def kramer_solve(self):
        return self.kramer(self.vector)

    @staticmethod
    @benchmark
    def get_matrix_data_and_vector_from_file(file_name, separator) -> tuple:
        with open(file_name, 'r') as file:
            text_values = file.read()
        values = text_values.split(separator)

        n = int(sqrt(len(values)))

        if len(values) != n * (n + 1):
            raise ValueError('Input file is incorrect! it must contain n*(n+1) values.')

        data_for_matrix = values[:n * n]
        data_for_vector = values[n * n:]

        data_for_matrix = list(map(float, data_for_matrix))
        data_for_vector = list(map(float, data_for_vector))

        data_for_matrix = [data_for_matrix[i:i + n] for i in range(0, len(data_for_matrix), n)]
        return data_for_matrix, data_for_vector


class RowsLengthError(ArithmeticError):
    pass


class VectorLengthError(ArithmeticError):
    pass


m = SquareMatrix([
    [1, 2, 1],
    [3, -1, -1],
    [-2, 2, 3],
])

n = int(input('Type size of matrix (n): '))
l = float(input('Type left border for random values: '))
r = float(input('Type right border for random values: '))
eps_len = int(input('Type the number of symbols after comma (default 5): '))

values = generate_values(n, l, r, eps_len)
create_file_with_values(values, 'random_values.txt', ' ')

matrix = GeneratedMatrixWithVector('random_values.txt', ' ')
ti = time.time()
print(matrix.kramer_solve())
print(times)

print('total time: ', time.time()-ti)
# times = []
#
# for n in range(2,9):
#     l = 1
#     r = 10
#     eps_len = 1
#
#     values = generate_values(n, l, r, eps_len)
#     create_file_with_values(values, 'random_values.txt', ' ')
#     matrix = GeneratedMatrixWithVector('random_values.txt', ' ')
#     start_time = time.time()
#     print(matrix.kramer_solve())
#     times.append(time.time()-start_time)
