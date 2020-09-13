from typing import Iterable


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

    @staticmethod
    def _validate_matrix(matrix):
        if not isinstance(matrix, Iterable):
            raise TypeError('Matrix should be iterable by rows')

    def copy(self):
        return Matrix([row.copy() for row in self])


class MatrixRow(list):
    """
    Row of 2D matrix.
    Can be added/subbed/multied/divided to other matrix row.
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
                    raise TypeError('Row elements must be numbers')

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
        return MatrixRow([item*value for item in self])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, value):
        if type(value) not in (int, float):
            raise TypeError('Value must be number')
        return MatrixRow([item/value for item in self])


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

    def get_determinant(self):
        if len(self) == 1:
            return self[0][0]
        if len(self) == 2:
            return self[0][0] * self[1][1] - self[1][0] * self[0][1]
        else:
            result = 0
            for current_row_number in range(len(self)):
                small_matrix = self.copy()

                del small_matrix[current_row_number]
                for row in small_matrix:
                    del row[0]
                result += (-1)**(current_row_number+2) * self[current_row_number][0] * SquareMatrix(small_matrix).get_determinant()
            return result


class RowsLengthError(ArithmeticError):
    pass


m = SquareMatrix([
    [2, 2, 2],
    [21, 11, 34],
    [1, 2, 4]
])
