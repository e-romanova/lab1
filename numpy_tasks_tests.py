import numpy as np

from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)

from grader_contracts.numpy_tasks import (
    MatrixVectorBatchInput,
    BinarizeInput,
    MatrixInput,
    RandomMatrixInput,
    ChessInput,
    RectangleInput,
    EllipseInput,
    TimeSeriesInput,
    OneHotInput,
)


def test_sum_prod():
    data = MatrixVectorBatchInput(
        matrices=[[[1, 2], [3, 4]], [[2, 0], [0, 2]]],
        vectors=[[[1], [2]], [[3], [4]]],
    )
    np.testing.assert_array_equal(
    sum_prod(data),
    [[11], [19]],
)


def test_binarize():
    data = BinarizeInput(matrix=[[0.2, 0.5], [0.7, 0.9]], threshold=0.5)
    np.testing.assert_array_equal(
        binarize(data),
        [[0, 0], [1, 1]],
    )


def test_unique_rows():
    result = unique_rows(MatrixInput(matrix=[[2, 1, 2], [3, 3, 1]]))
    assert result == [[1, 2], [1, 3]]


def test_unique_columns():
    result = unique_columns(MatrixInput(matrix=[[2, 1], [2, 3]]))
    assert result == [[2], [1, 3]]


def test_matrix_statistics():
    data = RandomMatrixInput(rows=3, columns=4, seed=42)
    result = matrix_statistics(data)

    assert result.matrix.shape == (3, 4)
    assert result.row_means.shape == (3,)
    assert result.column_means.shape == (4,)
    assert result.row_variances.shape == (3,)
    assert result.column_variances.shape == (4,)


def test_chess():
    result = chess(ChessInput(rows=2, columns=3, first=1, second=0))
    np.testing.assert_array_equal(result, [[1, 0, 1], [0, 1, 0]])


def test_draw_rectangle():
    data = RectangleInput(
        width=2,
        height=2,
        image_height=4,
        image_width=4,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0),
    )
    result = draw_rectangle(data)

    assert result.shape == (4, 4, 3)
    np.testing.assert_array_equal(
    result[1:3, 1:3],
    np.full((2, 2, 3), (255, 0, 0), dtype=np.uint8),
)


def test_draw_ellipse():
    data = EllipseInput(
        semi_axis_x=2,
        semi_axis_y=1,
        image_height=3,
        image_width=5,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0),
    )
    result = draw_ellipse(data)

    assert result.shape == (3, 5, 3)
    np.testing.assert_array_equal(result[1, 2], (255, 0, 0))
    np.testing.assert_array_equal(result[0, 0], (0, 0, 0))


def test_analyze_time_series_local_extrema():
    result = analyze_time_series(
        TimeSeriesInput(values=[1, 3, 1, 0, 2, 0], window=2)
    )

    np.testing.assert_array_equal(result.local_maxima_indices, [1, 4])
    np.testing.assert_array_equal(result.local_minima_indices, [3])


def test_one_hot():
    result = one_hot(OneHotInput(labels=[0, 2, 1]))
    np.testing.assert_array_equal(
        result,
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
    )