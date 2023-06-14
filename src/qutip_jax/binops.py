import qutip
from .jaxarray import JaxArray
from .jaxdia import JaxDia, clean_diag
import jax.numpy as jnp
import jax

__all__ = [
    "add_jaxarray",
    "add_jaxdia",
    "sub_jaxarray",
    "sub_jaxdia",
    "mul_jaxarray",
    "mul_jaxdia",
    "matmul_jaxarray",
    "multiply_jaxarray",
    "multiply_jaxdia",
    "kron_jaxarray",
    "kron_jaxdia",
    "pow_jaxarray",
]


def _check_same_shape(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"""Incompatible shapes for addition of two matrices:
                         left={left.shape} and right={right.shape}"""
        )


def _check_matmul_shape(left, right, out):
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "incompatible matrix shapes " + str(left.shape)
            + " and " + str(right.shape)
        )
    if (
        out is not None
        and out.shape[0] != left.shape[0]
        and out.shape[1] != right.shape[1]
    ):
        raise ValueError(
            "incompatible output shape, got "
            + str(out.shape)
            + " but needed "
            + str((left.shape[0], right.shape[1]))
        )


def add_jaxarray(left, right, scale=1):
    """
    Perform the operation
        left + scale*right
    where `left` and `right` are matrices, and `scale` is an optional complex
    scalar.
    """
    _check_same_shape(left, right)

    if scale == 1 and isinstance(scale, int):
        out = JaxArray._fast_constructor(
            left._jxa + right._jxa, shape=left.shape
        )
    else:
        out = JaxArray._fast_constructor(
            left._jxa + scale * right._jxa, shape=left.shape
        )
    return out


def add_jaxdia(left, right, scale=1):
    """
    Perform the operation
        left + scale*right
    where `left` and `right` are matrices, and `scale` is an optional complex
    scalar.
    """
    _check_same_shape(left, right)
    diag_left = 0
    diag_right = 0
    data = []
    offsets = []

    all_diag = set(left.offsets) | set(right.offsets)

    for diag in all_diag:
        if diag in left.offsets and diag in right.offsets:
            diag_left = left.offsets.index(diag)
            diag_right = right.offsets.index(diag)
            offsets.append(diag)
            data.append(
                left.data[diag_left, :] + right.data[diag_right, :] * scale
            )

        elif diag in left.offsets:
            diag_left = left.offsets.index(diag)
            offsets.append(diag)
            data.append(left.data[diag_left, :])

        elif diag in right.offsets:
            diag_right = right.offsets.index(diag)
            offsets.append(diag)
            data.append(right.data[diag_right, :] * scale)

    """
    while diag_left < left.num_diags and diag_right < right.num_diags:
        if left.offsets[diag_left] == right.offsets[diag_right]:
            offsets.append(left.offsets[diag_left])
            data.append(left.data[diag_left, :] + right.data[diag_right, :] * scale)
            diag_left += 1
            diag_right += 1
        elif left.offsets[diag_left] <= right.offsets[diag_right]:
            offsets.append(left.offsets[diag_left])
            data.append(left.data[diag_left, :])
            diag_left += 1
        else:
            offsets.append(right.offsets[diag_right])
            data.append(right.data[diag_right, :] * scale)
            diag_right += 1

    for i in range(diag_left, left.num_diags):
        offsets.append(left.offsets[i])
        data.append(left.data[i, :])

    for i in range(diag_right, right.num_diags):
        offsets.append(right.offsets[i])
        data.append(right.data[i, :] * scale)
    """

    # if not sorted:
    #     dia.clean_diag(out, True)
    # if settings.core['auto_tidyup']:
    #     tidyup_dia(out, settings.core['auto_tidyup_atol'], True)
    return JaxDia((tuple(offsets), jnp.array(data)), left.shape, False)


def sub_jaxarray(left, right):
    """
    Perform the operation
        left - right
    where `left` and `right` are matrices.
    """
    return add_jaxarray(left, right, -1)


def sub_jaxdia(left, right):
    """
    Perform the operation
        left - right
    where `left` and `right` are matrices.
    """
    return add_jaxdia(left, right, -1)


def mul_jaxarray(matrix, value):
    """Multiply a matrix element-wise by a scalar."""
    return JaxArray._fast_constructor(matrix._jxa * value, matrix.shape)


def mul_jaxdia(matrix, value):
    """Multiply a matrix element-wise by a scalar."""
    return JaxDia._fast_constructor(matrix.offsets, matrix.data * value, matrix.shape)


def matmul_jaxarray(left, right, scale=1, out=None):
    """
    Compute the matrix multiplication of two matrices, with the operation
        scale * (left @ right)
    where `scale` is (optionally) a scalar, and `left` and `right` are
    matrices.

    Arguments
    ---------
    left : Data
        The left operand as either a bra or a ket matrix.

    right : Data
        The right operand as a ket matrix.

    scale : complex, optional
        The scalar to multiply the output by.
    """
    _check_matmul_shape(left, right, out)
    shape = (left.shape[0], right.shape[1])

    result = left._jxa @ right._jxa

    if scale != 1 or not isinstance(scale, int):
        result *= scale

    if out is None:
        return JaxArray._fast_constructor(result, shape=shape)
    else:
        out._jxa = result + out._jxa


def multiply_jaxarray(left, right):
    """Element-wise multiplication of matrices."""
    _check_same_shape(left, right)
    return JaxArray._fast_constructor(left._jxa * right._jxa, shape=left.shape)


def multiply_jaxdia(left, right):
    """
    Perform the operation
        left + scale*right
    where `left` and `right` are matrices, and `scale` is an optional complex
    scalar.
    """
    _check_same_shape(left, right)
    diag_left = 0
    diag_right = 0
    data = []
    offsets = []
    # print(left.offsets)
    # print(right.offsets)

    for i, diag in enumerate(left.offsets):
        if diag not in right.offsets:
            continue
        j = right.offsets.index(diag)
        offsets.append(diag)
        data.append(left.data[i, :] * right.data[j, :])

    """while diag_left < left.num_diags and diag_right < right.num_diags:
        if left.offsets[diag_left] == right.offsets[diag_right]:
            offsets.append(left.offsets[diag_left])
            data.append(left.data[diag_left, :] * right.data[diag_right, :])
            diag_left += 1
            diag_right += 1
        elif left.offsets[diag_left] <= right.offsets[diag_right]:
            diag_left += 1
        else:
            diag_right += 1"""

    # if not sorted:
    #     dia.clean_diag(out, True)
    # if settings.core['auto_tidyup']:
    #     tidyup_dia(out, settings.core['auto_tidyup_atol'], True)
    return JaxDia((jnp.array(offsets), jnp.array(data)), left.shape, False)


def kron_jaxarray(left, right):
    """
    Compute the Kronecker product of two matrices.  This is used to represent
    quantum tensor products of vector spaces.
    """
    return JaxArray(jnp.kron(left._jxa, right._jxa))


def multiply_outer(left, right):
    return jax.vmap(jax.vmap(jnp.multiply, (None, 0)), (0, None))(left, right).ravel()


def kron_jaxdia(left, right):
    nrows = left.shape[0] * right.shape[0]
    ncols = left.shape[1] * right.shape[1]
    left = clean_diag(left)
    right = clean_diag(right)
    out = {}

    if right.shape[0] == right.shape[1]:
        for diag_left in range(left.num_diags):
            for diag_right in range(right.num_diags):
                print(diag_left, diag_right)
                out_diag = (
                    left.offsets[diag_left] * right.shape[0]
                    + right.offsets[diag_right]
                )
                out_data = multiply_outer(
                    left.data[diag_left],
                    right.data[diag_right]
                )
                if out_diag in out:
                    out[out_diag] = out[out_diag] + out_data
                else:
                    out[out_diag] = out_data

    else:
        delta = right.shape[0] - right.shape[1]
        for diag_left in range(left.num_diags):
            start_left = max(0, left.offsets[diag_left])
            end_left = min(left.shape[1], left.shape[0] + left.offsets[diag_left])
            for diag_right in range(right.num_diags):
                start_right = max(0, right.offsets[diag_right])
                end_right = min(right.shape[1], right.shape[0] + right.offsets[diag_right])

                for col_left in range(start_left, end_left):
                    out_diag =  (
                        left.offsets[diag_left] * right.shape[0]
                        + right.offsets[diag_right]
                        - col_left * delta
                    )
                    data = jnp.zeros(ncols, dtype=jnp.complex128)
                    data = data.at[col_left*right.shape[1]:col_left*right.shape[1] + right.shape[1]].set(
                        left.data[diag_left, col_left] * right.data[diag_right]
                    )

                    if out_diag in out:
                        out[out_diag] = out[out_diag] + data
                    else:
                        out[out_diag] = data

    out = JaxDia(
        (tuple(out.keys()), jnp.array(list(out.values()))),
        shape=(nrows, ncols)
    )
    out = clean_diag(out)
    return out


def pow_jaxarray(matrix, n):
    """
    Compute the integer matrix power of the square input matrix.  The power
    must be an integer >= 0.  `A ** 0` is defined to be the identity matrix of
    the same shape.

    Arguments
    ---------
    matrix : Data
        Input matrix to take the power of.

    n : non-negative integer
        The power to which to raise the matrix.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix power only works with square matrices")
    return JaxArray(jnp.linalg.matrix_power(matrix._jxa, n))


qutip.data.add.add_specialisations([
    (JaxArray, JaxArray, JaxArray, add_jaxarray),
    (JaxDia, JaxDia, JaxDia, add_jaxdia),
])

qutip.data.sub.add_specialisations([
    (JaxArray, JaxArray, JaxArray, sub_jaxarray),
    (JaxDia, JaxDia, JaxDia, sub_jaxdia),
])

qutip.data.mul.add_specialisations([
    (JaxArray, JaxArray, mul_jaxarray),
    (JaxDia, JaxDia, mul_jaxdia),
])

qutip.data.matmul.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, matmul_jaxarray),]
)

qutip.data.multiply.add_specialisations([
    (JaxArray, JaxArray, JaxArray, multiply_jaxarray),
    (JaxDia, JaxDia, JaxDia, multiply_jaxdia),
])

qutip.data.kron.add_specialisations([
    (JaxArray, JaxArray, JaxArray, kron_jaxarray),
    (JaxDia, JaxDia, JaxDia, kron_jaxdia),
])

qutip.data.pow.add_specialisations(
    [(JaxArray, JaxArray, pow_jaxarray),]
)
