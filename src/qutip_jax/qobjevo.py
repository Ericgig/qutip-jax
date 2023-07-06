import jaxlib
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from .jaxarray import JaxArray
from qutip import Qobj, Coefficient
from qutip.core.cy._element import *
from qutip.core.coefficient import coefficient_builders
import qutip.core.data as _data

__all__ = []


class JaxJitCoeff(Coefficient):
    func: callable
    static_argnames: tuple
    args: dict

    def __init__(self, func, args={}, static_argnames=(), **_):
        self.func = func
        self.static_argnames = static_argnames
        Coefficient.__init__(self, args)
        self.jit_call = jit(self._caller, static_argnames=self.static_argnames)

    def __call__(self, t, _args=None, **kwargs):
        if _args:
            kwargs.update(_args)
        return self.jit_call(t, **kwargs)

    def _caller(self, t, **kwargs):
        args = self.args.copy()
        for key in kwargs:
            if key in args:
                args[key] = kwargs[key]
        return self.func(t, **args)

    def replace_arguments(self, _args=None, **kwargs):
        if _args:
            kwargs.update(_args)
        args = self.args.copy()
        for key in kwargs:
            if key in args:
                args[key] = kwargs[key]
        return JaxJitCoeff(
            self.func, args=args, static_argnames=self.static_argnames
        )

    def __add__(self, other):
        if isinstance(other, JaxJitCoeff):
            merge_static = tuple(
                set(self.static_argnames) | set(other.static_argnames)
            )

            def f(t, **kwargs):
                return self._caller(t, **kwargs) + other._caller(t, **kwargs)

            return JaxJitCoeff(
                jit(f, static_argnames=merge_static),
                args={**self.args, **other.args},
                static_argnames=merge_static,
            )

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, JaxJitCoeff):
            merge_static = tuple(
                set(self.static_argnames) | set(other.static_argnames)
            )

            def f(t, **kwargs):
                return self._caller(t, **kwargs) * self._caller(t, **kwargs)

            return JaxJitCoeff(
                jit(f, static_argnames=merge_static),
                args={**self.args, **other.args},
                static_argnames=merge_static,
            )

        return NotImplemented

    def conj(self):
        def f(t, **kwargs):
            return jnp.conj(self._caller(t, **kwargs))

        return JaxJitCoeff(
            jit(f, static_argnames=self.static_argnames),
            args=self.args,
            static_argnames=self.static_argnames,
        )

    def _cdc(self):
        def f(t, **kwargs):
            val = self(t, **kwargs)
            return jnp.conj(val) * val

        return JaxJitCoeff(
            jit(f, static_argnames=self.static_argnames),
            args=self.args,
            static_argnames=self.static_argnames,
        )

    def copy(self):
        return self

    def __reduce__(self):
        # Jitted function cannot be pickled.
        # Extract the original function and re-jit it.
        # This can fail depending on the wrapped object.
        return (
            self.restore,
            (self.func.__wrapped__, self.args, self.static_argnames)
        )

    @classmethod
    def restore(cls, func, args, static_argnames):
        return cls(
            jit(func, static_argnames=static_argnames),
            args,
            static_argnames
        )

    def flatten(self):
        static_args = {
            key: val for key, val in self.args.items()
            if key in self.static_argnames
        }
        jax_args = {
            key: val for key, val in self.args.items()
            if key not in self.static_argnames
        }
        return (jax_args,), (self.func, static_args, self.static_argnames)

    @classmethod
    def unflatten(cls, aux_data, children):
        func, static_args, static_argnames = aux_data

        return JaxJitCoeff(
            func,
            args={**children[0], **static_args},
            static_argnames=static_argnames
        )


coefficient_builders[jaxlib.xla_extension.PjitFunction] = JaxJitCoeff
jax.tree_util.register_pytree_node(
    JaxJitCoeff, JaxJitCoeff.flatten, JaxJitCoeff.unflatten
)


class _JaxMapElement:
    def __init__(self, base_element):
        if isinstance(base_element, _FuncElement):
            self._read_func(base_element)
            self._transform = []
            self._coeff = 1.
        else:
            self._read_func(base_element._base)
            self._transform = base_element._transform
            self._coeff = base_element._coeff

    def _read_func(self, f_element):
        if not f_element._f_pythonic:
            self._f_parameters = set(f_element._args.keys())

            def call(t, **kwargs):
                return f_element._func(t, kwargs)

        else:
            call = f_element._func
            self._f_parameters = f_element._f_parameters

        self._func = call
        self._args = f_element._args

    def matmul_data_t(self, t, y, out=None):
        data = self.data(t)
        return _data.matmul(data, y, self._coeff, out=out)

    def qobj(self, t):
        qobj = self._func(t, **self._args)
        for func in self._transform:
            qobj = func(qobj)
        return qobj

    def data(self, t):
        return self.qobj(t).data

    def coeff(self, t):
        return self._coeff

    def replace_arguments(self, args, cache=None):
        self._args = {k: args[k] for k in self._f_parameters & args.keys()}
        return self

    def flatten(self):
        aux_data = (self._func, self._f_parameters, self._transform)
        children = (self._args, self._coeff)
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        out = cls.__new__(cls)
        out._args = children[0]
        out._coeff = children[1]
        out._func = aux_data[0]
        out._f_parameters = aux_data[1]
        out._transform = aux_data[2]
        return out


jax.tree_util.register_pytree_node(
    _JaxMapElement, _JaxMapElement.flatten, _JaxMapElement.unflatten
)


class _JaxProdElement:
    def __init__(self, base_element):
        if isinstance(base_element._left, (_MapElement, _FuncElement)):
            self._left = _JaxMapElement(base_element._left)
        elif isinstance(base_element._left, _ProdElement):
            self._left = _JaxProdElement(base_element._left)
        else:
            self._left = base_element._left

        if isinstance(base_element._right, (_MapElement, _FuncElement)):
            self._right = _JaxMapElement(base_element._right)
        elif isinstance(base_element._right, _ProdElement):
            self._right = _JaxProdElement(base_element._right)
        else:
            self._right = base_element._right

        self._transform = base_element._transform
        self._conj = base_element._conj

    def matmul_data_t(self, t, y, out=None):
        if not self._transform:
            temp = _data.matmul(self._right.data(t), y, self._right.coeff(t))
            temp = _data.matmul(self._left.data(t), temp, self._left.coeff(t))
            return _data.add(out, temp)
        elif out is None:
            return _data.matmul(self.data(t), y, self.coeff(t))
        else:
            return _data.add(
                out,
                _data.matmul(self.data(t), y, self.coeff(t))
            )

    def qobj(self, t):
        qobj = self._left.qobj(t) @ self._right.qobj(t)
        for func in self._transform:
            qobj = func(qobj)
        return qobj

    def data(self, t):
        return self.qobj(t).data

    def coeff(self, t):
        out = self._left.coeff(t) * self._right.coeff(t)
        if self._conj:
            out = jnp.conj(out)
        return out

    def replace_arguments(self, args):
        self._left = self._left.replace_arguments(args)
        self._right = self._right.replace_arguments(args)
        return self

    def flatten(self):
        aux_data = (self._conj, self._transform)
        children = (self._left, self._right)
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        out = cls.__new__(cls)
        out._left = children[0]
        out._right = children[1]
        out._conj = aux_data[0]
        out._transform = aux_data[1]
        return out


jax.tree_util.register_pytree_node(
    _JaxProdElement, _JaxProdElement.flatten, _JaxProdElement.unflatten
)


class JaxQobjEvo:
    """
    Pytree friendly QobjEvo for the Diffrax integrator.

    It only support list based `QobjEvo`.
    """

    batched_data: jnp.ndarray
    coeffs: list
    func_parts: list
    dims: object

    def __init__(self, qobjevo):
        as_list = qobjevo.to_list()
        self.coeffs = []
        qobjs = []
        self.func_parts = []
        self.dims = qobjevo.dims
        self.batched_data = None

        constant = JaxJitCoeff(jit(lambda t, **_: 1.0))

        for part in as_list:
            if isinstance(part, Qobj):
                qobjs.append(part)
                self.coeffs.append(constant)
            elif (
                isinstance(part, list) and isinstance(part[0], Qobj)
            ):
                qobjs.append(part[0])
                self.coeffs.append(part[1])
            else:
                if isinstance(part[0], _MapElement):
                    self.func_parts.append(_JaxMapElement(part[0]))
                elif isinstance(part[0], _ProdElement):
                    self.func_parts.append(_JaxProdElement(part[0]))
                else:
                    self.func_parts.append(
                        _JaxMapElement(_FuncElement(part[0], args=part[1]))
                    )

        if qobjs:
            shape = qobjs[0].shape
            self.batched_data = jnp.zeros(
                shape + (len(qobjs),), dtype=np.complex128
            )
            for i, qobj in enumerate(qobjs):
                self.batched_data = self.batched_data.at[:, :, i].set(
                    qobj.to("jax").data._jxa
                )

    @jit
    def _coeff(self, t):
        list_coeffs = [coeff(t) for coeff in self.coeffs]
        return jnp.array(list_coeffs, dtype=np.complex128)

    def __call__(self, t, _args=None, **kwargs):
        if _args is not None:
            kwargs.update(_args)
        if kwargs:
            caller = self.arguments(kwargs)
        else:
            caller = self
        return Qobj(caller.data(t), dims=self.dims)

    @jit
    def data(self, t):
        coeff = self._coeff(t)
        data = jnp.dot(self.batched_data, coeff)
        out = JaxArray._fast_constructor(data, data.shape)
        for part in self.func_parts:
            out = _data.add(out, part.data(t), part.coeff(t), dtype=type(out))
        return out

    @jit
    def matmul_data(self, t, y):
        coeffs = self._coeff(t)
        out = JaxArray(jnp.dot(jnp.dot(self.batched_data, coeffs), y._jxa))
        for part in self.func_parts:
            out = part.matmul_data_t(t, y, out)
        return out

    def arguments(self, args):
        out = JaxQobjEvo.__new__(JaxQobjEvo)
        out.batched_data = self.batched_data
        out.coeffs = [coeff.replace_arguments(args) for coeff in self.coeffs]
        out.func_parts = [
            part.replace_arguments(args) for part in self.func_parts
        ]
        out.dims = self.dims
        return out

    def flatten(self):
        return (
            (self.batched_data, self.coeffs, self.func_parts),
            {"dims": self.dims},
        )

    @classmethod
    def unflatten(cls, aux_data, children):
        out = cls.__new__(cls)
        out.batched_data = children[0]
        out.coeffs = children[1]
        out.func_parts = children[2]
        out.dims = aux_data["dims"]
        return out


jax.tree_util.register_pytree_node(
    JaxQobjEvo, JaxQobjEvo.flatten, JaxQobjEvo.unflatten
)
