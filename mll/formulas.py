from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Atom:
    name: str
    neg: bool = False


@dataclass(frozen=True)
class One:
    pass


@dataclass(frozen=True)
class Bottom:
    pass


@dataclass(frozen=True)
class Tensor:
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Par:
    left: Formula
    right: Formula


Formula = Atom | One | Bottom | Tensor | Par


def atom_name(index: int) -> str:
    """Letter-only names: a, b, ..., z, aa, ab, ... (no digits)."""
    name = ""
    n = index
    while True:
        name = chr(ord("a") + n % 26) + name
        n = n // 26 - 1
        if n < 0:
            return name


def dual(f: Formula) -> Formula:
    if isinstance(f, Atom):
        return Atom(f.name, not f.neg)
    if isinstance(f, One):
        return Bottom()
    if isinstance(f, Bottom):
        return One()
    if isinstance(f, Tensor):
        return Par(dual(f.left), dual(f.right))
    if isinstance(f, Par):
        return Tensor(dual(f.left), dual(f.right))
    raise TypeError(type(f))


def fstr(f: Formula) -> str:
    if isinstance(f, Atom):
        return f"{f.name}⊥" if f.neg else f.name
    if isinstance(f, One):
        return "𝟙"
    if isinstance(f, Bottom):
        return "⊥"
    if isinstance(f, Tensor):
        return f"({fstr(f.left)} ⊗ {fstr(f.right)})"
    if isinstance(f, Par):
        return f"({fstr(f.left)} ⅋ {fstr(f.right)})"
    raise TypeError(type(f))
