from __future__ import annotations

from dataclasses import dataclass

from .formulas import Atom, Bottom, Formula, One, Par, Tensor
from .proofs import Sequent


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class RawAx:
    pass


@dataclass(frozen=True)
class RawOne:
    pass


@dataclass(frozen=True)
class RawBot:
    index: int
    child: RawProof


@dataclass(frozen=True)
class RawPar:
    index: int
    child: RawProof


@dataclass(frozen=True)
class RawTensor:
    index: int
    left_positions: tuple[int, ...]
    left: RawProof
    right: RawProof


RawProof = RawAx | RawOne | RawBot | RawPar | RawTensor


class Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.i = 0

    def remaining(self) -> str:
        return self.text[self.i :]

    def eof(self) -> bool:
        return self.i >= len(self.text)

    def eat(self, expected: str) -> None:
        if not self.text.startswith(expected, self.i):
            raise ParseError(
                f"expected {expected!r} at {self.i}, got {self.remaining()[:20]!r}"
            )
        self.i += len(expected)

    def try_eat(self, expected: str) -> bool:
        if self.text.startswith(expected, self.i):
            self.i += len(expected)
            return True
        return False

    def parse_int(self) -> int:
        start = self.i
        if self.eof() or not self.text[self.i].isdigit():
            raise ParseError(f"expected integer at {self.i}")
        while not self.eof() and self.text[self.i].isdigit():
            self.i += 1
        return int(self.text[start : self.i])

    def parse_atom_name(self) -> str:
        start = self.i
        if self.eof() or not self.text[self.i].islower():
            raise ParseError(f"expected atom name at {self.i}")
        while not self.eof() and self.text[self.i].islower():
            self.i += 1
        return self.text[start : self.i]

    def parse_formula(self) -> Formula:
        if self.try_eat("𝟙"):
            return One()
        if self.try_eat("⊥"):
            return Bottom()
        if self.try_eat("¬"):
            return Atom(self.parse_atom_name(), neg=True)
        if not self.eof() and self.text[self.i].islower():
            return Atom(self.parse_atom_name(), neg=False)
        if self.try_eat("("):
            left = self.parse_formula()
            if self.try_eat(" ⊗ "):
                right = self.parse_formula()
                self.eat(")")
                return Tensor(left, right)
            if self.try_eat(" ⅋ "):
                right = self.parse_formula()
                self.eat(")")
                return Par(left, right)
            raise ParseError(f"expected ⊗ or ⅋ at {self.i}")
        raise ParseError(f"unexpected formula at {self.i}: {self.remaining()[:20]!r}")

    def parse_sequent(self) -> Sequent:
        self.eat("⊢ ")
        formulas: list[Formula] = [self.parse_formula()]
        while self.try_eat(", "):
            formulas.append(self.parse_formula())
        return tuple(formulas)

    def parse_positions(self) -> tuple[int, ...]:
        self.eat("[")
        if self.try_eat("]"):
            return ()
        positions = [self.parse_int()]
        while self.try_eat(","):
            positions.append(self.parse_int())
        self.eat("]")
        return tuple(positions)

    def parse_proof_term(self) -> RawProof:
        if self.try_eat("AX"):
            return RawAx()
        if self.try_eat("𝟙"):
            return RawOne()
        if self.try_eat("⊥("):
            index = self.parse_int()
            self.eat(",")
            child = self.parse_proof_term()
            self.eat(")")
            return RawBot(index, child)
        if self.try_eat("⅋("):
            index = self.parse_int()
            self.eat(",")
            child = self.parse_proof_term()
            self.eat(")")
            return RawPar(index, child)
        if self.try_eat("⊗("):
            index = self.parse_int()
            self.eat(",")
            left_positions = self.parse_positions()
            self.eat(",")
            left = self.parse_proof_term()
            self.eat(",")
            right = self.parse_proof_term()
            self.eat(")")
            return RawTensor(index, left_positions, left, right)
        raise ParseError(
            f"unexpected proof term at {self.i}: {self.remaining()[:20]!r}"
        )


def parse(line: str) -> tuple[Sequent, RawProof]:
    """Parse a dataset line into a conclusion sequent and a raw proof term."""
    text = line.strip()
    parser = Parser(text)
    sequent = parser.parse_sequent()
    parser.eat(" || ")
    raw = parser.parse_proof_term()
    parser.eat(".")
    if not parser.eof():
        raise ParseError(f"trailing junk: {parser.remaining()!r}")
    return sequent, raw


def try_parse(line: str) -> tuple[Sequent, RawProof] | None:
    try:
        return parse(line)
    except ParseError:
        return None
