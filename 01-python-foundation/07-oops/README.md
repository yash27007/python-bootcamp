# 07 – Object-Oriented Programming

Detailed notes (encapsulation/inheritance/polymorphism with real worked examples showing *why*
each helps, composition-vs-inheritance argued as a genuine design decision with a concrete example
of each, a from-scratch closure-based "object" built before `class`):
[notes.md](notes.md)

Real, actually-executed notebook covering classes/objects, single/multiple inheritance, method
overriding and polymorphism, abstract base classes, encapsulation with `@property` validation,
magic/dunder methods, operator overloading, custom exceptions, a from-scratch closure-based bank
account compared against the `class` version, and a composition-vs-inheritance comparison
(`ComposedCar`, and a `Bird`/`Penguin` fragile-base-class reproduction), all with real pasted
output: [main.ipynb](main.ipynb)

## What you'll learn

Why bundling state with the functions that operate on it beats passing loose variables around
(encapsulation, with real input validation as the payoff), how inheritance reuses behavior across
related types, how polymorphism resolves a method call against an object's *actual* type at
runtime instead of a caller-side `if isinstance()` chain — and when composition, not inheritance,
is the right tool for reuse.

| Topic | Status |
|-------|--------|
| Classes/objects, instance methods, `dir()` | ✅ Complete |
| Single and multiple inheritance, `super()` | ✅ Complete |
| Polymorphism: method overriding, duck typing, abstract base classes | ✅ Complete |
| Encapsulation: public/protected/private, `@property` getters/setters with real validation | ✅ Complete |
| Magic methods, operator overloading, custom exceptions | ✅ Complete |
| Conceptual foundation: encapsulation/inheritance/polymorphism, each with a real worked example | ✅ Complete |
| From-scratch: closure-based "object" (dict of functions) built before `class`, method-sharing measured | ✅ Complete |
| Composition vs. inheritance: `ComposedCar` (swap at runtime) and `Bird`/`Penguin` (fragile base class), both reproduced | ✅ Complete |
| Failure modes: deep inheritance hierarchies, overusing inheritance where composition fits better | ✅ Complete |

## Why it matters

Encapsulation, inheritance, and polymorphism aren't vocabulary to memorize — each solves a
specific problem that shows up the moment a program has more than a couple of related data-plus-
behavior bundles. Knowing *why* `class` helps (and what it replaces — the from-scratch closure
version shows it's not new capability, just reusable structure) is also what makes its failure
modes — a fragile deep hierarchy, a subclass forced to override-and-break instead of purely
extend — recognizable as a design smell rather than a mystery bug.

## Prerequisites

`04-functions` (closures — the from-scratch bridge is a closure-based object) and
`06-file-exception` (the `__enter__`/`__exit__` context-manager class there is a first taste of a
custom class with real behavior).

## What you'll build

- A `BankAccount` reimplemented with no `class` keyword at all — a closure over private state,
  returning a dict of functions — with byte-for-byte identical output to the `class` version.
- A measured comparison: 1,000 closure-based accounts build 1,000 separate method objects; 1,000
  class-based accounts share one — the concrete cost `class` removes.
- `ComposedCar` (HAS-A `Engine`) alongside the notebook's existing `Tesla(Car)` (IS-A) on the same
  domain, including swapping an engine on an *existing* object at runtime — something inheritance
  can't do.
- A reproduced fragile-base-class failure: `Penguin(Bird)` forced to override `fly()` with a
  crash, fixed by composing a `Swim` locomotion object instead of inheriting a promise it can't
  keep.

## Where it appears in real systems

ORMs and data-validation models (Django, SQLAlchemy, Pydantic) as encapsulation at scale;
framework base classes (`sklearn.BaseEstimator`, PyTorch's `nn.Module`) as inheritance giving
subclasses shared behavior for free; dependency-injection and plugin systems as composition,
swapping behavior at runtime without touching a class hierarchy; and custom exception hierarchies
for precise, catchable domain errors, extending `06-file-exception`'s exception-handling
discussion.

## What's next

`08-advanced-concepts` — iterators/generators, building on classes here (`__iter__`/`__next__` is
another instance of the dunder-method mechanism this topic introduced) before showing what `yield`
automates.
</content>
