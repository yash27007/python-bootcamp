# Object-Oriented Programming

## Problem

Once a program needs more than a few independent variables and functions, a new kind of pressure
shows up: several pieces of data genuinely belong together (a bank account's owner *and* its
balance, not two separate loose variables that happen to share an index somewhere), and several
different *kinds* of thing need to respond to the same operation in their own way (a shape's
"area," a vehicle's "start engine," an animal's "speak" — each one computed differently depending
on what specifically it is). The question this topic answers: how does a program bundle related
data with the functions that operate on it, and let code written once work correctly across many
different specific kinds of a thing, without either duplicating logic per kind or writing a giant
`if/elif` chain that has to be extended by hand every time a new kind is added?

## Intuition

Compare two ways to represent a bank account. As loose data: `owner = "Yashwanth"`,
`balance = 1000.0`, and a `deposit(owner, balance, amount)` function that takes and returns the
balance explicitly every time — nothing stops `balance` from being passed to a completely
unrelated function, or two accounts' balances from being mixed up because they're just numbers
sitting in variables with similar names. As an object: `account = BankAccount("Yashwanth")`,
`account.deposit(1000)` — the balance lives *inside* the account, the functions that are allowed
to touch it are attached to it, and there is no way to accidentally hand one account's balance to
another account's withdraw logic, because each account carries its own state along with the
operations on it.

**Encapsulation** is that bundling — data and the functions that operate on it, kept together, with
control over what's exposed. **Inheritance** is reuse across kinds that share structure — a
`Tesla` doesn't need to redefine what a `Car` already knows how to do, only what's different about
it. **Polymorphism** is one interface, many implementations — `shape.area()` computes a different
formula depending on whether `shape` is actually a `Rectangle` or a `Square`, and the calling code
doesn't need to know which.

## Why simpler approaches fail

**"Just use plain functions and pass the data around explicitly."** This is a real, viable style
(and Python supports it fine) — but as the Bank Account example above shows, it loses the
guarantee that a given piece of state and the functions allowed to touch it travel together.
Nothing stops `withdraw(balance_a, amount)` from being called with `balance_b` by a typo. It also
has no answer for polymorphism: `area(shape)` for a growing list of shape kinds either becomes a
long `if isinstance(shape, Rectangle): ... elif isinstance(shape, Square): ...` chain that has to
be edited every time a new shape is added, or it has to be a differently-named function per shape
(`rectangle_area`, `square_area`, ...), pushing the "which formula" decision onto every single
caller instead of onto the shape itself.

`class` doesn't add a new capability Python didn't already have — the From-scratch section below
rebuilds a working "object" with nothing but closures and a dict — but it removes the burden of
re-deriving that bundling correctly, by hand, every time, and gives it a name (a type) that
`isinstance`, inheritance, and polymorphic dispatch can all build on.

## Conceptual foundation

*(Substituting for "Mathematical foundation" — OOP has no numeric derivation; the underlying
mechanism to make explicit is what `class`, inheritance, and method dispatch actually do, with a
real worked example for each, not just the vocabulary.)*

**Encapsulation, concretely.** A `Person` with `self.__name`/`self.__age` (double-underscore)
cannot have those attributes read or written directly from outside the class — Python renames
them internally (`_Person__name`) specifically to make accidental external access fail loudly.
Controlled access goes through `@property` getters/setters, which is what makes *validation*
possible: the practical notebook's `Person.age` setter rejects `age = 200` with a `ValueError`
*before* the invalid value ever reaches `self.__age`, so `person.age` is guaranteed to be a valid
value at every point after construction, not just "usually" valid. This is the real payoff of
encapsulation — not hiding data for its own sake, but making an invariant ("age is always between
0 and 130") impossible to violate from outside the class, in one enforced place, instead of relying
on every caller everywhere to remember to check.

**Inheritance, concretely.** `Tesla(Car)` gets `Car`'s `__init__` (via `super().__init__(...)`)
and `Car`'s `drive()` method for free, and adds `isSelfDriving`/`selfDriving()` on top — the
`windows`/`doors`/`engineType` logic is written once, in `Car`, and never re-typed in `Tesla`.
This is real code reuse, not just a taxonomy exercise: a bug fix or extension to `Car.drive()`
automatically applies to every subclass, without touching `Tesla` at all.

**Polymorphism, concretely.** `Rectangle.area()` and `Square.area()` compute genuinely different
formulas, but both are reachable through the same `.area()` call — `animal_speak(dog)` and
`animal_speak(cat)` run different code (`Dog.speak`/`Cat.speak`) through the identical call
`animal.speak()`, because Python resolves `.speak()` at *runtime*, based on the object's actual
type, not the declared parameter type. This is what removes the `if isinstance(...): elif
isinstance(...):` chain from the Why-simpler-approaches-fail section — the dispatch decision moves
from the caller into the type system itself, and adding a new shape only means adding a new class
with its own `.area()`, not editing existing dispatch logic anywhere else.

**Composition vs. inheritance — a genuine design decision, not vocabulary.** Both model code
reuse, but they answer different questions. Inheritance answers "is this new type a more specific
version of an existing type, for *every* behavior that type offers?" Composition answers "does
this object need *part* of another object's behavior, possibly swappable at runtime, without
becoming that other type?" The notebook's `Tesla(Car)` is a clean case for inheritance — a Tesla
really is a Car in every respect `Car` defines. `ComposedCar` (HAS-A `Engine`) is the composition
alternative for the same domain: an engine can be swapped on an *existing* `ComposedCar` at
runtime (`electric_car.engine = Engine("Hybrid")`) with no change to any class definition —
something inheritance, which is fixed at class-definition time, cannot do. The `Bird`/`Penguin`
example makes the tradeoff sharp: `Penguin(Bird)` looks like clean is-a inheritance until `fly()`
turns out not to apply, forcing an override that either lies or raises — the base class made a
promise (every `Bird` can `fly()`) that isn't actually true of every subtype. Composing a
`Penguin` with a `Swim` locomotion object instead of inheriting a `fly()` it can't honor sidesteps
the broken promise entirely, at the cost of an explicit `self.locomotion.move()` delegation
instead of free inherited behavior.

## Algorithm

Building an "object" without `class`, then adding what `class` provides:

1. **Bundle state and behavior with a closure.** A function creates local state (a dict) and
   inner functions that read/write it via closure, then returns those functions — the state is
   private (nothing outside the outer function can see it directly) and the returned functions are
   the only way to touch it.
2. **Observe what's missing.** Every call to the outer function builds brand-new function objects
   — no sharing across instances, no `type()` identity, no way to express "this new thing extends
   that existing thing."
3. **`class` provides all three at once**: methods defined once on the class, shared by every
   instance and looked up through the instance (`self`) at call time; a real type (`isinstance`
   works); and `class Child(Parent):` as a built-in mechanism for "reuse Parent's behavior, add or
   override what's different."
4. **Dispatch.** A method call `obj.method()` looks up `method` starting from `type(obj)` (the
   instance's actual class), not from whatever type a variable was declared or annotated as — this
   is the mechanism polymorphism above relies on.

## From-scratch implementation

OOP *is* the practical implementation here — there's no lower-level derivation the way there is
for hashing or Big-O. The from-scratch bridge instead rebuilds what `class` replaces: the exact
`BankAccount` example from section 1, using only closures and a dict, then measures the one
concrete cost that motivates `class`'s method sharing. See [`main.ipynb`](main.ipynb), section "0.
From-scratch: what `class` replaces."

Real executed output — identical behavior to the `class`-based `BankAccount`:

```
The amount of 1000 is deposited and the balance is now : 1000.0
Insufficient funds!!!
400 is withdrawn. New Balance is 600.0
600.0
```

Measured cost of the closure version — 1,000 closure-based accounts build 1,000 separate `deposit`
function objects; 1,000 class-based accounts share one:

```
1000 closure accounts -> 1000 distinct 'deposit' function objects
1000 class accounts   -> 1 distinct 'deposit' function object(s)
```

## Practical implementation

The full practical notebook — [`main.ipynb`](main.ipynb) — covers, with real executed examples:
class/object basics (`class Car: pass`, `dir()`), instance construction and methods, single and
multiple inheritance (`super().__init__(...)` vs. calling each parent's `__init__` explicitly),
method overriding and duck typing, polymorphism through a `Shape`/`Rectangle`/`Square` hierarchy,
abstract base classes (`abc.ABC`/`@abstractmethod`) enforcing that subclasses implement required
methods, encapsulation via public/protected (`_name`)/private (`__name`) attributes, getters and
setters via `@property` with real input validation, magic/dunder methods (`__str__`, `__repr__`),
operator overloading (`__add__`/`__sub__`/`__mul__`/`__eq__` on a `Vector` class), and custom
exception hierarchies. This maps directly back to the Conceptual foundation: every mechanism
described there (encapsulation's `@property` validation, inheritance's shared `Car`/`Tesla` logic,
polymorphism's runtime dispatch) is exercised with real, pasted output in this notebook, plus the
composition-vs-inheritance comparison (`ComposedCar`, `Bird`/`Penguin`) added alongside it.

## Experiment

The from-scratch comparison above is this topic's experiment: same behavior (the bank account),
two implementations (closures vs. `class`), one measured structural difference (shared vs.
duplicated method objects across 1,000 instances) rather than a timing comparison — the point
being made is about *sharing*, not speed. **Hypothesis (stated before running):** a class's
methods, being defined once on the class, will be the same function object across every instance;
a closure's inner functions, being created fresh inside every call to the outer function, will not.
**Result:** confirmed exactly — 1,000 distinct `deposit` objects for the closures, 1 for the class
(pasted above). **Limitations:** this measures object identity, not memory bytes or wall-clock
time directly — the qualitative claim (class methods are shared; closures are not) is the point,
not an absolute memory-savings number.

## Failure modes

- **Deep inheritance hierarchies — the fragile base class problem.** Every subclass in an
  inheritance chain depends on the exact behavior of every ancestor above it. A change to a base
  class's method — even one that looks safe in isolation — can silently break a distant
  subclass that depended on the old behavior, without touching that subclass's own code at all.
  The deeper the hierarchy, the more of this hidden, undocumented dependency accumulates, and the
  harder it becomes to change any single class without auditing everything beneath it.
- **Overusing inheritance where composition would be simpler.** Reproduced for real:
  `Penguin(Bird)` inherits a `fly()` it cannot honor, forcing a subclass override that raises
  instead of behaving —

  ```
  flying
  broke the substitution: Penguins can't fly
  ```

  — versus composing a `Penguin` with the locomotion it actually has (`Swim`), which never makes a
  promise it can't keep:

  ```
  swimming
  flying
  ```

  The tell that inheritance is the wrong tool here: the subclass has to *override to remove or
  break* a capability the base class advertised, rather than purely *adding* to it — a sign the
  "is-a" relationship was true for some of the base class's behavior but not all of it.

## Real-world usage

- **ORMs and data models** (Django models, SQLAlchemy, Pydantic) are encapsulation applied at
  scale — validation logic lives on the class, in one place, exactly like the `Person.age` setter
  above, instead of being re-checked at every call site that touches the data.
  Contrast with `07-nlp`'s Pydantic models for structured-output validation elsewhere in this repo.
- **Framework base classes** (`sklearn.BaseEstimator`, PyTorch's `nn.Module`) use inheritance so
  that user-defined models automatically get shared behavior (`.fit()`/`.predict()`'s calling
  convention, parameter registration) by subclassing one base class, exactly like `Tesla(Car)`
  above.
- **Dependency injection and plugin systems** lean on composition specifically for the swap-at-
  runtime property demonstrated by `ComposedCar` — a payment processor, a logging backend, or a
  storage layer is handed to an object as a constructor argument rather than baked into the class
  hierarchy, so it can be swapped (including for a test double) without redefining any class.
- **Custom exception hierarchies** (`ModelNotTrainedError`, `DataValidationError` in the practical
  notebook) let calling code catch precisely the failure it knows how to handle, at the right
  level of specificity — the same principle covered in `06-file-exception`'s exception-hierarchy
  discussion, applied to domain-specific errors instead of built-in ones.

## Mental model

**A `class` bundles state with the functions allowed to touch it (encapsulation), lets one
definition be reused and extended by more specific kinds (inheritance), and lets one call site
work correctly across every one of those kinds by resolving the call against the object's actual
type at runtime (polymorphism) — none of which required new language capability beyond ordinary
functions and dicts, only a name and a mechanism for it.** Reach for inheritance when a new type
is genuinely a more specific version of an existing one, for everything that existing type
promises; reach for composition the moment that promise only holds for *part* of the base type's
behavior, or the behavior needs to change independently of the object's type.

## Questions to think about

1. The closure-based `make_bank_account` and the `class`-based `BankAccount` produce byte-for-byte
   identical printed output. If `class` adds no new *capability*, what concretely does it add, and
   under what circumstances (many instances? need for `isinstance`? need for inheritance?) does
   that difference actually start to matter in practice?
2. `Person.__name`/`Person.__age` use double-underscore name mangling (`_Person__name`), while
   `Employee`'s `self._name` uses a single underscore and is directly readable by the subclass in
   the practical notebook. Explain the concrete difference this makes for a subclass trying to
   access an inherited attribute, using the `dir()` output shown for each.
3. `Rectangle.area()` and `Square.area()` are called through the identical `.area()` interface but
   run different code. Contrast this with the "just use plain functions" alternative from
   Why-simpler-approaches-fail — write out what calling code for a *new* shape (say, `Triangle`)
   would have to change under each approach.
4. The `Penguin(Bird)` failure mode is fixed by switching `Penguin` from inheriting `Bird` to
   composing a `Swim` locomotion object. Construct a *different* concrete example (not birds) where
   an "is-a" relationship holds for most of a base class's behavior but breaks for one specific
   method, and decide whether inheritance-with-an-override or composition is the better fit for it.
5. `ComposedCar.engine` can be reassigned after construction (`electric_car.engine =
   Engine("Hybrid")`), changing the object's behavior without changing its class. Is there an
   inheritance-based way to achieve the same "change behavior on an existing object, at runtime"
   effect — and if the honest answer is no, why not, given that inheritance is resolved at
   class-definition time?
</content>
