# Traced

A taint analysis tool that tracks arithmetic and flow control operations in
Python functions and represents them as a compact, flowchart-like structure.

For a quick glance at what it does, look at [demo.ipynb](demo.ipynb).

## Motivation

We'd like to show the rationale for the medical decisions our code base makes.
Ideally, we'd show what formula was applied to which lab result or biometric
data, which thresholds were applied, and how this process cascaded into
subsequent medical decisions.  But in our medical decision systems, the core
decision logic is often interleaved with non-medical business logic: retrieving
rules and thresholds from databases, fetching patient information, reformatting
data, logging, exception handling, and other infrastructure concerns. While all
of this work is necessary to reach a medical decision, we expect the actual
medical rationale to be as straightforward as a flowchart.

This package extracts straightforward medical rationale from Python code. It
uses the Numba compiler toolchain to insert instrumentation into the Python
code, and simplifies its logic by, for example, inlining function calls, and
eliminating dead code.  When the resulting instrumented and compiled function
executes, it generates a log of the operations and decisions it performed on
objects that were marked for tracing in the function signature. This log looks a
lot like a path through a flowchart. Later, when you want to understand how the
function actually reached its result, you can query this log, pretty-print it,
etc.

## How to use it

1. **Mark the variables you'd like to trace**: Add type hints to the function you'd like to instrument. Annotate the arguments you'd like to trace with `tracer.Traceable[T]`, where `T` is the underlying type of the argument.

2. **Decorate the functions you'd like to trace**: Decorate the function with `@tracer.trace`. This replaces the function with a compiled version that traces its arguments.

3. **Call the function as usual**: Call the wrapped function as you would the unwrapped function. When the function executes, its operations on traceable objects (and any variables derived from them) are logged.

4. **Retrieve the rationale**: After the function has finished executing, you can
   retrieve a compact data structure that represents the operations that were
   carried out on the traceable variables. This data structure is a mix of condition
   statements, return values, and infix expressions.

Here's how this looks:

```python
import tracer
from typing import NamedTuple

class Patient(NamedTuple):
    blood_pressure: float
    heart_rate: int
    temperature: float

def calculate_risk_score(blood_pressure: float, heart_rate: int) -> float:
    """Calculate a risk score based on blood pressure and heart rate."""
    pressure_factor = blood_pressure / 100.0
    rate_factor = heart_rate / 80.0
    return pressure_factor * rate_factor

def is_critical_temperature(temperature: float) -> bool:
    """Check if temperature indicates a critical condition."""
    return temperature > 37.5

@tracer.trace
def assess_patient(patient: tracer.Traceable[Patient]) -> bool:
    bp, hr, temp = patient.blood_pressure, patient.heart_rate, patient.temperature
    risk_score = calculate_risk_score(bp, hr)
    
    if risk_score > 1.5:
        if hr > 100:
            return is_critical_temperature(temp)
    return False
```

You call `assess_patient` like you would any other function:

```python
patient = Patient(blood_pressure=130.0, heart_rate=110, temperature=38.0)
result = assess_patient(patient)

print('Function value:', result)
print('Value provenance:\n', assess_patient.trace.pretty_print())
```

This outputs:

```
Function value: True
Value provenance:
 if (((blood_pressure / 100.0000) * (heart_rate / 80.0000)) > 1.5000) (=True):
  if (heart_rate > 100) (=True):
    return (temperature > 37.5000)
```

## How It Works Internally

1. **Compilation**: Unlike regular Python functions, which are compiled to Python bytecode and executed by the CPython interpreter, functions are compiled using the Numba compiler. But compared to the stock Numba compiler, the compiler is told to implement a few custom passes:

   1. **Inlining**: All function calls are inlined to produce a flat trace. We do this because subroutines and flowcharts don't mix nicely. It also
      allows us to do some aggressive constant folding.

   2. **Tracing Injection**: The compiler injects logging code at key points,
      like binary operations (arithmetic, comparisons),
      conditional branches, returns, loads, and stores.

2. **Trace Collection**: During execution, operations on traceable variables are
   logged. Values that are loaded from memory are matched to previous store
   operations to recover the name of the corresponding variable. This allows us to
   later reconstruct a compact symbolic representation of the expression.

3. **IR Generation**: After the function has finished executing, the recorded
   operations are converted into an even simpler expression tree that consists of
   just if-statements, infix expressions, and return values.

## Does this problem really need a compiler?

You might be wondering if we couldn't do this kind of tracing using runtime
tricks, for example by overloading the operators of traceable objects to log
arithmetic operations. This was the original idea, and I think it can work.
There's just one complication: there is no way to overload if-statements in Python to
trace objects.  In Python, the only way for an object to know that it's being used
as a condition for an if-statement is that its `__bool__` method gets called.
But other things can cause `__bool__` to get called. Furthermore, operations
don't know what branch of an if-statement they're occupying. They have to assume
that they're 1) in a branch of an if-statement because `__bool__` has been
called, and 2) they're in the branch of the if-statement dictated by the value
of the bool.  If one is careful, the trace will never be technically wrong, but
it can be misleading.

Consider:

```python
if traceable > 0.5:
    a = 2
else:
    a = 0.5

return 3
```

Ideally, the log for this snippet would just show that 3 is returned regardless
of the value of `traceable`.  Instead, because a runtime logger has to assume
that every call to `__bool__` is the result of an if-statement, and that the branch was
taken, it must output something like

```
if traceable > 0.5:
    return 3
```

This trace is technically correct, but it's misleading because it leads the
reader to conclude that `traceable` influences the output.

Here is another confusing but technically correct situation:

```python
r = (traceable > 0.5)

if traceable2 > 0:
    return r
else:
    return -1
```

Since the tracer has to assume `__bool__` calls are conditionals, it must
assumes the statement `r = (traceable > 0.5)` is a branch statement on `r`.  But
no branch is taken on `r` (in fact `r` is completely ignored when `traceable2 <
0`).
