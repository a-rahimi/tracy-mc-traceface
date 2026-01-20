# Traced

A taint analysis tool that tracks arithmetic operations in Python functions and
represents them as a compact, flowchart-like structure.

## Motivation

Ideally, in a medical rationale, some formulas might be applied to various lab
results or biometric data, some thresholds might be applied, and the process
might be repeated a few times until a final decision is reached. But in our
medical decision systems, the core decision logic is often interleaved with
non-medical business logic: retrieving rules and thresholds from databases,
fetching patient information, reformatting data, logging, exception handling,
and other infrastructure concerns. While all of this work is necessary to obtain
a medical decision, we expect the actual medical rationale to be as
straightforward as a flowchart.

This package extracts straightforward medical rationale from Python code. It
uses a the Numba compiler toolchain to inserting instrumentation into the Python
code, and simplifies its logic by inlining function calls.  When the resulting
instrumented and compiled function executes, it generates a log of the
operations and decisions it performed on objects that subclass from
`trace.Traceable`. This log looks a lot like a path through a flowchart. Later,
when you want to undersatand how the function actually reached its result, you
can query this log, pretty-print it, etc.

## How to use it

1. **Mark the variables you'd like to trace**: Subclass the variables you want
   to trace from `Traceable`. For example, if you want to trace operations on some
   integers or floating point values, define an Int or Float class that inherits
   from the builtin float or int classes, and mixing the Traceable base class. Make
   your variables instances of these variables.

2. **Decorate the functions you'd like to trace**: Use `@tracer.trace` to wrap
   the functions that should produce rationale for the values they compute.

3. **Call the function as usual**: Call the wrapped function as you would the
   unwrapped function. When the function executes, its operations on Traceable
   objects are logged.

4. **Retriee the rationale**: After the function has finished executing, you can
   retrieve a compact data structure that represents the oeprations that were
   carried our on the Traceables. This data structure is a mix of condition
   statements, return values, and infix expressions.

Here's how this looks:

```python
from tracer import trace, Float, Int
from typing import NamedTuple

class Patient(NamedTuple):
    blood_pressure: Float
    heart_rate: Int
    temperature: Float

def calculate_risk_score(blood_pressure: Float, heart_rate: Int) -> Float:
    """Calculate a risk score based on blood pressure and heart rate."""
    pressure_factor = blood_pressure / 100.0
    rate_factor = heart_rate / 80.0
    return pressure_factor * rate_factor

def is_critical_temperature(temperature: Float) -> bool:
    """Check if temperature indicates a critical condition."""
    return temperature > 37.5

@trace
def assess_patient(patient: Patient) -> bool:
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

1. **Compilation**: Unlike regular Python functions, which are compiled to Python bytecode and executed by the CPython interpeter, functions are compiled using the Numba compiler. But compared to the stock Numba compiler, the comiler is told to implement a few custom passes:

   1. **Inlining**: All function calls are inlined to produce a flat trace. We do this because subroutines and flowcharts don't mix nicely. It also
      allows us to do some aggressive constant folding.

   2. **Tracing Injection**: The compiler injects logging code at key points,
      like binary operations (arithmetic, comparisons),
      conditional branches, returns, loads, and stores.

2. **Trace Collection**: During execution, operations on `Traceable` objects are
   logged. Values that are loaded from memory are matched to previous store
   operations to recover the name of the corresponding variable. This allows us to
   later reconstruct a compact symbolic representation of the expression.

3. **IR Generation**: After the function has finished executing, the recorded
   operations are converted into an even simpler expression tree that consists of
   just if-statements, infix expressions, and return values.
