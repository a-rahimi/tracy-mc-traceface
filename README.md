# Traced

A Python taint analysis tool that tracks operations and represents them as a
compact, flowchart-like structure.

## Motivation

In our medical decision systems, the core decision logic is often interleaved
with non-medical business logic: retrieving rules and thresholds from databases,
fetching patient information, reformatting data, and other infrastructure
concerns. While all of this work is necessary to execute a decision, we expect
the actual medical logic to be straightforward as a flowchart: Some formulas are
applied to lab results or biometric data, some thresholds are applied, and the
process a repeated a few times until we reach a final decision.

This package solves this by running taint analysis on Python code. It consists
of a Python compiler (it uses Numba's compiler toolchain under the hood) to
inserting instrumentation into the code and simplify its logic by inlining
function calls.  When the resulting compiled function executes, it generates a
log of the operations and decisions it performed on objects that subclass from
`trace.Traceable`. This log looks a lot like a path through a flowchart. Later,
when you want to undersatand how the function actually reached its result, you
can query this log, pretty-print it, etc.
 
## How to use it

1. **Mark traceable data**: Subclass the data you want to trace from `Traceable`. For example, if you want to trace the operations on certain integer or floating point values, define an Int or Float class that inherits from the builtin float or int classes, and mixing the Traceable base class.
2. **Decorate functions**: Use `@tracer.trace` to wrap decision functions.
3. **Use the function as usual**: Call the wrapped function as you would the unwrapped function.
3. **Get a flowchart**: The system automatically tracks operations and produces a readable trace

The tracer uses Numba's compiler infrastructure to inject tracing code, inline function calls, and produce a flattened representation of only the operations on traceable data.


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
    rate_factor = Float(heart_rate) / 80.0
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

# Use the traced function
patient = Patient(blood_pressure=130.0, heart_rate=110, temperature=38.0)
result = assess_patient(patient)

print(result)  # True
print(assess_patient.trace.pretty_print())
# Output:
# if (risk_score > 1.5000) (=True):
#   if (heart_rate > 100) (=True):
#     return (temperature > 37.5000)
```

## How It Works Internally

1. **Compilation**: Functions are compiled using Numba with a custom compiler pipeline
2. **Inlining**: All function calls are inlined to produce a flat trace
3. **Tracing Injection**: The compiler injects logging code at key points:
   - Binary operations (arithmetic, comparisons)
   - Conditionals (branches)
   - Returns
   - Assignments
4. **Trace Collection**: During execution, operations on `Traceable` objects are logged
5. **IR Generation**: The trace is converted to a structured representation showing the decision path
