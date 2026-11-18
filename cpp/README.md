# C++ port

Goal: a C++17 version of `python/estimator.py` that gives the same output as the Python version to within 1e-9 m on every scenario, with no heap allocation in the update loop.

## Target layout

```
cpp/
  CMakeLists.txt
  include/track/estimator.hpp   # types, functions, class declaration
  src/estimator.cpp             # implementation
  src/main.cpp                  # CSV runner, same contract as python/runner.py
  tests/test_kalman.cpp         # maths tests
  tests/test_estimator.cpp      # behaviour tests
```

Build (once stage 1 exists):

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

## Stages

| # | Stage | Build | C++ concepts | Done when |
|---|---|---|---|---|
| 1 | Skeleton + tests | Library, executable and test executable (GoogleTest via `FetchContent`); strict warnings | Headers vs sources, `#pragma once`, namespaces, CMake targets, `ctest` | Build and one trivial test pass with `-Wall -Wextra -Wpedantic -Werror` |
| 2 | Types | Constants, `AxisState`, `Covariance`, `TargetState` | Structs, `constexpr`, `std::array`, `const` | A test constructs each type |
| 3 | Pure maths | `transition`, `propagate_state`, `process_noise`, `propagate_covariance`, `update_axis`; port `tests/test_kalman.py` | Functions, value vs `const&`, structured bindings, `<cmath>` | Every function matches Python to 1e-12 |
| 4 | Estimator class | `reset`, restart, `ingest_measurement`, `predict` | Classes, private members, `std::optional`, `const` methods, exceptions, `noexcept` | A straight line gives exact position and velocity |
| 5 | Behaviour tests | Port the key cases of `tests/test_estimator.py` | Test design | All ported tests pass |
| 6 | CSV runner | `main.cpp`: same input check, 50 Hz ticks and output columns as `python/runner.py` | `<fstream>`, parsing, `std::vector` outside the loop, `std::setprecision(17)` | Output files for both tracks |
| 7 | Equivalence | A script in `tools/` running both versions on every scenario and reporting the largest difference | Floating-point determinism, operation order, why `-ffast-math` breaks equality | Max difference < 1e-9 m |
| 8 | Real-time hygiene | No allocation in `ingest`/`predict`; `std::chrono` timing (median, max); Debug vs Release; optional `float` vs `double` | Stack vs heap, `-O2`, templates | Timing table in the README |
| 9 | Polish (optional) | Sanitizers, `clang-format`, CI build and test | Tooling | CI green |

## Python → C++ map

| Python | C++ |
|---|---|
| Module constants | `constexpr double kAccelTauS = 6.0;` in `namespace track` |
| `AxisState` (3-tuple) | `struct AxisState { double p, v, a; };` |
| `Covariance` (6-tuple, upper triangle) | `struct Covariance { double pp, pv, pa, vv, va, aa; };` |
| `TargetState` dataclass | `struct TargetState { double t; std::array<double, 3> position, velocity; };` |
| Tuple returns and unpacking | Small structs + `auto [a, b, c] = f();` |
| `float \| None` | `std::optional<double>` |
| `predict` stores nothing | `TargetState predict(double t) const;` (the compiler enforces read-only) |
| `raise ValueError` | `throw std::invalid_argument(...)` |
| `math.isfinite` | `std::isfinite` |

**Matching Python to 1e-12:** write each expression in the same order as the Python code. Floating-point addition isn't associative, so a reordered sum can differ in the last bits.
