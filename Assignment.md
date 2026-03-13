# RegexpExtract Implementation

## Overview

This assignment implements the Spark-compatible `regexp_extract`
function in the DataFusion Spark module.

`regexp_extract(str, pattern, idx)` extracts a substring from `str` that
matches a regular expression capture group defined by `pattern`.

Example:

regexp_extract("abc-123", "(\[a-z\]+)-(\[0-9\]+)", 1) → "abc"\
regexp_extract("abc-123", "(\[a-z\]+)-(\[0-9\]+)", 2) → "123"

If the pattern does not match or the capture group does not exist, the
function returns an empty string.

------------------------------------------------------------------------

# Design

The implementation follows the existing architecture used for
Spark-compatible functions in the DataFusion Spark module.

The solution consists of three main components:

1.  Function definition and registration
2.  Arrow type dispatch
3.  Vectorized execution logic

------------------------------------------------------------------------

# Function Definition

The function is implemented as a `ScalarUDFImpl`:

``` rust
pub struct SparkRegexpExtract {
    signature: Signature,
}
```

The function signature accepts three parameters:

-   `str`: input string
-   `pattern`: regex pattern
-   `idx`: capture group index

```{=html}
<!-- -->
```
    (str: string, pattern: string, idx: int64)

The function is registered within the Spark string functions module.

------------------------------------------------------------------------

# Arrow Type Dispatch

DataFusion operates on Arrow arrays rather than scalar values, so the
function must support multiple Arrow string types.

The implementation handles:

-   `Utf8`
-   `LargeUtf8`
-   `Utf8View`

Each variant converts `ArrayRef` inputs into concrete string array types
using casting helpers and then forwards execution to a shared
implementation.

This pattern is consistent with other Spark string functions such as
`substring`.

------------------------------------------------------------------------

# Execution Logic

The core logic is implemented in a shared function:

    spark_regexp_extract_impl(...)

The function iterates over Arrow arrays in a vectorized loop:

    for i in 0..string_array.len()

For each row:

1.  If any input is NULL → return NULL
2.  Compile the regex pattern
3.  Match the regex against the input string
4.  Extract capture group `idx`

Behavior rules:

Scenario                          Result
  --------------------------------- ---------------------------
Regex matches and group exists    return captured substring
Regex matches but group missing   return empty string
Regex does not match              return empty string
Input is NULL                     return NULL
Negative group index              execution error

Regex matching is implemented using the Rust `regex` crate.

------------------------------------------------------------------------

# Edge Cases

The implementation handles several important edge cases:

-   NULL propagation for any NULL input
-   Optional capture groups returning empty strings when not matched
-   Invalid regex patterns returning execution errors
-   Negative capture group indices rejected

------------------------------------------------------------------------

# Testing

Unit tests were added directly in the module to verify behavior.

Test coverage includes:

-   basic capture group extraction
-   extracting different capture groups
-   cases where the regex does not match
-   missing capture groups
-   NULL input propagation
-   invalid regex patterns
-   vectorized execution across multiple rows

These tests ensure correctness for both simple and edge-case scenarios.

------------------------------------------------------------------------

# Performance Considerations

The current implementation compiles the regex for each row.

A potential optimization would be to detect constant patterns and
compile the regex once per batch rather than per row.

However, the current implementation prioritizes correctness, clarity,
and consistency with existing Spark function patterns in DataFusion.


------------------------------------------------------------------------

# Test the implementation

```bash
cargo test -p datafusion-spark --lib function::string::regexp_extract
```



------------------------------------------------------------------------

# Summary

This implementation integrates `regexp_extract` into the DataFusion
Spark module using the same architecture as existing Spark string
functions.

The function supports Arrow vectorized execution, handles relevant edge
cases, and includes unit tests validating its behavior across multiple
scenarios.