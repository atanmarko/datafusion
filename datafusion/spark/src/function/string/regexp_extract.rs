// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{
    Array, ArrayBuilder, ArrayRef, GenericStringBuilder, Int64Array, OffsetSizeTrait,
    StringArrayType, StringViewBuilder,
};
use arrow::datatypes::DataType;
use datafusion_common::arrow::datatypes::{Field, FieldRef};
use datafusion_common::cast::{
    as_generic_string_array, as_int64_array, as_string_view_array,
};
use datafusion_common::types::{
    NativeType, logical_int32, logical_int64, logical_string,
};
use datafusion_common::{Result, exec_err};
use datafusion_expr::{Coercion, ReturnFieldArgs, TypeSignatureClass};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use regex::Regex;
use std::any::Any;
use std::sync::Arc;

/// Spark-compatible `regexp_extract` expression
/// <https://spark.apache.org/docs/latest/api/sql/index.html#regexp_extract>
/// Extract the first string in the str that match the regexp expression and corresponding to the regex group index.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkRegexpExtract {
    signature: Signature,
}

impl Default for SparkRegexpExtract {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkRegexpExtract {
    pub fn new() -> Self {
        // Types of the arguments. Possible coercions
        let string = Coercion::new_exact(TypeSignatureClass::Native(logical_string()));
        let int64 = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_int64()),
            vec![TypeSignatureClass::Native(logical_int32())],
            NativeType::Int64,
        );
        Self {
            // Signature of the function regexp_extract(str, regexp, idx)
            // first argument: input string
            // second argument: regex pattern string
            // third argument: integer capture-group index
            signature: Signature::new(
                TypeSignature::Coercible(vec![
                    string.clone(),
                    string.clone(),
                    int64.clone(),
                ]),
                Volatility::Immutable,
            )
            .with_parameter_names(vec![
                "str".to_string(),
                "pattern".to_string(),
                "idx".to_string(),
            ])
            .expect("valid parameter names"),
        }
    }
}

// DataFusion integration.
// This struct is a scalar user-defined function implementation.
impl ScalarUDFImpl for SparkRegexpExtract {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "regexp_extract"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    // DataFusion executes the function
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // Bridge between DataFusion and actual execution function.
        // Use the implementation in `spark_regexp_extract`
        // When someone calls regexp_extract, run the array-based implementation named spark_regexp_extract
        make_scalar_function(spark_regexp_extract, vec![])(&args.args)
    }

    // Do not use this one, use `return_field_from_args`
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion_common::internal_err!(
            "return_type should not be called for Spark regexp_extract"
        )
    }

    // This function tells DataFusion what kind of column comes out
    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        // Spark semantics: regexp_extract returns NULL if ANY input is NULL
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());

        // Returns a field describing the output column:
        // name: "regexp_extract"
        // type: same string type as input (input could be Utf8, LargeUtf8, Utf8View)
        // nullable: yes if any input is nullable (Spark semantics if any input is null, result is null)
        Ok(Arc::new(Field::new(
            "regexp_extract",
            args.arg_fields[0].data_type().clone(),
            nullable,
        )))
    }
}

fn spark_regexp_extract(args: &[ArrayRef]) -> Result<ArrayRef> {
    let idx_array = as_int64_array(&args[2])?;

    // Input data format handling
    match args[0].data_type() {
        DataType::Utf8 => {
            let string_array = as_generic_string_array::<i32>(&args[0])?;
            let pattern_array = as_generic_string_array::<i32>(&args[1])?;
            spark_regexp_extract_impl(
                &string_array,
                &pattern_array,
                idx_array,
                GenericStringBuilder::<i32>::new(),
            )
        }
        DataType::LargeUtf8 => {
            let string_array = as_generic_string_array::<i64>(&args[0])?;
            let pattern_array = as_generic_string_array::<i64>(&args[1])?;
            spark_regexp_extract_impl(
                &string_array,
                &pattern_array,
                idx_array,
                GenericStringBuilder::<i64>::new(),
            )
        }
        DataType::Utf8View => {
            let string_array = as_string_view_array(&args[0])?;
            let pattern_array = as_string_view_array(&args[1])?;
            spark_regexp_extract_impl(
                &string_array,
                &pattern_array,
                idx_array,
                StringViewBuilder::new(),
            )
        }
        other => exec_err!("Unsupported data type {other:?} for function regexp_extract"),
    }
}

trait StringArrayBuilder: ArrayBuilder {
    fn append_value(&mut self, val: &str);
    fn append_null(&mut self);
}

impl<O: OffsetSizeTrait> StringArrayBuilder for GenericStringBuilder<O> {
    fn append_value(&mut self, val: &str) {
        GenericStringBuilder::append_value(self, val);
    }
    fn append_null(&mut self) {
        GenericStringBuilder::append_null(self);
    }
}

impl StringArrayBuilder for StringViewBuilder {
    fn append_value(&mut self, val: &str) {
        StringViewBuilder::append_value(self, val);
    }
    fn append_null(&mut self) {
        StringViewBuilder::append_null(self);
    }
}

// Actual function implementation
fn spark_regexp_extract_impl<'a, V, B>(
    string_array: &'a V,
    pattern_array: &'a V,
    idx_array: &Int64Array,
    mut builder: B,
) -> Result<ArrayRef>
where
    V: StringArrayType<'a>,
    B: StringArrayBuilder,
{
    for i in 0..string_array.len() {
        if string_array.is_null(i) || pattern_array.is_null(i) || idx_array.is_null(i) {
            builder.append_null();
            continue;
        }

        let value = string_array.value(i);
        let pattern = pattern_array.value(i);
        let idx = idx_array.value(i);

        if idx < 0 {
            return exec_err!(
                "regexp_extract group index must be non-negative, got {idx}"
            );
        }

        let regex = Regex::new(pattern).map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Invalid regex in regexp_extract: {e}"
            ))
        })?;

        match regex.captures(value) {
            Some(captures) => match captures.get(idx as usize) {
                Some(m) => builder.append_value(m.as_str()),
                None => builder.append_value(""),
            },
            None => builder.append_value(""),
        }
    }

    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use std::sync::Arc;

    #[test]
    fn test_regexp_extract_basic() -> Result<()> {
        let values = Arc::new(StringArray::from(vec!["abc-123"])) as ArrayRef;
        let patterns = Arc::new(StringArray::from(vec!["([a-z]+)-([0-9]+)"])) as ArrayRef;
        let idx = Arc::new(Int64Array::from(vec![1])) as ArrayRef;

        let result = spark_regexp_extract(&[values, patterns, idx])?;

        let result = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("result should be StringArray");

        assert_eq!(result.value(0), "abc");

        Ok(())
    }

    #[test]
    fn test_regexp_extract_idx_zero_returns_full_match() -> Result<()> {
        let values = Arc::new(StringArray::from(vec!["abc-123"])) as ArrayRef;
        let patterns = Arc::new(StringArray::from(vec!["([a-z]+)-([0-9]+)"])) as ArrayRef;
        let idx = Arc::new(Int64Array::from(vec![0])) as ArrayRef;

        let result = spark_regexp_extract(&[values, patterns, idx])?;

        let result = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("result should be StringArray");

        assert_eq!(result.value(0), "abc-123");

        Ok(())
    }

    #[test]
    fn test_regexp_extract_second_group() -> Result<()> {
        let values = Arc::new(StringArray::from(vec!["abc-123"])) as ArrayRef;
        let patterns = Arc::new(StringArray::from(vec!["([a-z]+)-([0-9]+)"])) as ArrayRef;
        let idx = Arc::new(Int64Array::from(vec![2])) as ArrayRef;

        let result = spark_regexp_extract(&[values, patterns, idx])?;

        let result = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(result.value(0), "123");

        Ok(())
    }

    #[test]
    fn test_regexp_extract_null_pattern_returns_null() -> Result<()> {
        let values = Arc::new(StringArray::from(vec![Some("abc-123")])) as ArrayRef;
        let patterns = Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef;
        let idx = Arc::new(Int64Array::from(vec![Some(1)])) as ArrayRef;

        let result = spark_regexp_extract(&[values, patterns, idx])?;

        let result = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("result should be StringArray");

        assert!(result.is_null(0));

        Ok(())
    }

    #[test]
    fn test_regexp_extract_null_idx_returns_null() -> Result<()> {
        let values = Arc::new(StringArray::from(vec![Some("abc-123")])) as ArrayRef;
        let patterns =
            Arc::new(StringArray::from(vec![Some("([a-z]+)-([0-9]+)")])) as ArrayRef;
        let idx = Arc::new(Int64Array::from(vec![None])) as ArrayRef;

        let result = spark_regexp_extract(&[values, patterns, idx])?;

        let result = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("result should be StringArray");

        assert!(result.is_null(0));

        Ok(())
    }

    #[test]
    fn test_regexp_extract_invalid_regex_returns_error() {
        let values = Arc::new(StringArray::from(vec!["abc-123"])) as ArrayRef;
        let patterns = Arc::new(StringArray::from(vec!["("])) as ArrayRef;
        let idx = Arc::new(Int64Array::from(vec![1])) as ArrayRef;

        let result = spark_regexp_extract(&[values, patterns, idx]);

        assert!(result.is_err());
    }

    #[test]
    fn test_regexp_extract_negative_idx_returns_error() {
        let values = Arc::new(StringArray::from(vec!["abc-123"])) as ArrayRef;
        let patterns = Arc::new(StringArray::from(vec!["([a-z]+)-([0-9]+)"])) as ArrayRef;
        let idx = Arc::new(Int64Array::from(vec![-1])) as ArrayRef;

        let result = spark_regexp_extract(&[values, patterns, idx]);

        assert!(result.is_err());
    }

    #[test]
    fn test_regexp_extract_optional_group_not_matched_returns_empty_string() -> Result<()>
    {
        let values = Arc::new(StringArray::from(vec!["b"])) as ArrayRef;
        let patterns = Arc::new(StringArray::from(vec!["(a)?b"])) as ArrayRef;
        let idx = Arc::new(Int64Array::from(vec![1])) as ArrayRef;

        let result = spark_regexp_extract(&[values, patterns, idx])?;

        let result = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("result should be StringArray");

        assert_eq!(result.value(0), "");

        Ok(())
    }

    #[test]
    fn test_regexp_extract_multiple_rows() -> Result<()> {
        let values = Arc::new(StringArray::from(vec![
            Some("abc-123"),
            Some("def-456"),
            Some("hello"),
            None,
        ])) as ArrayRef;

        let patterns = Arc::new(StringArray::from(vec![
            Some("([a-z]+)-([0-9]+)"),
            Some("([a-z]+)-([0-9]+)"),
            Some("([a-z]+)-([0-9]+)"),
            Some("([a-z]+)-([0-9]+)"),
        ])) as ArrayRef;

        let idx = Arc::new(Int64Array::from(vec![Some(2), Some(1), Some(1), Some(1)]))
            as ArrayRef;

        let result = spark_regexp_extract(&[values, patterns, idx])?;

        let result = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("result should be StringArray");

        assert_eq!(result.value(0), "123");
        assert_eq!(result.value(1), "def");
        assert_eq!(result.value(2), "");
        assert!(result.is_null(3));

        Ok(())
    }
}
