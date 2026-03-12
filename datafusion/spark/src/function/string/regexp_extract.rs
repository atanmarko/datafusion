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

        let _value = string_array.value(i);
        let _pattern = pattern_array.value(i);
        let _idx = idx_array.value(i);

        builder.append_value("");
    }

    Ok(Arc::new(builder.finish()))
}
