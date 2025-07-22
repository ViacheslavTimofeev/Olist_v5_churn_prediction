from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
import math
import pandas as pd

from olist_churn_prediction.schemas_raw import (
    SellersSchemaRaw,
    MeasuresSchemaRaw,
    TranslationSchemaRaw,
    PaymentsSchemaRaw,
    MainPublicSchemaRaw,
    CustomersSchemaRaw,
    GeolocationSchemaRaw,
    MainClassifiedSchemaRaw )

class SellersSchemaInterim(SellersSchemaRaw):
    """Interim schema – currently identical to raw, but ready for extension."""
    pass
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class MeasuresSchemaInterim(MeasuresSchemaRaw):
    """Interim schema – currently identical to raw, but ready for extension."""
    pass
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")

class TranslationSchemaInterim(TranslationSchemaRaw):
    """Interim schema – currently identical to raw, but ready for extension."""
    pass
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")

class PaymentsSchemaInterim(PaymentsSchemaRaw):
    """Interim schema – currently identical to raw, but ready for extension."""
    pass
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")

class MainPublicSchemaInterim(MainPublicSchemaRaw):
    product_category_name: Optional[str] = Field(default=None, exclude=True, repr=False)
    product_category_name_english: Optional[str] = Field(default=None)
    order_sellers_qty: Optional[int] = Field(default=None, exclude=True, repr=False)
    review_comment_title: Optional[str] = Field(default=None, exclude=True, repr=False)
    order_status: Optional[str] = Field(default=None, exclude=True, repr=False)
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")

class CustomersSchemaInterim(CustomersSchemaRaw):
    """Interim schema – currently identical to raw, but ready for extension."""
    pass
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")

class GeolocationSchemaInterim(GeolocationSchemaRaw):
    """Interim schema – currently identical to raw, but ready for extension."""
    pass
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")

class MainClassifiedSchemaInterim(MainClassifiedSchemaRaw):
    product_category_name_english: Optional[str] = Field(default=None) # новый признак
    id: Optional[int] = Field(default=None, exclude=True, repr=False) # удаленные признаки
    order_sellers_qty: Optional[int] = Field(default=None, exclude=True, repr=False)
    product_category_name: Optional[str] = Field(default=None, exclude=True, repr=False)
    review_comment_title: Optional[str] = Field(default=None, exclude=True, repr=False)
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")