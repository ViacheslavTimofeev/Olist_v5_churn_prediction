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
from olist_churn_prediction.schemas_raw import _NaN2NoneMixin

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
'''
class MainClassifiedSchemaInterim(MainClassifiedSchemaRaw):
    product_category_name_english: Optional[str] = Field(default=None) # новый
    id: Optional[int] = Field(default=None, exclude=True, repr=False) # удален
    order_sellers_qty: Optional[int] = Field(default=None, exclude=True, repr=False) # удален
    product_category_name: Optional[str] = Field(default=None, exclude=True, repr=False) # удален
    review_comment_title: Optional[str] = Field(default=None, exclude=True, repr=False) # удален
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
'''
class MergedDfSchemaInterim(_NaN2NoneMixin):
    order_id: str
    order_status: Optional[str] = Field(default=None, exclude=True, repr=False) # удален
    order_products_value: float
    order_freight_value: float
    order_items_qty: int
    order_sellers_qty: Optional[int] = Field(default=None, exclude=True, repr=False) # удален
    order_purchase_timestamp: Optional[datetime] = Field(default=None)
    order_aproved_at: Optional[datetime] = Field(default=None)
    order_estimated_delivery_date: Optional[datetime] = Field(default=None)
    order_delivered_customer_date: Optional[datetime] = Field(default=None)
    customer_id: str
    customer_unique_id: str
    customer_city: str
    customer_state: str
    customer_zip_code_prefix: str
    product_id: str
    product_category_name: Optional[str] = Field(default=None, exclude=True, repr=False) # удален
    product_category_name_english: str
    product_name_lenght: int
    product_description_lenght: int
    product_photos_qty: int
    review_id: str
    review_score: int
    review_comment_title: Optional[str] = Field(default=None, exclude=True, repr=False) # удален
    review_comment_message: Optional[str] = Field(default=None, exclude=True, repr=False) # удален
    review_creation_date: Optional[datetime] = Field(default=None)
    review_answer_timestamp: Optional[datetime] = Field(default=None)
    value: float
    installments: int
    sequential: int
    product_weight_g: float
    product_length_cm: float
    product_height_cm: float
    product_width_cm: float
    seller_id: str
    seller_zip_code_prefix: str
    seller_city: str
    seller_state: str
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")