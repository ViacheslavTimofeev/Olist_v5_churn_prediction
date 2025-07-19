from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
import math
import pandas as pd

class _NaN2NoneMixin(BaseModel):
    """Добавьте этот миксин в любую модель — и NaN превратится в None."""
    # ⬇︎ сработает для КАЖДОГО поля (строка *, mode='before')
    @field_validator("*", mode="before")
    @classmethod
    def _nan_to_none(cls, v):
        # пропускаем настоящие None
        if v is None:
            return v
        # numpy.nan / float('nan')
        if isinstance(v, float) and math.isnan(v):
            return None
        # pandas NA / NaT
        if pd.isna(v):
            return None
        return v

class SellersSchemaRaw(_NaN2NoneMixin):
    order_id: str
    product_id: str
    seller_id: str
    seller_zip_code_prefix: str
    seller_city: str
    seller_state: str
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")          # лишние или пропущенные колонки → ошибка

class MeasuresSchemaRaw(_NaN2NoneMixin):
    product_id: str
    product_weight_g: float
    product_length_cm: float
    product_height_cm: float
    product_width_cm: float
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class TranslationSchemaRaw(_NaN2NoneMixin):
    product_category_name: str
    product_category_name_english: str
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class PaymentsSchemaRaw(_NaN2NoneMixin):
    order_id: str
    installments: int
    sequential: int
    payment_type: str
    value: float
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class MainPublicSchemaRaw(_NaN2NoneMixin):
    order_id: str
    order_status: str
    order_products_value: float
    order_freight_value: float
    order_items_qty: int
    order_sellers_qty: int
    order_purchase_timestamp: Optional[datetime] = Field(default=None)
    order_aproved_at: Optional[datetime] = Field(default=None)
    order_estimated_delivery_date: Optional[datetime] = Field(default=None)
    order_delivered_customer_date: Optional[datetime] = Field(default=None)
    customer_id: str
    customer_city: str
    customer_state: str
    customer_zip_code_prefix: str
    product_category_name: str
    product_name_lenght: int
    product_description_lenght: int
    product_photos_qty: int
    product_id: str
    review_id: str
    review_score: int
    review_comment_title: str
    review_comment_message: str
    review_creation_date: Optional[datetime] = Field(default=None)
    review_answer_timestamp: Optional[datetime] = Field(default=None)
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class CustomersSchemaRaw(_NaN2NoneMixin):
    customer_id: str
    customer_unique_id: str
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class GeolocationSchemaRaw(_NaN2NoneMixin):
    zip_code_prefix: str
    city: str
    state: str
    lat: float
    lng: float
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class MainClassifiedSchemaRaw(_NaN2NoneMixin):
    id: int
    order_status: str
    order_products_value: float
    order_freight_value: float
    order_items_qty: int
    order_sellers_qty: int
    order_purchase_timestamp: Optional[datetime] = Field(default=None)
    order_aproved_at: Optional[datetime] = Field(default=None)
    order_estimated_delivery_date: Optional[datetime] = Field(default=None)
    order_delivered_customer_date: Optional[datetime] = Field(default=None)
    customer_city: str
    customer_state: str
    customer_zip_code_prefix: str
    product_category_name: str
    product_name_lenght: int
    product_description_lenght: int
    product_photos_qty: int
    review_score: int
    review_comment_title: Optional[str] = Field(default=None)
    review_comment_message: str
    review_creation_date: Optional[datetime] = Field(default=None)
    review_answer_timestamp: Optional[datetime] = Field(default=None)
    votes_before_estimate: Optional[int] = Field(default=None)
    votes_delayed: Optional[int] = Field(default=None)
    votes_low_quality: Optional[int] = Field(default=None)
    votes_return: Optional[int] = Field(default=None)
    votes_not_as_anounced: Optional[int] = Field(default=None)
    votes_partial_delivery: Optional[int] = Field(default=None)
    votes_other_delivery: Optional[int] = Field(default=None)
    votes_other_order: Optional[int] = Field(default=None)
    votes_satisfied: Optional[int] = Field(default=None)
    most_voted_subclass: Optional[str] = Field(default=None)
    most_voted_class: Optional[str] = Field(default=None)
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="allow")