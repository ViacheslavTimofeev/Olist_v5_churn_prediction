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
    order_id: pd.Series[str]
    product_id: pd.Series[str]
    seller_id: pd.Series[str]
    seller_zip_code_prefix: pd.Series[str]
    seller_city: pd.Series[str]
    seller_state: pd.Series[str]
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")          # лишние или пропущенные колонки → ошибка

class MeasuresSchemaRaw(_NaN2NoneMixin):
    product_id: pd.Series[str]
    product_weight_g: pd.Series[float]
    product_length_cm: pd.Series[float]
    product_height_cm: pd.Series[float]
    product_width_cm: pd.Series[float]
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class TranslationSchemaRaw(_NaN2NoneMixin):
    product_category_name: pd.Series[str]
    product_category_name_english: pd.Series[str]
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class PaymentsSchemaRaw(_NaN2NoneMixin):
    order_id: pd.Series[str]
    installments: pd.Series[int]
    sequential: pd.Series[int]
    payment_type: pd.Series[str]
    value: pd.Series[float]
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class MainPublicSchemaRaw(_NaN2NoneMixin):
    order_id: pd.Series[str]
    order_status: pd.Series[str]
    order_products_value: pd.Series[float]
    order_freight_value: pd.Series[float]
    order_items_qty: pd.Series[int]
    order_sellers_qty: pd.Series[int]
    order_purchase_timestamp: pd.Series[datetime] | None
    order_aproved_at: pd.Series[datetime] | None
    order_estimated_delivery_date: pd.Series[datetime] | None
    order_delivered_customer_date: pd.Series[datetime] | None
    customer_id: pd.Series[str]
    customer_city: pd.Series[str]
    customer_state: pd.Series[str]
    customer_zip_code_prefix: pd.Series[str]
    product_category_name: pd.Series[str]
    product_name_lenght: pd.Series[int]
    product_description_lenght: pd.Series[int]
    product_photos_qty: pd.Series[int]
    product_id: pd.Series[str]
    review_id: pd.Series[str]
    review_score: pd.Series[int]
    review_comment_title: pd.Series[str]
    review_comment_message: pd.Series[str]
    review_creation_date: pd.Series[datetime] | None
    review_answer_timestamp: pd.Series[datetime] | None
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class CustomersSchemaRaw(_NaN2NoneMixin):
    customer_id: pd.Series[str]
    customer_unique_id: pd.Series[str]
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class GeolocationSchemaRaw(_NaN2NoneMixin):
    zip_code_prefix: pd.Series[str]
    city: pd.Series[str]
    state: pd.Series[str]
    lat: pd.Series[float]
    lng: pd.Series[float]
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")
    
class MainClassifiedSchemaRaw(_NaN2NoneMixin):
    id: pd.Series[int]
    order_status: pd.Series[str]
    order_products_value: pd.Series[float]
    order_freight_value: pd.Series[float]
    order_items_qty: pd.Series[int]
    order_sellers_qty: pd.Series[int]
    order_purchase_timestamp: pd.Series[datetime] | None
    order_aproved_at: pd.Series[datetime] | None
    order_estimated_delivery_date: pd.Series[datetime] | None
    order_delivered_customer_date: pd.Series[datetime] | None
    customer_city: pd.Series[str]
    customer_state: pd.Series[str]
    customer_zip_code_prefix: pd.Series[str]
    product_category_name: pd.Series[str]
    product_name_lenght: pd.Series[int]
    product_description_lenght: pd.Series[int]
    product_photos_qty: pd.Series[int]
    review_score: pd.Series[int]
    review_comment_title: pd.Series[datetime] | None
    review_comment_message: pd.Series[str]
    review_creation_date: pd.Series[datetime] | None
    review_answer_timestamp: pd.Series[datetime] | None
    votes_before_estimate: Optional[pd.Series[int]] = Field(default=None)
    votes_delayed: Optional[pd.Series[int]] = Field(default=None)
    votes_low_quality: Optional[pd.Series[int]] = Field(default=None)
    votes_return: Optional[pd.Series[int]] = Field(default=None)
    votes_not_as_anounced: Optional[pd.Series[int]] = Field(default=None)
    votes_partial_delivery: Optional[pd.Series[int]] = Field(default=None)
    votes_other_delivery: Optional[pd.Series[int]] = Field(default=None)
    votes_other_order: Optional[pd.Series[int]] = Field(default=None)
    votes_satisfied: Optional[pd.Series[int]] = Field(default=None)
    most_voted_subclass: Optional[pd.Series[int]] = Field(default=None)
    most_voted_class: Optional[pd.Series[int]] = Field(default=None)
    
    model_config = ConfigDict(
        dataframe_mode="series",
        extra="forbid")