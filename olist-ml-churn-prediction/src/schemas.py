from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class SellersSchema(BaseModel):
    order_id: str
    product_id: str
    seller_id: str
    seller_zip_code_prefix: int
    seller_city: str
    seller_state: str

class MeasuresSchema(BaseModel):
    product_id: str
    product_weight_g: float
    product_length_cm: float
    product_height_cm: float
    product_width_cm: float

class TranslationSchema(BaseModel):
    product_category_name: str
    product_category_name_english: str

class PaymentsSchema(BaseModel):
    order_id: str
    installments: int
    sequential: int
    payment_type: int
    value: int

class MainPublicSchema(BaseModel):
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
    customer_zip_code_prefix: int
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

class CustomersSchema(BaseModel):
    customer_id: str
    customer_unique_id: str

class GeolocationSchema(BaseModel):
    zip_code_prefix: int
    city: str
    state: str
    lat: float
    lng: float

class MainClassifiedSchema(BaseModel):
    id: str
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
    customer_zip_code_prefix: int
    product_category_name: str
    product_name_lenght: int
    product_description_lenght: int
    product_photos_qty: int
    review_score: int
    review_comment_title: str
    review_comment_message: str
    review_creation_date: Optional[datetime] = Field(default=None)
    review_answer_timestamp: Optional[datetime] = Field(default=None)
    votes_before_estimate: int
    votes_delayed: int
    votes_low_quality: int
    votes_return: int
    votes_not_as_anounced: int
    votes_partial_delivery: int
    votes_other_delivery: int
    votes_other_order: int
    votes_satisfied: int
    most_voted_subclass: str
    most_voted_class: str