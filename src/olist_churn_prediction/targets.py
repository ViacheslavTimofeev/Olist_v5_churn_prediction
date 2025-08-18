import pandas as pd

def create_churn_label(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    purchase_ts_col: str = "order_purchase_timestamp",
    target_col: str = "churned",
    horizon_days: int = 120,
    reference_date: pd.Timestamp | str | None = "max",  # "max" | ISO-строка | pd.Timestamp | None
    filter_status_col: str | None = "order_status",
    keep_statuses: tuple[str, ...] | None = ("delivered",),
    force: bool = False,
) -> pd.DataFrame:
    """
    Возвращает копию df с колонкой target_col (0/1).
    Идемпотентна: если target уже есть и force=False — не пересчитывает.
    """

    if target_col in df.columns and not force:
        return df

    _df = df.copy()

    # (optionally) фильтрация статусов, чтобы не ловить утечку по недоставленным заказам
    if filter_status_col and keep_statuses is not None and filter_status_col in _df.columns:
        _df = _df[_df[filter_status_col].isin(keep_statuses)]

    # гарантируем datetime без tz
    _df[purchase_ts_col] = pd.to_datetime(_df[purchase_ts_col], utc=False, errors="coerce")

    # reference_date
    if reference_date is None or reference_date == "max":
        ref = _df[purchase_ts_col].max()
    else:
        ref = pd.to_datetime(reference_date, utc=False)

    # last order per customer
    last_orders = (_df
        .groupby(customer_col, as_index=False)[purchase_ts_col]
        .max()
        .rename(columns={purchase_ts_col: "_last_purchase"}))

    last_orders["_days_since_last"] = (ref - last_orders["_last_purchase"]).dt.days
    last_orders[target_col] = (last_orders["_days_since_last"] > horizon_days).astype("int8")

    out = df.merge(last_orders[[customer_col, target_col]], on=customer_col, how="left", validate="m:1")
    return out
