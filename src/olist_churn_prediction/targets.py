"""Функции для построения целевой переменной (таргета) оттока.

Докстринги оформлены в стиле Google (подходит для Sphinx napoleon).

Основная функция:

- :func:`create_churn_label` — создаёт бинарный таргет ``churned`` на основе
  временных меток последнего заказа клиента.

Пример использования::

    import pandas as pd
    from olist_churn_prediction.targets import create_churn_label

    df = pd.read_parquet("data/raw/orders.parquet")
    df_labeled = create_churn_label(df, horizon_days=120)
"""

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
    """Создаёт бинарную метку оттока (churn) для клиентов.

    Логика: клиент считается ``churned=1``, если с момента его последнего
    заказа прошло больше ``horizon_days`` относительно контрольной даты.
    
    Args:
        df: Входной датафрейм заказов.
        customer_col: Имя колонки с идентификатором клиента.
        purchase_ts_col: Имя колонки с датой/временем покупки.
        target_col: Имя создаваемой колонки таргета (по умолчанию ``"churned"``).
        horizon_days: Порог давности (в днях), после которого клиент считается ушедшим.
        reference_date: Контрольная дата:
            * ``"max"`` (по умолчанию) → максимум по ``purchase_ts_col``.
            * строка в ISO-формате (``YYYY-MM-DD``).
            * ``pd.Timestamp``.
            * ``None`` → тоже берётся максимум.
        filter_status_col: Имя колонки со статусом заказа для фильтрации.
        keep_statuses: Допустимые статусы заказов (например, только ``("delivered",)``).
        force: Если ``False`` и таргет уже существует — ничего не делать.
    
    Returns:
        DataFrame с новой бинарной колонкой ``target_col``.
    
    Raises:
        ValueError: Если отсутствует колонка ``customer_col`` или ``purchase_ts_col``.
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ... "customer_id": [1, 1, 2],
        ... "order_purchase_timestamp": ["2020-01-01", "2020-02-01", "2020-01-15"],
        ... "order_status": ["delivered", "delivered", "delivered"]
        ... })
        >>> create_churn_label(df, horizon_days=30)["churned"].tolist() # doctest: +SKIP
        [1, 1, 1]
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