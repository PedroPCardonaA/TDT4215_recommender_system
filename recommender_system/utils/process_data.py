import polars as pl

def process_item_data(item_data: pl.DataFrame) -> pl.DataFrame:
    drop_cols = [
        "title", "subtitle", "body",
        "image_ids", "url", "category_str",
        "sentiment_label", "ner_clusters", "entity_groups"
    ]
    existing_drop_cols = [c for c in drop_cols if c in item_data.columns]
    item_data = item_data.drop(existing_drop_cols)

    for time_col in ["last_modified_time", "published_time"]:
        if time_col in item_data.columns:
            item_data = item_data.with_columns(
                pl.col(time_col).dt.epoch(time_unit="s").alias(time_col)
            )

    if "document_vector" in item_data.columns:
        vector_length = len(item_data[0, "document_vector"])
        vector_cols = [
            pl.col("document_vector").list.get(i).alias(f"vector_{i+1}")
            for i in range(vector_length)
        ]
        item_data = item_data.with_columns(vector_cols).drop("document_vector")

    if "article_type" in item_data.columns:
        item_data = item_data.with_columns(
            (pl.col("article_type").rank("dense") - 1).alias("article_type_ranked")
        ).drop("article_type")

    if "subcategory" in item_data.columns:
        item_data = item_data.with_row_count("row_nr_subcat")
        exploded = item_data.explode("subcategory")
        subcat_dummies = exploded.to_dummies(columns=["subcategory"])
        subcat_dummies_agg = subcat_dummies.group_by("row_nr_subcat").agg([
            pl.col(col).max().alias(col)
            for col in subcat_dummies.columns
            if col.startswith("subcategory_")
        ])
        item_data = item_data.join(subcat_dummies_agg, on="row_nr_subcat", how="left")
        item_data = item_data.drop(["row_nr_subcat", "subcategory"])

    if "topics" in item_data.columns:
        item_data = item_data.with_row_count("row_nr_topics")
        exploded = item_data.explode("topics")

        topics_dummies = exploded.to_dummies(columns=["topics"])
        topics_dummies_agg = topics_dummies.group_by("row_nr_topics").agg([
            pl.col(col).max().alias(col)
            for col in topics_dummies.columns
            if col.startswith("topics_")
        ])

        item_data = item_data.join(topics_dummies_agg, on="row_nr_topics", how="left")
        item_data = item_data.drop(["row_nr_topics", "topics"])

    skip_standardize = {"article_id", "category"}
    numeric_types = (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    numeric_cols = [
        c for c in item_data.columns
        if item_data.schema[c] in numeric_types and c not in skip_standardize
    ]
    if numeric_cols:
        item_data = item_data.with_columns([
            ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
            for c in numeric_cols
        ])

    return item_data


def explode_user_interactions(user_data: pl.DataFrame) -> pl.DataFrame:
    # Explode all these list columns simultaneously into separate rows
    df_long = user_data.explode([
        "impression_time_fixed",
        "scroll_percentage_fixed",
        "article_id_fixed",
        "read_time_fixed"
    ])

    # Rename columns for clarity
    df_long = df_long.rename({
        "impression_time_fixed": "impression_time",
        "scroll_percentage_fixed": "scroll_percentage",
        "article_id_fixed": "article_id",
        "read_time_fixed": "read_time"
    })
    return df_long
