import polars as pl


def process_item_data(item_data: pl.DataFrame) -> pl.DataFrame:
    item_data = item_data.drop([
        "title", "subtitle", "body", 
        "image_ids", "url", "category_str", 
        "sentiment_label", "ner_clusters", "entity_groups"
    ])

    item_data = item_data.with_columns(
        pl.col("last_modified_time").dt.epoch(time_unit="s").alias("last_modified_time"),
        pl.col("published_time").dt.epoch(time_unit="s").alias("published_time")
    )

    max_subcategories = item_data["subcategory"].map_elements(lambda s: len(s),return_dtype=int).max()
    subcategory_columns = [
        pl.col("subcategory").map_elements(lambda x: x[i] if i < len(x) else None, return_dtype=int).alias(f"subcategory_{i+1}")
        for i in range(max_subcategories)
    ]
    item_data = item_data.with_columns(subcategory_columns).drop("subcategory")

    item_data = item_data.with_columns(
        (pl.col("article_type").rank("dense") - 1).alias("article_type_ranked")
    )

    vector_length = len(item_data[0, "document_vector"]) 
    vector_columns = [
        pl.col("document_vector").list.get(i).alias(f"vector_{i+1}")
        for i in range(vector_length)
    ]
    item_data = item_data.with_columns(vector_columns).drop("document_vector")

    item_data = item_data.with_columns(
        pl.col("article_type").rank("dense")-1
    )

    max_topics = item_data["topics"].map_elements(lambda s: len(s),return_dtype=int).max()

    unique_topics = set(topic for row in item_data["topics"] for topic in row)
    topic_to_label = {topic: i for i, topic in enumerate(unique_topics)}

    item_data = item_data.with_columns(
        pl.col("topics").map_elements(lambda topics: [topic_to_label[topic] for topic in topics], return_dtype=list).alias("encoded_topics")
    )

    for i in range(max_topics):
        item_data = item_data.with_columns(
            pl.col("encoded_topics").map_elements(lambda x: x[i] if i < len(x) else None, return_dtype=int).alias(f"topic_{i+1}")
        )


    item_data = item_data.drop(["topics", "encoded_topics"])

    return item_data
