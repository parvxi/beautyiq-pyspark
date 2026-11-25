"""
Step 1 of the pipeline: convert the raw Amazon Beauty JSONL files to Parquet.

Why: JSON is row-oriented text, so Spark must parse every byte of every line.
Parquet is columnar and compressed, which lets later queries skip columns
(column pruning) and skip whole files/row groups (filter + partition pushdown).

Run once:  .venv/bin/python scripts/01_bronze_ingest.py
"""
import os
import time

from pyspark.sql import SparkSession, functions as F, types as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("RAW_DIR", os.path.join(ROOT, "data", "raw"))
BRONZE = os.environ.get("BRONZE_DIR", os.path.join(ROOT, "data", "bronze"))

spark = (
    SparkSession.builder.appName("beauty-bronze-ingest")
    .master("local[*]")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Explicit schemas: faster than inference (no extra pass over 11 GB) and
# lets us drop heavy nested columns (image URLs, videos) at read time.
review_schema = T.StructType([
    T.StructField("rating", T.DoubleType()),
    T.StructField("title", T.StringType()),
    T.StructField("text", T.StringType()),
    T.StructField("asin", T.StringType()),
    T.StructField("parent_asin", T.StringType()),
    T.StructField("user_id", T.StringType()),
    T.StructField("timestamp", T.LongType()),          # epoch milliseconds
    T.StructField("helpful_vote", T.IntegerType()),
    T.StructField("verified_purchase", T.BooleanType()),
])

meta_schema = T.StructType([
    T.StructField("main_category", T.StringType()),
    T.StructField("title", T.StringType()),
    T.StructField("average_rating", T.DoubleType()),
    T.StructField("rating_number", T.LongType()),
    T.StructField("price", T.StringType()),            # mixed number/string in raw data
    T.StructField("store", T.StringType()),
    T.StructField("categories", T.ArrayType(T.StringType())),
    T.StructField("details", T.StringType()),          # free-form object kept as JSON text
    T.StructField("parent_asin", T.StringType()),
])

t0 = time.time()
reviews = (
    spark.read.schema(review_schema)
    .json(os.path.join(RAW, "Beauty_and_Personal_Care.jsonl"))
    .withColumn("review_ts", (F.col("timestamp") / 1000).cast("timestamp"))
    .withColumn("review_year", F.year("review_ts"))
    .drop("timestamp")
)
(reviews.repartition("review_year")
    .write.mode("overwrite")
    .partitionBy("review_year")
    .parquet(os.path.join(BRONZE, "reviews")))
print(f"reviews -> parquet in {time.time() - t0:.1f}s")

t0 = time.time()
products = (
    spark.read.schema(meta_schema)
    .json(os.path.join(RAW, "meta_Beauty_and_Personal_Care.jsonl"))
    .withColumn("price", F.regexp_replace("price", "[$,]", "").cast("double"))
    .withColumn("brand", F.coalesce(F.get_json_object("details", "$.Brand"), F.col("store")))
    .withColumn("sub_category", F.element_at("categories", 2))
    .withColumn("leaf_category", F.element_at("categories", -1))
    .select("parent_asin", "title", "main_category", "sub_category", "leaf_category",
            "brand", "store", "price", "average_rating", "rating_number")
)
products.coalesce(8).write.mode("overwrite").parquet(os.path.join(BRONZE, "products"))
print(f"products -> parquet in {time.time() - t0:.1f}s")

spark.stop()
