# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    ValueType,
    BigQuerySource,
    Project,
    PushSource,
    RequestSource,
)

from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, Json, Map, String, Struct
from config.paths_config import *
from utils.common_functions import read_yaml

CONFIG = read_yaml(f"../../{CONFIG_PATH}")
config = CONFIG["data_ingestion"]

# Define a project for the feature repo
project = Project(name="feast_repo", description="A project for iris classification")

# Define an entity for the iris. You can think of an entity as a primary key used to
# fetch features.
iris_entity = Entity(name="iris", join_keys=["iris_id"],value_type=ValueType.STRING)

# Read data from BigQuery. See Feast documentation
# for more info.


iris_stats_source = BigQuerySource(
    name="iris_stats_source",
    query=f"""
        SELECT
            * EXCEPT(event_timestamp),
            TIMESTAMP(event_timestamp) AS event_ts
        FROM `{config['project_id']}.{config['bq_dataset']}.{config['bq_table']}`
    """,
    timestamp_field="event_ts",
)


# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
iris_stats_fv = FeatureView(
    name="iris_data",
    entities=[iris_entity],
    ttl=timedelta(weeks=52),
    schema=[
        Field(name="sepal_length", dtype=Float64),
        Field(name="sepal_width", dtype=Float64),
        Field(name="petal_length", dtype=Float64),
        Field(name="petal_width", dtype=Float64),
    ],
    online=True,
    source=iris_stats_source,
    # Tags are user defined key/value pairs that are attached to each
    # feature view
    tags={"team": "iris_classification"},
    enable_validation=True,
)



iris_stats_v1 = FeatureService(
    name="iris_stats_v1",
    features=[iris_stats_fv],
    logging_config=LoggingConfig(
        destination=FileLoggingDestination(path="data")
    ),
)


# Defines a way to push data (to be available offline, online or both) into Feast.
driver_stats_push_source = PushSource(
    name="iris_stats_push_source",
    batch_source=iris_stats_source,
)

