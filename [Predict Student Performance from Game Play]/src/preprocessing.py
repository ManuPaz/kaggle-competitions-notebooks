import polars as pl
from src.constants import (
    CATS,
    NUMS,
)
from src.constants import (
    EVENT_NAME_FEATURE as event_name_feature,
)
from src.constants import (
    FQID_LISTS as fqid_lists,
)
from src.constants import (
    NAME_FEATURE as name_feature,
)
from src.constants import (
    ROOM_LISTS as room_lists,
)
from src.constants import (
    TEXT_LISTS as text_lists,
)
from tsfresh.feature_extraction import feature_calculators


def feature_engineer_pl(x, grp, use_extra, feature_suffix, texts):
    aggs = [
        pl.col("index").count().alias(f"session_number_{feature_suffix}"),
        *[pl.col(c).drop_nulls().n_unique().alias(f"{c}_unique_{feature_suffix}") for c in CATS],
        *[pl.col(c).quantile(0.1, "nearest").alias(f"{c}_quantile1_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.2, "nearest").alias(f"{c}_quantile2_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.4, "nearest").alias(f"{c}_quantile4_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.6, "nearest").alias(f"{c}_quantile6_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.8, "nearest").alias(f"{c}_quantile8_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.9, "nearest").alias(f"{c}_quantile9_{feature_suffix}") for c in NUMS],
        *[pl.col(c).mean().alias(f"{c}_mean_{feature_suffix}") for c in NUMS],
        *[pl.col(c).std().alias(f"{c}_std_{feature_suffix}") for c in NUMS],
        *[pl.col(c).min().alias(f"{c}_min_{feature_suffix}") for c in NUMS],
        *[pl.col(c).max().alias(f"{c}_max_{feature_suffix}") for c in NUMS],
        *[pl.col(c).sum().alias(f"{c}_sum_{feature_suffix}") for c in NUMS],
        *[
            pl.col(c).apply(lambda x: feature_calculators.kurtosis(x)).alias(f"{c}_kurtosis_{feature_suffix}")
            for c in NUMS
        ],
        *[
            pl.col(c).apply(lambda x: feature_calculators.skewness(x)).alias(f"{c}_skewness_{feature_suffix}")
            for c in NUMS
        ],
        *[
            pl.col(c)
            .apply(lambda x: feature_calculators.first_location_of_maximum(x))
            .alias(f"{c}_first_position_maximun_{feature_suffix}")
            for c in NUMS
        ],
        *[
            pl.col(c)
            .apply(lambda x: feature_calculators.last_location_of_maximum(x))
            .alias(f"{c}_last_position_maximun_{feature_suffix}")
            for c in NUMS
        ],
        *[
            pl.col("event_name")
            .filter(pl.col("event_name") == c)
            .count()
            .alias(f"{c}_event_name_counts{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("event_name") == c)
            .quantile(0.1, "nearest")
            .alias(f"{c}_ET_quantile1_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("event_name") == c)
            .quantile(0.2, "nearest")
            .alias(f"{c}_ET_quantile2_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("event_name") == c)
            .quantile(0.4, "nearest")
            .alias(f"{c}_ET_quantile4_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("event_name") == c)
            .quantile(0.6, "nearest")
            .alias(f"{c}_ET_quantile6_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("event_name") == c)
            .quantile(0.8, "nearest")
            .alias(f"{c}_ET_quantile8_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("event_name") == c)
            .quantile(0.9, "nearest")
            .alias(f"{c}_ET_quantile9_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).std().alias(f"{c}_ET_std_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).max().alias(f"{c}_ET_max_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).min().alias(f"{c}_ET_min_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("event_name") == c)
            .apply(lambda x: feature_calculators.kurtosis(x))
            .alias(f" {c}_ET_kurtosis_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("event_name") == c)
            .apply(lambda x: feature_calculators.skewness(x))
            .alias(f"{c}_ET_skewness_{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("name").filter(pl.col("name") == c).count().alias(f"{c}_name_counts{feature_suffix}")
            for c in name_feature
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("name") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}")
            for c in name_feature
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("name") == c).max().alias(f"{c}_ET_max_{feature_suffix}")
            for c in name_feature
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("name") == c).min().alias(f"{c}_ET_min_{feature_suffix}")
            for c in name_feature
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("name") == c).std().alias(f"{c}_ET_std_{feature_suffix}")
            for c in name_feature
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("name") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}")
            for c in name_feature
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("name") == c)
            .apply(lambda x: feature_calculators.kurtosis(x))
            .alias(f" {c}_ET_kurtosis_{feature_suffix}")
            for c in name_feature
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("name") == c)
            .apply(lambda x: feature_calculators.skewness(x))
            .alias(f"{c}_ET_skewness_{feature_suffix}")
            for c in name_feature
        ],
        *[
            pl.col("room_fqid").filter(pl.col("room_fqid") == c).count().alias(f"{c}_room_fqid_counts{feature_suffix}")
            for c in room_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}")
            for c in room_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}")
            for c in room_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}")
            for c in room_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).min().alias(f"{c}_ET_min_{feature_suffix}")
            for c in room_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}")
            for c in room_lists
        ],
        *[
            pl.col("fqid").filter(pl.col("fqid") == c).count().alias(f"{c}_fqid_counts{feature_suffix}")
            for c in fqid_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}")
            for c in fqid_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}")
            for c in fqid_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}")
            for c in fqid_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).min().alias(f"{c}_ET_min_{feature_suffix}")
            for c in fqid_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}")
            for c in fqid_lists
        ],
        *[
            pl.col("text_fqid").filter(pl.col("text_fqid") == c).count().alias(f"{c}_text_fqid_counts{feature_suffix}")
            for c in text_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}")
            for c in text_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}")
            for c in text_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}")
            for c in text_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).min().alias(f"{c}_ET_min_{feature_suffix}")
            for c in text_lists
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}")
            for c in text_lists
        ],
        *[
            pl.col("location_x_diff").filter(pl.col("event_name") == c).mean().alias(f"{c}_ET_mean_x{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("location_x_diff").filter(pl.col("event_name") == c).std().alias(f"{c}_ET_std_x{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("location_x_diff").filter(pl.col("event_name") == c).max().alias(f"{c}_ET_max_x{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("location_x_diff").filter(pl.col("event_name") == c).min().alias(f"{c}_ET_min_x{feature_suffix}")
            for c in event_name_feature
        ],
        *[
            pl.col("text_fqid").filter(pl.col("text") == c).count().alias(f"{c}_text_counts{feature_suffix}")
            for c in texts
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("text") == c).std().alias(f"{c}_text_ET_std_{feature_suffix}")
            for c in texts
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("text") == c).mean().alias(f"{c}_text_ET_mean_{feature_suffix}")
            for c in texts
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("text") == c).max().alias(f"{c}_text_ET_max_{feature_suffix}")
            for c in texts
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("text") == c).min().alias(f"{c}_text_ET_min_{feature_suffix}")
            for c in texts
        ],
        *[
            pl.col("elapsed_time_diff").filter(pl.col("text") == c).sum().alias(f"{c}_text_ET_sum_{feature_suffix}")
            for c in texts
        ],
        *[pl.col("text").filter(pl.col("text").is_null()).count().alias(f"null_text_counts{feature_suffix}")],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("text").is_null())
            .std()
            .alias(f"null_text_ET_std_{feature_suffix}")
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("text").is_null())
            .mean()
            .alias(f"null_text_ET_mean_{feature_suffix}")
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("text").is_null())
            .max()
            .alias(f"null_text_ET_max_{feature_suffix}")
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("text").is_null())
            .min()
            .alias(f"null_text_ET_min_{feature_suffix}")
        ],
        *[
            pl.col("elapsed_time_diff")
            .filter(pl.col("text").is_null())
            .sum()
            .alias(f"null_text_ET_sum_{feature_suffix}")
        ],
    ]

    df = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")

    if use_extra:
        if grp == "5-12":
            aggs = [
                pl.col("elapsed_time")
                .filter((pl.col("text") == "Here's the log book.") | (pl.col("fqid") == "logbook.page.bingo"))
                .apply(lambda s: s.max() - s.min())
                .alias("logbook_bingo_duration"),
                pl.col("index")
                .filter((pl.col("text") == "Here's the log book.") | (pl.col("fqid") == "logbook.page.bingo"))
                .apply(lambda s: s.max() - s.min())
                .alias("logbook_bingo_indexCount"),
                pl.col("elapsed_time")
                .filter(
                    ((pl.col("event_name") == "navigate_click") & (pl.col("fqid") == "reader"))
                    | (pl.col("fqid") == "reader.paper2.bingo")
                )
                .apply(lambda s: s.max() - s.min())
                .alias("reader_bingo_duration"),
                pl.col("index")
                .filter(
                    ((pl.col("event_name") == "navigate_click") & (pl.col("fqid") == "reader"))
                    | (pl.col("fqid") == "reader.paper2.bingo")
                )
                .apply(lambda s: s.max() - s.min())
                .alias("reader_bingo_indexCount"),
                pl.col("elapsed_time")
                .filter(
                    ((pl.col("event_name") == "navigate_click") & (pl.col("fqid") == "journals"))
                    | (pl.col("fqid") == "journals.pic_2.bingo")
                )
                .apply(lambda s: s.max() - s.min())
                .alias("journals_bingo_duration"),
                pl.col("index")
                .filter(
                    ((pl.col("event_name") == "navigate_click") & (pl.col("fqid") == "journals"))
                    | (pl.col("fqid") == "journals.pic_2.bingo")
                )
                .apply(lambda s: s.max() - s.min())
                .alias("journals_bingo_indexCount"),
            ]
            tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
            df = df.join(tmp, on="session_id", how="left")

        if grp == "13-22":
            aggs = [
                pl.col("elapsed_time")
                .filter(
                    ((pl.col("event_name") == "navigate_click") & (pl.col("fqid") == "reader_flag"))
                    | (pl.col("fqid") == "tunic.library.microfiche.reader_flag.paper2.bingo")
                )
                .apply(lambda s: s.max() - s.min() if s.len() > 0 else 0)
                .alias("reader_flag_duration"),
                pl.col("index")
                .filter(
                    ((pl.col("event_name") == "navigate_click") & (pl.col("fqid") == "reader_flag"))
                    | (pl.col("fqid") == "tunic.library.microfiche.reader_flag.paper2.bingo")
                )
                .apply(lambda s: s.max() - s.min() if s.len() > 0 else 0)
                .alias("reader_flag_indexCount"),
                pl.col("elapsed_time")
                .filter(
                    ((pl.col("event_name") == "navigate_click") & (pl.col("fqid") == "journals_flag"))
                    | (pl.col("fqid") == "journals_flag.pic_0.bingo")
                )
                .apply(lambda s: s.max() - s.min() if s.len() > 0 else 0)
                .alias("journalsFlag_bingo_duration"),
                pl.col("index")
                .filter(
                    ((pl.col("event_name") == "navigate_click") & (pl.col("fqid") == "journals_flag"))
                    | (pl.col("fqid") == "journals_flag.pic_0.bingo")
                )
                .apply(lambda s: s.max() - s.min() if s.len() > 0 else 0)
                .alias("journalsFlag_bingo_indexCount"),
            ]
            tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
            df = df.join(tmp, on="session_id", how="left")

    return df.to_pandas()
