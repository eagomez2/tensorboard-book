from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import pytest

from tensorboard_book import analysis as an


def test_guess_direction():
    assert an.guess_direction("val/loss") == "min"
    assert an.guess_direction("eval/wer") == "min"
    assert an.guess_direction("val/acc") == "max"
    assert an.guess_direction("top1_accuracy") == "max"


def test_filter_parsing_and_evaluation():
    frame = pl.DataFrame(
        {
            "optimizer.lr": [1e-2, 1e-3, 1e-4],
            "model": ["resnet50", "resnet18", "vit"],
            "val/acc": [0.8, 0.92, np.nan],
            "flag": [True, False, True],
        }
    )
    terms = an.parse_filter("optimizer.lr <= 1e-3 and model ~ RES")
    assert terms == [("optimizer.lr", "<=", "1e-3"), ("model", "~", "RES")]
    assert list(an.apply_filter(frame, terms)) == [False, True, False]
    assert list(an.apply_filter(frame, an.parse_filter("val/acc > 0.85"))) == [
        False,
        True,
        False,
    ]
    assert list(an.apply_filter(frame, an.parse_filter("flag == true"))) == [
        True,
        False,
        True,
    ]
    assert list(an.apply_filter(frame, an.parse_filter("model = 'vit'"))) == [
        False,
        False,
        True,
    ]


def test_filter_errors_are_helpful():
    frame = pl.DataFrame({"optimizer.lr": [1.0]})
    with pytest.raises(ValueError, match="optimizer.lr"):
        an.apply_filter(frame, an.parse_filter("optimizer.lrr < 1"))
    with pytest.raises(ValueError):
        an.parse_filter("just words")


def test_metric_table_uses_direction():
    metrics = pl.DataFrame(
        {
            "run_id": [1, 1, 2],
            "tag": ["val/loss", "val/acc", "val/loss"],
            "min": [0.1, 0.5, 0.2],
            "max": [0.9, 0.95, 0.8],
            "last": [0.15, 0.9, 0.25],
        }
    )
    table = an.metric_table(metrics, [1, 2], ["val/loss", "val/acc"], {})
    rows = {r["id"]: r for r in table.iter_rows(named=True)}
    assert table["id"].to_list() == [1, 2]
    assert rows[1]["val/loss"] == 0.1
    assert rows[1]["val/acc"] == 0.95
    assert rows[2]["val/acc"] is None
    last = an.metric_table(metrics, [1], ["val/acc"], {}, mode="last")
    assert last.row(0, named=True)["val/acc"] == 0.9


def test_value_at_step_picks_nearest():
    series = pl.DataFrame({"step": [0, 100, 200], "value": [1.0, 2.0, 3.0]})
    assert an.value_at_step(series, 140) == (2.0, 100.0)
    assert an.value_at_step(series, 160) == (3.0, 200.0)
    assert np.isnan(an.value_at_step(series.head(0), 5)[0])


def test_ema_matches_tensorboard_debiasing():
    values = np.array([1.0, 1.0, 1.0])
    assert np.allclose(an.ema_smooth(values, 0.9), values)
    assert np.array_equal(an.ema_smooth(values, 0), values)


def test_seed_and_hparam_keys():
    assert an.seed_key("aug_mixup_seed2", r"[_-]?seed[_-]?\d+") == "aug_mixup"
    a = an.hparam_key({"lr": 0.1, "seed": 1, "trainer.seed": 3})
    b = an.hparam_key({"lr": 0.1, "seed": 2, "trainer.seed": 4})
    assert a == b == "lr=0.1"


def test_daily_hours_split_at_midnight():
    tz = timezone.utc
    start = datetime(2026, 9, 1, 22, tzinfo=tz).timestamp()
    end = datetime(2026, 9, 2, 3, tzinfo=tz).timestamp()
    hours = an.daily_hours([[start, end]], tz)
    assert hours == {date(2026, 9, 1): 2.0, date(2026, 9, 2): 3.0}


def test_concurrency_counts_overlaps():
    conc = an.concurrency([[0, 10], [5, 15]])
    assert list(conc["running"]) == [1, 2, 1, 0]


def test_run_status():
    assert an.run_status(100, 100, 0, 100 + 60, 10) == "Active"
    assert an.run_status(100, 100, 0, 100 + 3600, 10) == "Stopped"
    assert an.run_status(100, 100, 1, 100, 10) == "Missing"


def test_formatting():
    assert an.format_duration(3 * 3600 + 12 * 60) == "3h 12m"
    assert an.format_duration(2 * 86400 + 4 * 3600) == "2d 4h"
    assert an.format_bytes(1536) == "1.5 KB"
    assert an.format_value(0.123456) == "0.1235"


def test_exports():
    df = pl.DataFrame({"run": ["a_b"], "acc": ["0.9 ± 0.01"]})
    md = an.to_markdown(df)
    assert md.splitlines()[0] == "| run | acc |"
    tex = an.to_latex(df, {(0, "acc")})
    assert r"a\_b" in tex and r"\textbf{0.9 $\pm$ 0.01}" in tex


def test_hparam_columns_keep_types():
    frame = an.hparam_frame(
        {
            1: {"lr": 0.1, "layers": 2, "model": "a", "flag": True},
            2: {"lr": 1, "model": "b", "mixed": 3},
            3: {"mixed": "x"},
        }
    )
    assert frame.schema["lr"] == pl.Float64
    assert frame.schema["layers"] == pl.Int64
    assert frame.schema["flag"] == pl.Boolean
    assert frame.schema["mixed"] == pl.String
    assert an.varying_columns(frame) == [
        "flag",
        "layers",
        "lr",
        "mixed",
        "model",
    ]


def test_best_position_ignores_missing():
    assert an.best_position([None, 0.3, float("nan"), 0.9], "max") == 3
    assert an.best_position([0.5, 0.2], "min") == 1
    assert an.best_position([None], "max") is None


def test_explorer_axis_numeric_log_and_categories():
    lr = an.explorer_axis([1e-4, 3e-4, 1e-3, 1e-2])
    assert lr["kind"] == "number" and lr["log"]
    assert lr["positions"][0] == pytest.approx(-4)
    assert lr["tickvals"] == [-4, -3, -2]

    bs = an.explorer_axis([32, 64, 64])
    assert bs["kind"] == "number" and not bs["log"]
    assert bs["positions"] == [32.0, 64.0, 64.0]

    opt = an.explorer_axis(["sgd", None, "adamw"])
    assert opt["kind"] == "category"
    assert opt["ticktext"] == ["adamw", "sgd", "–"]
    assert opt["positions"] == [1.0, 2.0, 0.0]

    gaps = an.explorer_axis([0.1, None])
    assert gaps["kind"] == "category"
    assert gaps["ticktext"] == ["0.1", "–"]


def test_perplexity_has_no_lower_is_better_guess():
    assert an.guess_direction("val/perplexity") == "max"
    assert an.guess_direction("val/loss") == "min"


def test_baseline_changes():
    assert an.change_from(0.92, 0.9) == pytest.approx(0.02)
    assert an.change_from(None, 0.9) is None
    assert an.change_from(0.9, float("nan")) is None
    assert an.format_change(0.02) == "+0.02"
    assert an.format_change(-0.5) == "−0.5"
    assert an.format_change(0.0) == "±0"
    assert an.format_change(None) == ""
    assert an.change_kind(0.02, "max") == "good"
    assert an.change_kind(0.02, "min") == "bad"
    assert an.change_kind(-0.1, "min") == "good"
    assert an.change_kind(0.0, "max") == ""
    tex = an.to_latex(pl.DataFrame({"acc": ["0.9  (−0.01)"]}), set())
    assert "$-$0.01" in tex


def test_overlay_default_pairs_tags_with_the_same_leaf():
    tags = ["val/acc", "train/loss", "val/loss", "train/lr", "lr"]
    assert an.overlay_default(tags) == ["train/loss", "val/loss"]
    assert an.overlay_default(["val/acc", "val/loss"]) == []
