"""Generate a realistic folder of fake TensorBoard runs for trying the app.

The demo covers the cases the app handles: a learning-rate sweep with a
diverged run and a resumed run, a seed ablation, Keras-style ``train/`` and
``validation/`` subfolders written as TF2 tensor summaries, a run that is still
"active", hyperparameters from events and config files, text summaries, and
artifacts such as checkpoints, images, logs and CSVs.
"""

from __future__ import annotations

import os
import struct
import time
import zlib
from pathlib import Path

import numpy as np

from tensorboard_book import db

DAY = 86400.0


def _png(width: int, height: int, seed: int) -> bytes:
    """Encode a small colorful RGB image as PNG without extra dependencies."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = rng.integers(0, 255, 3)
    img = np.stack(
        [
            (base[0] + x * 3) % 256,
            (base[1] + y * 3) % 256,
            (base[2] + (x + y) * 2) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    raw = b"".join(b"\x00" + img[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _write_events(logdir: Path, events: list) -> None:
    """Write Event protos to a new event file dated like the simulated run.

    TensorBoard's own writer stamps the file with the current time, so the
    records are written directly with its ``RecordWriter`` instead.
    """
    from tensorboard.summary.writer.record_writer import RecordWriter

    logdir.mkdir(parents=True, exist_ok=True)
    first = min(e.wall_time for e in events)
    last = max(e.wall_time for e in events)
    path = (
        logdir / f"events.out.tfevents.{int(first)}.demo-host.{os.getpid()}.0"
    )
    with open(path, "wb") as fh:
        writer = RecordWriter(fh)
        for event in events:
            writer.write(event.SerializeToString())
    os.utime(path, (last, last))


def _curve(
    steps: np.ndarray, final: float, tau: float, noise: float, rng
) -> np.ndarray:
    return final * (1 - np.exp(-steps / tau)) + rng.normal(
        0, noise, len(steps)
    )


def _training_events(
    start: float,
    total_steps: int,
    final_acc: float,
    rng,
    steps_per_sec: float = 4.0,
    pause_at: int | None = None,
    pause_hours: float = 0.0,
    stop_at: int | None = None,
    nan_after: int | None = None,
    hparams: dict | None = None,
    lr: float = 1e-3,
) -> list:
    """Build the events of one simulated PyTorch-style training run."""
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.plugins.hparams import summary_v2

    def event(step, wall, **scalars):
        values = [
            summary_pb2.Summary.Value(tag=k, simple_value=v)
            for k, v in scalars.items()
        ]
        return event_pb2.Event(
            wall_time=wall,
            step=int(step),
            summary=summary_pb2.Summary(value=values),
        )

    def wall_of(step):
        wall = start + step / steps_per_sec
        return (
            wall + pause_hours * 3600
            if pause_at is not None and step > pause_at
            else wall
        )

    events = [event_pb2.Event(wall_time=start, file_version="brain.Event:2")]
    if hparams:
        summary = summary_v2.hparams_pb(hparams, start_time_secs=int(start))
        events.append(event_pb2.Event(wall_time=start, summary=summary))
    last_step = stop_at or total_steps
    train_steps = np.arange(0, last_step + 1, 500)
    loss = 2.3 * np.exp(-train_steps / (total_steps / 4)) + 0.15 * (
        1 - final_acc
    )
    loss += rng.normal(0, 0.03, len(train_steps))
    for step, value in zip(train_steps, loss):
        if nan_after is not None and step > nan_after:
            value = float("nan")
        events.append(
            event(
                step,
                wall_of(step),
                **{"train/loss": float(value), "train/lr": lr},
            )
        )
    val_steps = np.arange(2500, last_step + 1, 2500)
    acc = np.clip(
        _curve(val_steps, final_acc, total_steps / 5, 0.004, rng), 0, 1
    )
    for step, value in zip(val_steps, acc):
        if nan_after is not None and step > nan_after:
            events.append(
                event(step, wall_of(step), **{"val/loss": float("nan")})
            )
            continue
        events.append(
            event(
                step,
                wall_of(step),
                **{
                    "val/acc": float(value),
                    "val/loss": float(1.2 * (1 - value) + 0.05),
                },
            )
        )
    if last_step == total_steps and nan_after is None:
        test = float(np.clip(acc[-1] - abs(rng.normal(0.01, 0.004)), 0, 1))
        events.append(
            event(last_step, wall_of(last_step) + 30, **{"test/acc": test})
        )
    return events


def _keras_events(start: float, epochs: int, rng, split: str) -> list:
    """Build TF2-style tensor scalar events, like Keras' callback."""
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.util import tensor_util

    events = [event_pb2.Event(wall_time=start, file_version="brain.Event:2")]
    for epoch in range(epochs):
        values = []
        for tag, value in [
            (
                "epoch_loss",
                1.8 * np.exp(-epoch / 6) + 0.2 + rng.normal(0, 0.02),
            ),
            (
                "epoch_accuracy",
                0.9 * (1 - np.exp(-epoch / 5)) + rng.normal(0, 0.005),
            ),
        ]:
            metadata = None
            if epoch == 0:  # TF2 only writes plugin metadata once per tag.
                metadata = summary_pb2.SummaryMetadata(
                    plugin_data=summary_pb2.SummaryMetadata.PluginData(
                        plugin_name="scalars"
                    )
                )
            values.append(
                summary_pb2.Summary.Value(
                    tag=tag,
                    tensor=tensor_util.make_tensor_proto(np.float32(value)),
                    metadata=metadata,
                )
            )
        events.append(
            event_pb2.Event(
                wall_time=start
                + epoch * 600
                + (5 if split == "validation" else 0),
                step=epoch,
                summary=summary_pb2.Summary(value=values),
            )
        )
    return events


def _artifacts(
    run: Path, seed: int, steps: list[int], config: str | None
) -> None:
    """Write checkpoints, sample images, a log and a CSV into a run folder."""
    rng = np.random.default_rng(seed)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    for step in steps:
        (run / "checkpoints" / f"step_{step}.pt").write_bytes(
            rng.bytes(64_000)
        )
    (run / "samples").mkdir(exist_ok=True)
    for i in range(3):
        (run / "samples" / f"sample_{i}.png").write_bytes(
            _png(96, 64, seed * 10 + i)
        )
    lines = [
        f"[step {s}] loss={2.3 * np.exp(-s / 12000):.4f}"
        for s in range(0, 20001, 2000)
    ]
    (run / "train.log").write_text("\n".join(lines) + "\n")
    rows = ["id,label,pred,confidence"] + [
        f"{i},{i % 10},{(i * 7) % 10},{rng.uniform(0.3, 1):.3f}"
        for i in range(50)
    ]
    (run / "preds_val.csv").write_text("\n".join(rows) + "\n")
    if config:
        (run / "config.yaml").write_text(config)


def make_demo(dest: str | Path) -> Path:
    """Create the demo runs folder.

    Args:
        dest: Folder to create (must not exist or be empty).

    Returns:
        The demo folder.

    Raises:
        FileExistsError: If ``dest`` exists and is not empty.
    """
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.util import tensor_util

    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    now = time.time()
    rng = np.random.default_rng(0)

    sweep = [
        ("resnet_lr1e-2", 1e-2, 0.62, {"nan_after": 12000, "stop_at": 12500}),
        (
            "resnet_lr3e-3",
            3e-3,
            0.88,
            {"pause_at": 16000, "pause_hours": 12, "stop_at": 31000},
        ),
        ("resnet_lr1e-3", 1e-3, 0.929, {}),
        ("resnet_lr3e-4", 3e-4, 0.941, {}),
        ("resnet_lr1e-4", 1e-4, 0.915, {}),
    ]
    for i, (name, lr, acc, extra) in enumerate(sweep):
        start = now - 20 * DAY + i * 0.6 * DAY
        hp = {
            "model": "resnet50",
            "optimizer": "adamw",
            "lr": lr,
            "batch_size": 256,
        }
        events = _training_events(
            start, 50000, acc, rng, hparams=hp, lr=lr, **extra
        )
        _write_events(dest / name, events)
        config = (
            f"model: resnet50\noptimizer:\n  name: adamw\n  lr: {lr}\n"
            "  weight_decay: 0.05\nmax_steps: 50000\nbatch_size: 256\n"
            "seed: 1\n"
        )
        _artifacts(dest / name, i, [10000, 20000, 30000], config)

    for j, aug in enumerate(["none", "mixup"]):
        for seed in (1, 2, 3):
            name = f"aug_{aug}_seed{seed}"
            start = now - 11 * DAY + j * 1.5 * DAY + seed * 0.2 * DAY
            acc = (0.905 if aug == "none" else 0.921) + rng.normal(0, 0.003)
            events = _training_events(
                start, 20000, acc, rng, steps_per_sec=3.0, lr=5e-4
            )
            _write_events(dest / name, events)
            config = (
                f"model: resnet18\naugmentation: {aug}\nseed: {seed}\n"
                "optimizer:\n  name: sgd\n  lr: 0.0005\n  momentum: 0.9\n"
            )
            _artifacts(dest / name, 100 + j * 10 + seed, [20000], config)

    keras = dest / "keras_baseline"
    start = now - 6 * DAY
    for split in ("train", "validation"):
        _write_events(keras / split, _keras_events(start, 30, rng, split))
    (keras / "hparams.yaml").write_text(
        "model: mobilenet_v2\nlr: 0.001\n"
        "optimizer: !!python/object:torch.optim.Adam {betas: [0.9, 0.999]}\n"
    )

    for seed in (1, 2):
        name = f"no_dropout_s{seed}"
        start = now - 3.5 * 3600 + seed * 1800
        steps_done = int((now - 60 - start) * 3.0) // 500 * 500
        events = _training_events(
            start,
            60000,
            0.93,
            rng,
            steps_per_sec=3.0,
            stop_at=steps_done,
            lr=3e-4,
        )
        note = summary_pb2.Summary.Value(
            tag="notes/text_summary",
            tensor=tensor_util.make_tensor_proto(
                np.array([b"Dropout disabled to test **regularization**."])
            ),
            metadata=summary_pb2.SummaryMetadata(
                plugin_data=summary_pb2.SummaryMetadata.PluginData(
                    plugin_name="text"
                )
            ),
        )
        events.insert(
            1,
            event_pb2.Event(
                wall_time=start, summary=summary_pb2.Summary(value=[note])
            ),
        )
        _write_events(dest / name, events)
        config = f"model: resnet50\ndropout: 0.0\nseed: {seed}\nlr: 0.0003\n"
        _artifacts(dest / name, 200 + seed, [], config)
    _write_annotations(dest)
    return dest


def _write_annotations(dest: Path) -> None:
    """Write a starter annotations file that the app imports on first start."""
    import json

    from tensorboard_book.scanner import fingerprint

    def fp(name: str) -> str:
        files = [
            (p.relative_to(dest / name).as_posix(), 0, 0)
            for p in (dest / name).rglob("events.out.tfevents.*")
        ]
        return fingerprint(dest / name, files)

    sweep = [
        "resnet_lr1e-2",
        "resnet_lr3e-3",
        "resnet_lr1e-3",
        "resnet_lr3e-4",
        "resnet_lr1e-4",
    ]
    aug = [f"aug_{a}_seed{s}" for a in ("none", "mixup") for s in (1, 2, 3)]
    runs = [
        {
            "path": "resnet_lr3e-4",
            "tags": ["baseline"],
            "notes": "Best of the sweep.",
            "starred": True,
        },
        {
            "path": "resnet_lr1e-2",
            "tags": ["unstable"],
            "notes": "Diverged: loss went NaN.",
            "starred": False,
        },
        {
            "path": "resnet_lr3e-3",
            "tags": ["unstable", "resumed"],
            "notes": "",
            "starred": False,
        },
        {
            "path": "keras_baseline",
            "tags": ["baseline"],
            "notes": "",
            "starred": False,
        },
    ]
    # Example custom properties: the dataset (a category) for every run,
    # and a hand-given rating (a number) for a few.
    dataset = {n: "imagenet-1k" for n in sweep}
    dataset.update({n: "cifar-100" for n in aug})
    dataset.update(
        {"keras_baseline": "cifar-10", "no_dropout_s1": "imagenet-1k"}
    )
    dataset["no_dropout_s2"] = "imagenet-1k"
    rating = {"resnet_lr3e-4": 5.0, "resnet_lr1e-3": 4.0, "resnet_lr1e-2": 1.0}
    listed = {r["path"] for r in runs}
    runs += [
        {"path": n, "tags": [], "notes": "", "starred": False}
        for n in dataset
        if n not in listed
    ]
    for r in runs:
        r["properties"] = {"dataset": dataset[r["path"]]}
        if r["path"] in rating:
            r["properties"]["rating"] = rating[r["path"]]
    data = {
        "format": "tensorboard-book-annotations",
        "version": 1,
        "properties": [
            {
                "name": "dataset",
                "kind": "category",
                "description": "Training dataset.",
            },
            {
                "name": "rating",
                "kind": "number",
                "description": "How promising the run looks, 1 to 5.",
            },
        ],
        "groups": [
            {
                "name": "lr_sweep",
                "description": "Learning rate sweep for ResNet-50 with AdamW.",
                "selection_metric": "val/acc",
                "columns": ["val/acc", "val/loss", "test/acc"],
                "members": [fp(n) for n in sweep],
            },
            {
                "name": "aug_ablation",
                "description": "Does mixup help? Three seeds per setting.",
                "selection_metric": "val/acc",
                "columns": ["val/acc", "val/loss", "test/acc"],
                "members": [fp(n) for n in aug],
            },
            {
                "name": "no_dropout_abl",
                "description": "Dropout off, still training.",
                "selection_metric": "val/acc",
                "columns": ["val/acc", "val/loss"],
                "members": [fp(f"no_dropout_s{s}") for s in (1, 2)],
            },
        ],
        "runs": [
            {**r, "fingerprint": fp(r["path"]), "archived": False}
            for r in runs
        ],
    }
    backup = db.storage_dir(dest) / db.ANNOTATIONS_NAME
    backup.write_text(json.dumps(data, indent=2))
