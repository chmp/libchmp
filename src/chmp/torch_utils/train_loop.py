import itertools as it
import json
import pathlib
import sys

import torch

from ..ds import status as _status, smap
from . import make_data_loader
from ._aop import (
    joinpoint,
    add_aspect,
    before,
    after,
    around,
    modify,
    replace,
)


__all__ = [
    # All extension points of the train loop
    "train_loop",
    # Utilities
    "TrainLossHistory",
    # Re-export the AOP API
    "joinpoint",
    "add_aspect",
    "before",
    "after",
    "around",
    "modify",
    "replace",
]


@joinpoint("train_loop")
def train_loop(*, max_epochs, epoch=0, **kwargs):
    step = train_step

    for epoch in range(epoch, max_epochs):
        next_step = step(epoch=epoch, max_epochs=max_epochs, **kwargs)

        if next_step is False:
            break

        elif next_step is not None:
            step = next_step


@joinpoint("train_step")
def train_step(epoch, max_epochs, **kwargs):
    dl = data_loader(epoch=epoch, max_epochs=max_epochs, **kwargs)

    for idx, batch in enumerate(dl):
        step_result = optimizer_step(
            batch,
            epoch=epoch,
            max_epochs=max_epochs,
            **kwargs,
        )
        step_result = smap(lambda v: v.item(), step_result)

        status = prepare_status(
            epoch=epoch,
            batch=idx,
            n_batches=len(dl),
            max_epochs=max_epochs,
            dl=dl,
            step_result=step_result,
            **kwargs,
        )
        log_status(
            status,
            epoch=epoch,
            max_epochs=max_epochs,
            dl=dl,
            step_result=step_result,
        )


@joinpoint("data_loader")
def data_loader(*, dataset, batch_size=10, **kwargs):
    return make_data_loader(dataset, batch_size=batch_size)


@joinpoint("optimizer_step")
def optimizer_step(batch, *, optimizer, **kwargs):
    optimizer.zero_grad()

    loss = compute_loss(batch, optimizer=optimizer, **kwargs)
    _get_optimization_loss(loss).backward()
    optimizer.step()

    return loss


@joinpoint("compute_loss")
def compute_loss(batch, *, model, loss_fn, **kwargs):
    x, y = _get_xy(batch)
    return loss_fn(y, model(x))


@joinpoint("prepare_status")
def prepare_status(step_result, epoch, max_epochs, batch, n_batches, **kwargs):
    # TODO: handle different optimizer step results
    return dict(
        done=(
            "{:.1%}",
            (epoch + (batch + 1) / n_batches) / max_epochs,
        ),
        **_get_status_loss(step_result),
    )


@joinpoint("log_status")
def log_status(status, **kwargs):
    _status(**status)


def _get_optimization_loss(loss):
    if isinstance(loss, (tuple, dict)):
        return loss[0]

    elif isinstance(loss, dict):
        return loss["loss"]

    return loss


def _get_status_loss(loss):
    if isinstance(loss, (tuple, list)):
        return {f"loss_{i}": ("{:.4g}", item) for i, item in enumerate(loss)}

    elif isinstance(loss, dict):
        return {key: ("{:.4g}", val) for key, val in loss.items()}

    else:
        return {"loss": ("{:.4g}", loss)}


def _get_xy(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]

    elif isinstance(batch, dict):
        return batch["x"], batch["y"]

    else:
        raise TypeError("Batch must be a sequence or dict")


## Utilities
class TrainLossHistory:
    """Collect the train loss history

    Usage::

        with modify(train_loop) as train_loop:
            history = add_aspect(TrainLossHistory())

    """

    def __init__(self):
        self.history = None

    def optimizer_step_advice(self, proceed, /, *args, **kwargs):
        res = proceed(*args, **kwargs)

        if self.history is None:
            self.history = smap(lambda _: [], res)

        smap(lambda h, r: h.append(r.item()), self.history, res)

        return res

    def __getitem__(self, idx):
        return self.history[idx]

    @property
    def _aspects(self):
        return {"optimizer_step": [self.optimizer_step_advice]}

    def _ipython_key_completions_(self):
        if isinstance(self.history, dict):
            return list(self.history)

        else:
            return []


class Checkpointer:
    def __init__(self, path, *, every=None, keep=None, objects=None):
        self.path = pathlib.Path(path)
        self.objects = objects
        self.keep = keep
        self.every = every

    def _train_loop(self, proceed, /, epoch=0, **kwargs):
        if self.objects is None:
            self.objects = self._build_default_objects(kwargs)

        checkpoint = self._find_latest_checkpoint()
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        return proceed(epoch=epoch, **kwargs)

    def _train_step(self, proceed, /, *, epoch, **kwargs):
        res = proceed(epoch=epoch, **kwargs)

        if self._should_save(epoch):
            self._save(epoch)
            self._delete_outdated()

        return res

    @staticmethod
    def _build_default_objects(kwargs):
        objects = []
        if "model" in kwargs:
            objects.append(kwargs["model"])

        if "optimizer" in kwargs:
            objects.append(kwargs["optimizer"])

        return objects

    def _find_latest_checkpoint(self):
        return max(
            self.path.glob("checkpoint_*.json"),
            default=None,
            key=self._parse_checkpoint_name,
        )

    def _load_checkpoint(self, checkpoint):
        with open(checkpoint, "r") as fobj:
            meta = json.load(fobj)

        epoch = meta["epoch"] + 1

        for i, path in enumerate(meta["objects"]):
            self.objects[i].load_state_dict(torch.load(self.path / path))

    def _should_save(self, epoch):
        if self.every is None:
            return True

        return epoch > 0 and (epoch % self.every) == 0

    def _save(self, epoch):
        meta = {"epoch": epoch, "objects": []}

        self.path.mkdir(parents=True, exist_ok=True)

        for i, obj in enumerate(self.objects):
            path = self.path / f"checkpoint_{epoch}_{i}.pth"
            meta["objects"].append(path.name)
            torch.save(obj.state_dict(), path)

        with open(self.path / f"checkpoint_{epoch}.json", "wt") as fobj:
            json.dump(meta, fobj)

    def _delete_outdated(self):
        if self.keep is None:
            return

        checkpoints = sorted(
            self.path.glob("checkpoint_*.json"),
            key=self._parse_checkpoint_name,
        )
        for p in checkpoints[: -self.keep]:
            # TODO: use proper logger here
            # print(f"delete {p}", file=sys.stderr)
            with open(p, "r") as fobj:
                meta = json.load(fobj)

            for path in meta["objects"]:
                self.path.joinpath(path).unlink()

            p.unlink()

    @staticmethod
    def _parse_checkpoint_name(path):
        *_, epoch = path.stem.partition("_")
        return int(epoch)

    @property
    def _aspects(self):
        return {
            "train_loop": [self._train_loop],
            "train_step": [self._train_step],
        }
