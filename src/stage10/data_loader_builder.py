import math
import random
from functools import partial

from src.utils.logger import get_logger
from src.stage10.batch_collate import asr_collate_fn

logger = get_logger("Stage10 DataLoader Builder")


# --------------------------------------------------
# Optional PyTorch DataLoader
# --------------------------------------------------

try:
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True

except Exception:
    DataLoader = None
    TORCH_AVAILABLE = False


# --------------------------------------------------
# Fallback Simple DataLoader
# This works even if PyTorch is not installed.
# --------------------------------------------------

class SimpleDataLoader:

    def __init__(
        self,
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=None
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def __len__(self):

        if len(self.dataset) == 0:

            return 0

        return math.ceil(
            len(self.dataset) / self.batch_size
        )

    def __iter__(self):

        indices = list(
            range(len(self.dataset))
        )

        if self.shuffle:

            random.shuffle(indices)

        for start_index in range(0, len(indices), self.batch_size):

            batch_indices = indices[
                start_index:start_index + self.batch_size
            ]

            batch = [

                self.dataset[index]

                for index in batch_indices

            ]

            if self.collate_fn is not None:

                yield self.collate_fn(batch)

            else:

                yield batch


def _build_single_loader(
    dataset,
    split_name,
    batch_size,
    shuffle,
    blank_id,
    num_workers=0
):

    logger.info("-" * 70)
    logger.info(f"Building DataLoader for split: {split_name}")
    logger.info("-" * 70)

    if dataset is None:

        return None, {

            "split": split_name,

            "status": "Failed",

            "message": "Dataset is None",

            "dataset_size": 0,

            "batch_size": batch_size,

            "num_batches": 0,

            "shuffle": shuffle,

            "loader_type": ""

        }

    dataset_size = len(dataset)

    if dataset_size == 0:

        return None, {

            "split": split_name,

            "status": "Failed",

            "message": "Dataset is empty",

            "dataset_size": 0,

            "batch_size": batch_size,

            "num_batches": 0,

            "shuffle": shuffle,

            "loader_type": ""

        }

    collate_function = partial(
        asr_collate_fn,
        feature_pad_value=0.0,
        label_pad_value=blank_id
    )

    # --------------------------------------------------
    # Use PyTorch DataLoader if available
    # --------------------------------------------------

    if TORCH_AVAILABLE:

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_function,
            num_workers=num_workers
        )

        loader_type = "PyTorch DataLoader"

    else:

        loader = SimpleDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_function
        )

        loader_type = "SimpleDataLoader"

    num_batches = len(loader)

    summary = {

        "split": split_name,

        "status": "Built",

        "message": "DataLoader built successfully",

        "dataset_size": dataset_size,

        "batch_size": batch_size,

        "num_batches": num_batches,

        "shuffle": shuffle,

        "loader_type": loader_type

    }

    logger.info(f"{split_name} dataset size : {dataset_size}")
    logger.info(f"{split_name} batch size   : {batch_size}")
    logger.info(f"{split_name} num batches  : {num_batches}")
    logger.info(f"{split_name} shuffle      : {shuffle}")
    logger.info(f"{split_name} loader type  : {loader_type}")

    return loader, summary


def build_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    blank_id=0,
    batch_size=4,
    num_workers=0
):

    logger.info("=" * 70)
    logger.info("Building DataLoaders")
    logger.info("=" * 70)

    train_loader, train_summary = _build_single_loader(
        dataset=train_dataset,
        split_name="train",
        batch_size=batch_size,
        shuffle=True,
        blank_id=blank_id,
        num_workers=num_workers
    )

    val_loader, val_summary = _build_single_loader(
        dataset=val_dataset,
        split_name="val",
        batch_size=batch_size,
        shuffle=False,
        blank_id=blank_id,
        num_workers=num_workers
    )

    test_loader, test_summary = _build_single_loader(
        dataset=test_dataset,
        split_name="test",
        batch_size=batch_size,
        shuffle=False,
        blank_id=blank_id,
        num_workers=num_workers
    )

    all_built = (
        train_summary["status"] == "Built"
        and val_summary["status"] == "Built"
        and test_summary["status"] == "Built"
    )

    loader_summary = {

        "status": "Built" if all_built else "Failed",

        "message": (
            "All DataLoaders built successfully"
            if all_built
            else "Some DataLoaders failed to build"
        ),

        "batch_size": batch_size,

        "num_workers": num_workers,

        "torch_available": TORCH_AVAILABLE,

        "train_summary": train_summary,

        "val_summary": val_summary,

        "test_summary": test_summary,

        "total_batches": (
            train_summary["num_batches"]
            + val_summary["num_batches"]
            + test_summary["num_batches"]
        )

    }

    data_loaders = {

        "train_loader": train_loader,

        "val_loader": val_loader,

        "test_loader": test_loader

    }

    logger.info("=" * 70)
    logger.info(f"DataLoader status : {loader_summary['status']}")
    logger.info(f"Batch size        : {batch_size}")
    logger.info(f"Total batches     : {loader_summary['total_batches']}")
    logger.info(f"PyTorch available : {TORCH_AVAILABLE}")
    logger.info("=" * 70)

    return data_loaders, loader_summary