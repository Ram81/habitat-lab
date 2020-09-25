#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_object_rearrangement_dataset():
    try:
        from habitat.datasets.object_rearrangement.object_rearrangement_dataset import (  # noqa: F401 isort:skip
            RearrangementDatasetV1,
        )
    except ImportError as e:
        object_rearrangement_import_error = e

        @registry.register_dataset(name="RearrangementDS-v1")
        class RearrangementDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise object_rearrangement_import_error