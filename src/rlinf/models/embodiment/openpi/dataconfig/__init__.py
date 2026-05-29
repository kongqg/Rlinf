# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import difflib
from typing import Optional

import openpi.models.pi0_config as pi0_config
from openpi.training.config import AssetsConfig, DataConfig, TrainConfig

from rlinf.models.embodiment.openpi.dataconfig.robotwin_aloha_dataconfig import (
    LeRobotAlohaDataConfig,
)

_CONFIGS = [
    TrainConfig(
        name="pi05_aloha_robotwin",
        model=pi0_config.Pi0Config(pi05=True, discrete_state_input=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/robotwin",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="checkpoints/torch/pi05_aloha_robotwin/assets"
            ),
            adapt_to_pi=False,
            extra_delta_transform=True,
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=20_000,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def _override_with_model_path(config: TrainConfig, model_path: str) -> TrainConfig:
    data_config = config.data
    if (
        dataclasses.is_dataclass(data_config)
        and hasattr(data_config, "assets")
        and dataclasses.is_dataclass(data_config.assets)
    ):
        data_config = dataclasses.replace(
            data_config,
            assets=dataclasses.replace(data_config.assets, assets_dir=model_path),
        )

    return dataclasses.replace(
        config,
        data=data_config,
        pytorch_weight_path=model_path,
    )


def _override_with_data_kwargs(config: TrainConfig, data_kwargs: dict) -> TrainConfig:
    return dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, **data_kwargs),
    )


def get_openpi_config(
    config_name: str,
    model_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    repo_id: Optional[str] = None,
    data_kwargs: Optional[dict] = None,
) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    config = _CONFIGS_DICT[config_name]
    if model_path is not None:
        config = _override_with_model_path(config, model_path)
    if data_kwargs is not None:
        config = _override_with_data_kwargs(config, data_kwargs)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)
    if repo_id is not None:
        original_repo_id = config.data.repo_id
        new_assets = dataclasses.replace(config.data.assets, asset_id=original_repo_id)
        new_data = dataclasses.replace(config.data, repo_id=repo_id, assets=new_assets)
        config = dataclasses.replace(config, data=new_data)
    return config
