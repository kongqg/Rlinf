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

import numpy as np
import torch

from rlinf.envs import SupportedEnvType


def prepare_actions(
    raw_chunk_actions,
    env_type: str,
    model_type: str,
    num_action_chunks,
    action_dim,
    action_scale: float = 1.0,
    policy: str = "aloha-agilex",
    wm_env_type=None,
) -> torch.Tensor | np.ndarray:
    del model_type, num_action_chunks, action_dim, action_scale, policy, wm_env_type

    env_type = SupportedEnvType(env_type)
    if env_type != SupportedEnvType.ROBOTWIN:
        raise NotImplementedError(
            f"Only robotwin actions are kept locally, got env_type={env_type.value!r}"
        )

    return (
        raw_chunk_actions.cpu().numpy()
        if isinstance(raw_chunk_actions, torch.Tensor)
        else raw_chunk_actions
    )
