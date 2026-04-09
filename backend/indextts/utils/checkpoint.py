# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import os
import re
from collections import OrderedDict

import torch
import yaml


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    checkpoint = torch.load(model_pth, map_location='cpu')
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Try strict loading first, fall back to non-strict if it fails
    try:
        model.load_state_dict(checkpoint, strict=True)
        print("✅ Model checkpoint loaded with strict matching")
    except RuntimeError as e:
        if "Unexpected key(s) in state_dict" in str(e) or "size mismatch" in str(e):
            print("⚠️ Strict loading failed, attempting non-strict loading...")
            print(f"   Error: {str(e)}")
            # Filter out incompatible keys
            model_dict = model.state_dict()
            filtered_checkpoint = {}
            missing_keys = []
            unexpected_keys = []
            
            for k, v in checkpoint.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        filtered_checkpoint[k] = v
                    else:
                        print(f"   ⚠️ Skipping {k} due to size mismatch: {v.shape} vs {model_dict[k].shape}")
                        missing_keys.append(k)
                else:
                    unexpected_keys.append(k)
            
            model.load_state_dict(filtered_checkpoint, strict=False)
            print(f"✅ Model checkpoint loaded with non-strict matching")
            print(f"   - Loaded {len(filtered_checkpoint)}/{len(checkpoint)} parameters")
            if unexpected_keys:
                print(f"   - Ignored {len(unexpected_keys)} unexpected keys (e.g., emotion conditioning modules)")
            if missing_keys:
                print(f"   - Skipped {len(missing_keys)} keys due to size mismatches")
        else:
            raise e
    
    info_path = re.sub('.pth$', '.yaml', model_pth)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs
