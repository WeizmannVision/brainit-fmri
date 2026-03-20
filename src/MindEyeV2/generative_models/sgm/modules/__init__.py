# Code copied from https://github.com/MedARC-AI/MindEyeV2

from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
