from pathlib import Path
import yaml
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parent.parent

def _load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f: # Added encoding='utf-8'
        return yaml.safe_load(f)

@dataclass
class Cfg:
    base   : dict = field(default_factory=dict)
    model  : dict = field(default_factory=dict)
    store  : dict = field(default_factory=dict)

    def __post_init__(self):
        self.device      = self.base['device']
        self.embed_dim   = self.base['embed_dim']
        self.topk_recall = self.base['topk']['recall']
        self.topk_return = self.base['topk']['return']
        self.diff_coeff  = self.base['difficulty']['coeff']
        self.store_name  = self.base['store']

# Determine the store config file name from base.yaml
_base_config = _load_yaml(ROOT / 'conf/base.yaml')
_store_type = _base_config["store"]
_store_config_file = ROOT / f'conf/{_store_type}.yaml'

CFG = Cfg(
    base  = _base_config,
    model = _load_yaml(ROOT / 'conf/model.yaml'),
    store = _load_yaml(_store_config_file)
) 