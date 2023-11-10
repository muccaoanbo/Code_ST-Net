import toml
from easydict import EasyDict
import os
def get_config():
    with open(os.path.join('/data0/BME_caoanbo/project_coding/K-trans_STNet_version1/ST/config.toml'), 'r', encoding='utf-8') as f:
        config = toml.load(f)
        return EasyDict(config)
