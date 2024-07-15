"""
Extends common config by stage-specific config and generates consolidated config file.

The script extends the common HOCON config (COMMON.conf) by the stage specific config according to the
stage parameter given and then creates a consolidated config of the given format (YAML, JSON or HOCON). The consolidated
config is then written to the output path given. The input directory for the config files can be specified optionally.
"""

import logging
from pathlib import Path
from typing import Optional

from pyhocon import ConfigFactory, ConfigTree, HOCONConverter

LOGGER = logging.getLogger(__name__)


class NoCommonConfigFoundException(Exception):
    pass


def create_config(conf_path: str) -> ConfigTree:
    """Merge a cross-stage HOCON config file (COMMON.conf) with an optional stage-specific config file.

    Args:
        conf_path: Input directory for all config files

    Returns:
        Merged PyHOCON config tree
    """
    common_config_file = Path(f"{conf_path}/COMMON.conf")

    if not common_config_file.exists():
        LOGGER.error("No COMMON config found.")
        raise NoCommonConfigFoundException

    config = ConfigFactory.parse_file(common_config_file, resolve=False)
    return config
