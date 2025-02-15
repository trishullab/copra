#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
import logging

def setup_logger(name, log_file, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger