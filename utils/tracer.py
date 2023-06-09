import logging
import warnings

import numpy as np
from utils.meter import MeterFactory


class Tracer(object):
    """Class for history tracer.
    
    Parameter
    ---------
    win_size: Set the meter for tracer.
    """

    def __init__(self, win_size, logger=None):
        self._history = dict()
        self._meter_factory = MeterFactory(win_size)
        self.logger = logger

    def get_history(self):
        return self._history

    def get_meter(self, key):
        return self._history.setdefault(key, self._meter_factory())

    def update_history(self, x: int, data: dict, **kwargs):
        """Update the history only."""
        for key, value in data.items():
            self.get_meter(key).update(x, value, **kwargs)

    def logging(self):
        for k, m in self._history.items():
            self.logger.info("-------- %s: %s" % (k, m))
        self.logger.info("")