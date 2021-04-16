"""Class to get global paths"""
from pathlib import Path
from mvrss.utils import MVRSS_HOME
from mvrss.utils.configurable import Configurable


class Paths(Configurable):

    def __init__(self):
        self.config_path = MVRSS_HOME / 'config_files' / 'config.ini'
        super().__init__(self.config_path)
        self.paths = dict()
        self._build()

    def _build(self):
        warehouse = Path(self.config['data']['warehouse'])
        self.paths['warehouse'] = warehouse
        self.paths['logs'] = Path(self.config['data']['logs'])
        self.paths['carrada'] = warehouse / 'Carrada'

    def get(self):
        return self.paths
