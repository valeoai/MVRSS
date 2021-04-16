"""Script to set the path to CARRADA in the config.ini file"""
import argparse
from mvrss.utils.configurable import Configurable
from mvrss.utils import MVRSS_HOME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings paths for training and testing.')
    parser.add_argument('--carrada', default='/datasets_local',
                        help='Path to the CARRADA dataset.')
    parser.add_argument('--logs', default='/root/workspace/logs',
                        help='Path to the save the logs and models.')
    args = parser.parse_args()
    config_path = MVRSS_HOME / 'config_files' / 'config.ini'
    configurable = Configurable(config_path)
    configurable.set('data', 'warehouse', args.carrada)
    configurable.set('data', 'logs', args.logs)
    with open(config_path, 'w') as fp:
        configurable.config.write(fp)
