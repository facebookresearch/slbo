# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import os
import yaml
from lunzi.Logger import logger


_frozen = False
_initialized = False


def expand(path):
    return os.path.abspath(os.path.expanduser(path))


class MetaFLAGS(type):
    _initialized = False

    def __setattr__(self, key, value):
        assert not _frozen, 'Modifying FLAGS after dumping is not allowed!'
        super().__setattr__(key, value)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not isinstance(value, classmethod):
                if isinstance(value, MetaFLAGS):
                    value = dict(value)
                yield key, value

    def as_dict(self):
        return dict(self)

    def merge(self, other: dict):
        for key in other:
            assert key in self.__dict__, f"Can't find key `{key}`"
            if isinstance(self[key], MetaFLAGS) and isinstance(other[key], dict):
                self[key].merge(other[key])
            else:
                setattr(self, key, other[key])

    def set_value(self, path, value):
        key, *rest = path
        assert key in self.__dict__, f"Can't find key `{key}`"
        if not rest:
            setattr(self, key, value)
        else:
            self[key]: MetaFLAGS
            self[key].set_value(rest, value)

    @staticmethod
    def set_frozen():
        global _frozen
        _frozen = True

    def freeze(self):
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, MetaFLAGS):
                    value.freeze()
        self.finalize()

    def finalize(self):
        pass


class BaseFLAGS(metaclass=MetaFLAGS):
    pass


def parse(cls):
    global _initialized

    if _initialized:
        return
    parser = argparse.ArgumentParser(description='Stochastic Lower Bound Optimization')
    parser.add_argument('-c', '--config', type=str, help='configuration file (YAML)', nargs='+', action='append')
    parser.add_argument('-s', '--set', type=str, help='additional options', nargs='*', action='append')

    args, unknown = parser.parse_known_args()
    for a in unknown:
        logger.info('unknown arguments: %s', a)
    # logger.info('parsed arguments = %s, unknown arguments: %s', args, unknown)
    if args.config:
        for config in sum(args.config, []):
            cls.merge(yaml.load(open(expand(config))))
    else:
        logger.info('no config file specified.')
    if args.set:
        for instruction in sum(args.set, []):
            path, *value = instruction.split('=')
            cls.set_value(path.split('.'), yaml.load('='.join(value)))

    _initialized = True

