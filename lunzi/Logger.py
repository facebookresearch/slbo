# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from termcolor import colored
import datetime
import sys
import os
from collections import Counter, defaultdict
import json_tricks


def a():
    pass


_srcfile = os.path.normcase(a.__code__.co_filename)


class BaseSink(object):
    @staticmethod
    def _time():
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')

    def info(self, fmt, *args, **kwargs):
        raise NotImplementedError

    def warning(self, fmt, *args, **kwargs):
        self.info(fmt, *args, **kwargs)

    def verbose(self, fmt, *args, **kwargs):
        pass


class StdoutSink(BaseSink):
    def __init__(self):
        self.freq_count = Counter()

    def info(self, fmt, *args, freq=1, caller=None):
        if args:
            fmt = fmt % args
        self.freq_count[caller] += 1
        if self.freq_count[caller] % freq == 0:
            print("%s - %s - %s" % (colored(self._time(), 'green'),
                                    colored(caller, 'cyan'), fmt), flush=True)

    def warning(self, fmt, *args, **kwargs):
        if args:
            fmt = fmt % args
        self.info(colored(fmt, 'yellow'), **kwargs)


class FileSink(BaseSink):
    def __init__(self, fn):
        self.log_file = open(fn, 'w')
        self.callers = {}

    def info(self, fmt, *args, **kwargs):
        self._kv(level='info', fmt=fmt, args=args, **kwargs)

    def warning(self, fmt, *args, **kwargs):
        self._kv(level='warning', fmt=fmt, args=args, **kwargs)

    def _kv(self, **kwargs):
        kwargs.update(time=datetime.datetime.now())
        self.log_file.write(json_tricks.dumps(kwargs, primitives=True) + '\n')
        self.log_file.flush()

    def verbose(self, fmt, *args, **kwargs):
        self._kv(level='verbose', fmt=fmt, args=args, **kwargs)


class LibLogger(object):
    logfile = ""

    def __init__(self, name='logger', is_root=True):
        self.name = name
        self.is_root = is_root
        self.tab_keys = None
        self.sinks = []
        self.key_prior = defaultdict(np.random.randn)

    def add_sink(self, sink):
        self.sinks.append(sink)

    def info(self, fmt, *args, **kwargs):
        caller = self.find_caller()
        for sink in self.sinks:
            sink.info(fmt, *args, caller=caller, **kwargs)

    def warning(self, fmt, *args, **kwargs):
        caller = self.find_caller()
        for sink in self.sinks:
            sink.warning(fmt, *args, caller=caller, **kwargs)

    def verbose(self, fmt, *args, **kwargs):
        caller = self.find_caller()
        for sink in self.sinks:
            sink.verbose(fmt, *args, caller=caller, **kwargs)

    def find_caller(self):
        """
        Copy from `python.logging` module

        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = sys._getframe(1)
        if f is not None:
            f = f.f_back
        caller = ''
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == _srcfile:
                f = f.f_back
                continue
            # if stack_info:
            #     sio = io.StringIO()
            #     sio.write('Stack (most recent call last):\n')
            #     traceback.print_stack(f, file=sio)
            #     sio.close()
            # rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
            rel_path = os.path.relpath(co.co_filename, '')
            caller = f'{rel_path}:{f.f_lineno}'
            break
        return caller


def get_logger(name):
    return LibLogger(name)


logger = get_logger('Logger')
logger.add_sink(StdoutSink())
