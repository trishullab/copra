#!/usr/bin/env python3

import typing
from threading import Lock

class SimpleLruCache(object):
    KiB = 2**10
    MiB = 2**20
    GiB = 2**30
    def __init__(self, max_size_in_bytes: int = 100*MiB):
        assert max_size_in_bytes is not None, "Max size in MiB cannot be None"
        assert max_size_in_bytes > 0, "Max size in MiB must be greater than 0"
        self.max_size_in_bytes = max_size_in_bytes
        self.cache = {}
        self.cache_order = []
        self.cache_size_in_bytes = 0
        self.mutex = Lock()
    
    def add_to_cache(self, key: str, value: typing.Any, size_in_bytes: int):
        assert key is not None, "File path cannot be None"
        assert value is not None, "File contents cannot be None"
        self.mutex.acquire()
        try:
            self._try_remove_from_cache(key)
            self.cache[key] = (value, size_in_bytes)
            self.cache_order.append(key)
            self.cache_size_in_bytes += size_in_bytes
            while self.cache_size_in_bytes > self.max_size_in_bytes:
                self._try_remove_lru_key()
        finally:
            self.mutex.release()
    
    def get_from_cache(self, key: str):
        assert key is not None, "File path cannot be None"
        self.mutex.acquire()
        try:
            if key in self.cache:
                self.cache_order.remove(key)
                self.cache_order.append(key)
                value, _ = self.cache[key]
                return value
            return None
        finally:
            self.mutex.release()
    
    def try_remove_from_cache(self, key: str) -> bool:
        self.mutex.acquire()
        try:
            return self._try_remove_from_cache(key)
        finally:
            self.mutex.release()
    
    def clear_cache(self):
        self.mutex.acquire()
        try:
            while self._try_remove_lru_key():
                pass
        finally:
            self.mutex.release()

    # define in operator
    def __contains__(self, key: str):
        self.mutex.acquire()
        try:
            return key in self.cache
        finally:
            self.mutex.release()

    # unsafe method
    def _try_remove_lru_key(self) -> bool:
        if len(self.cache_order) > 0:
            self._try_remove_from_cache(self.cache_order[0])
            return True
        return False
    
    # unsafe method
    def _try_remove_from_cache(self, key: str) -> bool:
        assert key is not None, "File path cannot be None"
        if key in self.cache:
            val, size = self.cache.pop(key)
            self.cache_size_in_bytes -= size
            self.cache_order.remove(key)
            del val
            return True
        return False