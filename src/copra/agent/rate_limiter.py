#!/usr/bin/env python3

import time

class RateLimiter(object):
    def __init__(self, token_limit_per_min: int, request_limit_per_min: int):
        assert token_limit_per_min > 0, "Token limit must be greater than 0"
        assert request_limit_per_min > 0, "Request limit must be greater than 0"
        self.token_limit_per_min = token_limit_per_min
        self.request_limit_per_min = request_limit_per_min
        self._token_count = 0
        self._request_count = 0
        self._last_request_time = None

    def check(self, new_tokens: int = 0) -> bool:
        current_time = time.time()
        if self._last_request_time is None:
            self._last_request_time = current_time
        if current_time - self._last_request_time <= 60:
            if (self._token_count + new_tokens) >= self.token_limit_per_min or \
            (self._request_count + 1) >= self.request_limit_per_min:
                return False
        else:
            self.reset()
        return True

    def reset(self):
        self._token_count = 0
        self._request_count = 0
        self._last_request_time = None

    def update(self, token_count: int, request_start_time: float, request_end_time: float):
        self._token_count += token_count
        self._request_count += 1
        self._last_request_time = (request_start_time + request_end_time) / 2

    def __str__(self) -> str:
        return f"""
Tokens: {self._token_count}/{self.token_limit_per_min}
Requests: {self._request_count}/{self.request_limit_per_min}
Time Gap: {time.time() - self._last_request_time}
"""

class InvalidActionException(Exception):
    def __init__(self, message):
        self.message = message
    pass