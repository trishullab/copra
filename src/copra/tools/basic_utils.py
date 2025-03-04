#!/usr/bin/env python3

import typing
from collections import Counter
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Stats(object):
    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0
    median: float = 0.0
    count: int = 0
    quartiles: typing.Dict[int, float] = field(default_factory=dict)
    percentile_with_sigmas : typing.Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.quartiles) == 0:
            self.quartiles[1] = 0.0
            self.quartiles[2] = 0.0
            self.quartiles[3] = 0.0
        if len(self.percentile_with_sigmas) == 0:
            self.percentile_with_sigmas[-3] = 0.0
            self.percentile_with_sigmas[-2] = 0.0
            self.percentile_with_sigmas[-1] = 0.0
            self.percentile_with_sigmas[0] = 0.0
            self.percentile_with_sigmas[1] = 0.0
            self.percentile_with_sigmas[2] = 0.0
            self.percentile_with_sigmas[3] = 0.0

class CounterUtils:
    @staticmethod
    def percentile(counter: Counter, p: float) -> float:
        assert isinstance(counter, Counter)
        assert isinstance(p, float)
        assert 0 <= p <= 1
        n = sum(counter.values())
        current = 0
        percentile_value = None
        r = p * (n - 1) + 1
        ri = int(r)
        rf = r - ri
        val_ri = None
        val_ri1 = None
        for k, v in sorted(counter.items()):
            if current + v >= ri and val_ri is None:
                val_ri = k
            if current + v >= ri + 1 and val_ri1 is None:
                val_ri1 = k
            current += v
            if val_ri is not None and val_ri1 is not None:
                percentile_value = val_ri + rf * (val_ri1 - val_ri)
                break
        if val_ri1 is None:
            percentile_value = val_ri
        assert percentile_value is not None
        return percentile_value
    
    @staticmethod
    def find_rank(counter: Counter, value: float) -> float:
        assert isinstance(counter, Counter)
        assert isinstance(value, float)
        current = 0
        prev_k = None
        current_k = None
        n = sum(counter.values())
        for k, v in sorted(counter.items()):
            current_k = k
            if k >= value:
                break
            prev_k = k
            current += v
        interpolation_factor = 0
        if prev_k is None:
            prev_k = current_k - 1
        elif current_k == prev_k:
            interpolation_factor = 0
        else:
            interpolation_factor = (value - prev_k) / (current_k - prev_k)

        rank = current + interpolation_factor
        return rank/n
        

    @staticmethod
    def mean(counter: Counter) -> float:
        assert isinstance(counter, Counter)
        current = 0
        n = 0
        for k, v in counter.items():
            current += k * v
            n += v
        mean_value = current / n
        return mean_value

    @staticmethod
    def median(counter: Counter) -> float:
        return CounterUtils.percentile(counter, 0.5)

    @staticmethod
    def max(counter: Counter) -> float:
        return max(counter)

    @staticmethod
    def min(counter: Counter) -> float:
        return min(counter)

    @staticmethod
    def std(counter: Counter) -> float:
        counter_squared = Counter()
        for k, v in counter.items():
            counter_squared[k*k] = v
        return (CounterUtils.mean(counter_squared) - CounterUtils.mean(counter)**2)**0.5
    
    @staticmethod
    def get_stats(counter: Counter) -> Stats:
        stats = Stats()
        stats.avg = CounterUtils.mean(counter)
        stats.min = CounterUtils.min(counter)
        stats.max = CounterUtils.max(counter)
        stats.std = CounterUtils.std(counter)
        stats.median = CounterUtils.median(counter)
        stats.count = sum(counter.values())
        stats.quartiles[1] = CounterUtils.percentile(counter, 0.25)
        stats.quartiles[2] = CounterUtils.percentile(counter, 0.5)
        stats.quartiles[3] = CounterUtils.percentile(counter, 0.75)
        for k in stats.percentile_with_sigmas:
            stats.percentile_with_sigmas[k] = CounterUtils.find_rank(counter, stats.avg + k * stats.std)
        return stats

if __name__ == "__main__":
    data = [1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    counter = Counter(data)
    print(f"counter: {counter}")
    print(f"percentile (0%): {CounterUtils.percentile(counter, 0.0)}")
    print(f"percentile (25%): {CounterUtils.percentile(counter, 0.25)}")
    print(f"percentile (50%): {CounterUtils.percentile(counter, 0.5)}")
    print(f"percentile (75%): {CounterUtils.percentile(counter, 0.75)}")
    mean = CounterUtils.mean(counter)
    print(f"mean: {mean}")
    print(f"median: {CounterUtils.median(counter)}")
    print(f"max: {CounterUtils.max(counter)}")
    print(f"min: {CounterUtils.min(counter)}")
    std = CounterUtils.std(counter)
    print(f"std: {std}")
    print(f"find_rank (0): {CounterUtils.find_rank(counter, 0.0)}")
    print(f"u + std: {mean + std}")
    print(f"u - std: {mean - std}")
    p_u_plus_std = CounterUtils.find_rank(counter, mean + std)
    p_u_minus_std = CounterUtils.find_rank(counter, mean - std)
    print(f"find_rank (u + std): {p_u_plus_std}")
    print(f"find_rank (u - std): {p_u_minus_std}")
    print(f"percentile (u + std): {CounterUtils.percentile(counter, p_u_plus_std)}")
    print(f"percentile (u - std): {CounterUtils.percentile(counter, p_u_minus_std)}")
    print(f"find_rank (u + 2 * std): {CounterUtils.find_rank(counter, mean + 2 * std)}")
    print(f"find_rank (u - 2 * std): {CounterUtils.find_rank(counter, mean - 2 * std)}")
    print(f"find_rank (u + 3 * std): {CounterUtils.find_rank(counter, mean + 3 * std)}")
    print(f"find_rank (u - 3 * std): {CounterUtils.find_rank(counter, mean - 3 * std)}")
    print(f"find_rank (10): {CounterUtils.find_rank(counter, 10.0)}")
    print(f"find_rank (11): {CounterUtils.find_rank(counter, 11.0)}")
    all_stats = CounterUtils.get_stats(counter)
    print(f"all_stats: {all_stats}")
    print(f"Serialized = \n{all_stats.to_json()}")
    pass