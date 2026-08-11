import asyncio

from vlmeval.ratelimit import RateLimiter


async def test_concurrency_cap():
    limiter = RateLimiter(concurrency=2)
    active = 0
    peak = 0

    async def job():
        nonlocal active, peak
        async with limiter:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(job() for _ in range(6)))
    assert peak == 2


async def test_rpm_sliding_window_with_fake_clock():
    t = [0.0]
    slept: list[float] = []

    async def fake_sleep(d):
        slept.append(d)
        t[0] += d

    limiter = RateLimiter(concurrency=10, rpm=3, clock=lambda: t[0], sleep=fake_sleep)

    for _ in range(3):  # first 3 pass instantly at t=0
        async with limiter:
            pass
    assert slept == []

    async with limiter:  # 4th must wait until the t=0 stamps age out
        pass
    assert sum(slept) >= 60.0
    assert t[0] >= 60.0


async def test_rpm_window_frees_up_after_time_passes():
    t = [0.0]

    async def fake_sleep(d):
        t[0] += d

    limiter = RateLimiter(concurrency=10, rpm=2, clock=lambda: t[0], sleep=fake_sleep)
    async with limiter:
        pass
    async with limiter:
        pass
    t[0] = 61.0  # window expired — no sleep needed
    before = t[0]
    async with limiter:
        pass
    assert t[0] == before
