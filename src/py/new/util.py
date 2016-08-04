import bisect


def time_adaptive(timer, ratio=0.9965, minreps=10, maxreps=100):
    times = []
    reps = 0
    cond = True
    while cond:
        time = timer()
        bisect.insort_left(times, time)
        reps += 1
        good_enough = times[0]/times[1] > ratio if reps > 1 else False
        stop = reps >= maxreps or good_enough
        cond = reps < minreps or not stop
    return times
