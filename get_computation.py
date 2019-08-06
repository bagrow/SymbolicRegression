import cpuinfo


def get_cycles_per_second(time, measurements=50):
    """Given the time allowed,
    determine the amount of computation
    that can be performed.

    Parameters
    ----------
    time : float
        The amount of time in seconds.
    measurements : int
        The number of times to measure
        the cylces per second.

    Returns
    -------
    computation : float
        The amount of computation. This
        is the speed of the cpu multiplied
        by time.
    """

    cycles_per_second = sum([cpuinfo.get_cpu_info()['hz_actual_raw'][0] for _ in range(measurements)])/measurements

    return cycles_per_second


def get_computation_time(time):
    """Assuming that the given time is on
    a 2 GHz processor, compute the amount
    of time to acheive the same amount of
    computation on the processor that this
    function is being run.

    Parameters
    ----------
    time : float
        The amount of time in seconds
        on a 2 GHz processor.

    Returns
    -------
    adjusted_time : float
        The amount of time in seconds
        on the current processor.
    """

    cycles_per_second = get_cycles_per_second(time)

    adjusted_time = 2*10**9 * time / cycles_per_second

    return adjusted_time


if __name__ == '__main__':

    # 30 hours
    time = 30*3600

    for _ in range(10):

        # cycles_per_second = get_cycles_per_second(time)

        # print('time', time, 'cycles per second', '{:.2e}'.format(cycles_per_second))

        adjusted_time = get_computation_time(time)

        print('adjusted_time', adjusted_time)
