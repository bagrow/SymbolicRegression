import cpuinfo


def get_cycles_per_second(measurements=50):
    """Get the cycles per second of the current
    processor.

    Parameters
    ----------
    measurements : int
        The number of times to measure
        the cylces per second.

    Returns
    -------
    cycles_per_second : float
        The cycles per second of 
        the current processor as
        determine through the
        average of multiple calls to
        cpuinfo.get_cpu_info
    """

    cycles_per_second = sum([cpuinfo.get_cpu_info()['hz_actual_raw'][0] for _ in range(measurements)])/measurements

    return cycles_per_second


def get_computation_time(time, return_cycles_per_second=False):
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
    return_cycles_per_second : bool
        If True, include return_cycles_per_second
        in the return.

    Returns
    -------
    adjusted_time : float
        The amount of time in seconds
        on the current processor.
    cycles_per_second : float
        This is only returned if
        return_cycles_per_second is
        true. The cycles per second of 
        the current processor as
        determine through the
        average of multiple calls to
        cpuinfo.get_cpu_info
    """

    cycles_per_second = get_cycles_per_second()

    adjusted_time = 2*10**9 * time / cycles_per_second

    if return_cycles_per_second:
        return adjusted_time, cycles_per_second

    else:
        return adjusted_time


if __name__ == '__main__':

    # 30 hours
    time = 30*3600

    for _ in range(10):

        # cycles_per_second = get_cycles_per_second(time)

        # print('time', time, 'cycles per second', '{:.2e}'.format(cycles_per_second))

        adjusted_time = get_computation_time(time)

        print('adjusted_time', adjusted_time)
