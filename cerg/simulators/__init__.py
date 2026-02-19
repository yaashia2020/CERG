"""Simulator backends.

Usage:
    from cerg.simulators import MuJoCoSimulator, DrakeSimulator
"""


def MuJoCoSimulator(*args, **kwargs):
    from cerg.simulators.mujoco_sim import MuJoCoSimulator as _Cls
    return _Cls(*args, **kwargs)


def DrakeSimulator(*args, **kwargs):
    from cerg.simulators.drake_sim import DrakeSimulator as _Cls
    return _Cls(*args, **kwargs)
