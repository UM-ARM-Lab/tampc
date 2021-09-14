import enum

pandaEndEffectorIndex = 11
pandaNumDofs = 7


class PandaGripperID(enum.IntEnum):
    FINGER_A = 9
    FINGER_B = 10


class PandaJustGripperID(enum.IntEnum):
    FINGER_A = 0
    FINGER_B = 1
