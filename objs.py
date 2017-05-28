import os
from constants import Constants


class Segment:
    def __init__(self, filename, start, end, prefix=Constants.VIDEO_FOLDER):
        self.filepath = os.path.join(prefix, filename)
        self.start = start
        self.end = end


class Face:
    def __init__(self, vector, pos=0, segment=None):
        self.descriptor = vector
        self.pos = pos
        self.segment = segment
