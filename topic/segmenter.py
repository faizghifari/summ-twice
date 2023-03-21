from abc import ABC, abstractmethod

class SegmenterInterface(ABC):
    @abstractmethod
    def segmentize(self, input_data, **kwargs):
        pass
