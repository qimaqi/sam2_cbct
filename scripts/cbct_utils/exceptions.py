class CBCTError(Exception):
    code = 0


class DownloadInputFailedError(CBCTError):
    code = 1


class DicomReadFailedError(CBCTError):
    code = 2


class SegmentationModelsInitFailedError(CBCTError):
    code = 3


class SegmentationFailedError(CBCTError):
    code = 4


class UploadResultFailedError(CBCTError):
    code = 5


class HashValidationFailedError(CBCTError):
    code = 6


class ExtractionFailedError(CBCTError):
    code = 7
