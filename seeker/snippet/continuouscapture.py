#date: 2021-10-26T16:57:24Z
#url: https://api.github.com/gists/82363c737a08ff02acd2fbf0a00b002a
#owner: https://api.github.com/users/jepler

# pure Python helper class living in Adafruit_CircuitPython_OVxxxx
class ContinuousCaptureFrame:
    def __init__(self, camera) -> None:
        self._camera = camera
        self._buffer = camera.take_frame()

    def __enter__(self) -> WriteableBuffer:
        return self._buffer

    def __exit__(self, _, _, _):
        self._camera.give_frame(self._buffer)
        
#  In the implementation of the camera class
class OV5640:
    ...

    @property
    def continuous_capture_frame(self) -> ContinuousCaptureFrame:
        """Return the next frame available, wrapped in a ContinuousCaptureFrame

        Typical usage:

        .. code-block::

            cam = ...
            buffer1 = bytearray(cam.capture_buffer_size)
            buffer2 = bytearray(cam.capture_buffer_size)
            cam.start_continuous_capture(buffer1, buffer2)
            while True:
                with cam.continuous_capture_frame as buf:
                    # do something with buf
        """
        return ContinuousCaptureFrame(self)

    def capture(self, buffer: WritableBuffer) -> None:
        """Capture a frame in single-shot mode."""
        ...

    def stop_continuous_capture(self) -> None:
        """Stop continuous capture"""
        ...

    @property
    def frame_available(self) -> bool:
        """Return True if a frame is ready right now.""
        ...
    
# The core class
class ParallelImageCapture:
    ...

    def capture(self, buffer: WritableBuffer) -> None:
        """Capture a frame in single-shot mode.

        If the camera is in continuous-capture mode, an error is raised."""
        ...


    def start_continuous_capture(self, buffer1: WriteableBuffer, buffer2: WriteableBuffer) -> None:
        """Enable continuous capture alternately into buffer1 and buffer2

        This is more efficient than using `capture`, because the capture
        hardware can capture the next frame while the previous frame is
        being used.
        """

    def stop_continuous_capture(self) -> None:
        """Stop continuous capture"""

    @property
    def frame_available(self) -> bool:
        """Return True if a frame is ready right now.

        Raises an exception if not in continuous capture mode"""

    # give/take nomenclature is from esp32-camera
    def take_frame(self) -> WritableBuffer:
        """Wait until a continuous-capture frame is available, then return its buffer.

        Once done using the buffer data, pass it to `give_buffer`.

        Usually, the `with cam.continuous_capture_frame as buf:`
        style of code should be used instead.

        Raises an exception if not in continuous capture mode."""

    @property
    def give_frame(self, buffer: WriteableBuffer):
        """Give back the buffer returned earlier by take_buffer

        Usually, the `with cam.capture() as buf:` style of code should be used
        instead.

        Raises an exception if not in continuous capture mode."""
