import cv2
import time
import numpy as np


class CaptureManager(object):

    def __init__(self, capture, preview_window_manager=None, should_mirror_preview=False):

        self.preview_window_manager = preview_window_manager
        self.should_mirror_preview = should_mirror_preview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None

        self._start_time = None
        self._frames_elapsed = long(0)
        self._fps_estimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()

        return self._frame

    def enter_frame(self):
        """ Capture the next frame, if any. """

        # But first, check that any previous enter frame was exited
        assert not self._enteredFrame, \
            'previous enter_frame() has no matching exit_frame()'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exit_frame(self):
        """Draw to the window. Write to files. Release the frame"""

        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.frame is None:
            self._enteredFrame = False
            return

        # Update the FPS estimate and related variables.
        if self._frames_elapsed == 0:
            self._start_time = time.time()
        else:
            time_elapsed = time.time() - self._start_time
            self._fps_estimate = self._frames_elapsed / time_elapsed
        self._frames_elapsed += 1

        # Draw to the window, if any.
        if self.preview_window_manager is not None:
            if self.should_mirror_preview:
                mirrored_frame = np.fliplr(self._frame).copy()
                self.preview_window_manager.show(mirrored_frame)
            else:
                self.preview_window_manager.show(self._frame)

        # Release the frame.
        self._frame = None
        self._enteredFrame = False


class WindowManager(object):
    def __init__(self, window_name, keypress_callback=None):
        self.keypressCallback = keypress_callback

        self._windowName = window_name
        self._isWindowCreated = False

    @property
    def is_window_created(self):
        return self._isWindowCreated

    def create_window(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroy_window(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def process_events(self):
        keycode = cv2.waitKey(1)

        if self.keypressCallback is not None and keycode != -1:
            keycode &= 0xFF
            self.keypressCallback(keycode)