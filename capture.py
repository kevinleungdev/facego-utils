import cv2
import numpy as np
import time
import os
import argparse

from mego.face import api
from managers import WindowManager, CaptureManager


DEFAULT_OUTPUT_DIR = "D:/Faces/data"


class App(object):

    def __init__(self, args):
        assert args.identity is not None
        assert os.path.exists(args.output_dir)

        self._output_dir = os.path.join(args.output_dir, args.identity)
        if not os.path.exists(self._output_dir):
            os.mkdir(self._output_dir)

        self._identity = args.identity

        self._window_manager = WindowManager('Face Capture', self.on_keypress)
        self._capture_manager = CaptureManager(cv2.VideoCapture(0), self._window_manager, True)

        self._img_count = long(0)
        self._auto_write_img = False

        self._cropped = None

        self._image_size = args.image_size

    def run(self):
        """Run the main loop"""
        self._window_manager.create_window()

        while self._window_manager.is_window_created:
            self._capture_manager.enter_frame()

            frame = self._capture_manager.frame

            if frame is not None:
                # face detection
                faces = api.detect_faces(frame)

                for face in faces:
                    self._cropped = cv2.resize(frame[face[1]:face[3], face[0]:face[2]],
                                               dsize=(self._image_size, self._image_size), interpolation=cv2.INTER_CUBIC)

                    if self._auto_write_img:
                        # face cropped from screen shot
                        self.write_image(self._cropped)

                    # mark face rectangle
                    cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)

            self._capture_manager.exit_frame()
            self._window_manager.process_events()

    def on_keypress(self, keycode):
        if keycode == 9:
            # tab
            self._auto_write_img = not self._auto_write_img
        elif keycode == 32:
            # space
            self.write_image(self._cropped)
        elif keycode == 27:
            # escape
            self._window_manager.destroy_window()

    def write_image(self, cropped):
        if cropped is None:
            print 'Cropped is none'
            return

        filename = os.path.join(self._output_dir,
                                (self._identity.lower() + ('_%06d.jpg' % self._img_count)))
        mirror_filename = os.path.join(self._output_dir,
                                (self._identity.lower() + ('_%06d_mirror.jpg' % self._img_count)))

        print 'Saving image ', filename
        # save to disk
        cv2.imwrite(filename, cropped)

        print 'Saving mirrored image ', mirror_filename
        mirror_cropped = np.fliplr(cropped).copy()
        cv2.imwrite(mirror_filename, mirror_cropped)

        # increase count
        self._img_count += 1


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('identity', type=str)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--image_size', type=int, default=160, help='Image size (height, width) in pixels.')

    return parser.parse_args(argv)


def main(args):
    app = App(args)
    app.run()


if __name__ == "__main__":
    import sys
    main(parse_arguments(sys.argv[1:]))



