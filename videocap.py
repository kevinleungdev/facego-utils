import cv2
import os

from managers import CaptureManager, WindowManager

is_writing_img = False
idx = long(1)

output_dir = 'D:/Downloads/temp/screenshots'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def on_keypress(keycode):
    global is_writing_img

    if keycode == 27:
        win_mgr.destroy_window()
    elif keycode == 32:
        # space
        is_writing_img = not is_writing_img


win_mgr = WindowManager('Video Capture', keypress_callback=on_keypress)
cap_mgr = CaptureManager(cv2.VideoCapture('D:/Downloads/temp/10.mp4'), win_mgr, should_mirror_preview=False)

win_mgr.create_window()

while win_mgr.is_window_created:
    cap_mgr.enter_frame()

    frame = cap_mgr.frame
    
    # write image
    if is_writing_img:
        img_file = os.path.join(output_dir, '%06d.jpg' % idx)
        cv2.imwrite(img_file, frame)
        idx += 1

    cap_mgr.exit_frame()
    win_mgr.process_events()