import cv2
import os
import numpy as np


class ExtractImageFromVideo(object):
    def __init__(self, path):
        assert os.path.exists(path)

        self._vc = cv2.VideoCapture(path)

        self.size = int(self._vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self._vc.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self._vc.get(cv2.CAP_PROP_FRAME_COUNT))

        self._count = self.total_frames

    def __del__(self):
        self.release()

    def extract(self):
        for i in range(0, self._count):
            success, frame = self._vc.read()
            if not success:
                print(f"index {i} exceeded.")
                break
            yield frame

    def release(self):
        if self._vc is not None:
            self._vc.release()


def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-2], images.shape[-3]

    if grid_size is not None:
        grid_h, grid_w = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros([grid_h * img_h, grid_w * img_w] + list(images.shape[-1:]), dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y: y + img_h, x: x + img_w, ...] = images[idx]
    return grid


def merge_videos(videos_in, video_out, grid_size=None):
    """
    Args:
        videos_in: List/Tuple
            List of input video paths. e.g.
                ('path/to/v1.mp4', 'path/to/v2.mp4', 'path/to/v3.mp4')
        video_out: String
            Path of output video. e.g.
                'path/to/output.mp4'
        grid_size: List/Tuple.
            Row and Column respectively. e.g.
                (1, 3)
    Returns:
        None
    """
    video_handles = []
    for v in videos_in:
        assert os.path.exists(v), f'{v} not exists!'
        video_handles.append(ExtractImageFromVideo(v))

    least_frames = sorted([e.total_frames for e in video_handles])[0]  # all with same number of frames

    least_size = sorted([e.size for e in video_handles])[0]  # all with same size WH
    generators = [e.extract() for e in video_handles]

    # read one frame and resize for each generator, then get the output video size
    cur_frames = np.array([cv2.resize(next(g), least_size) for g in generators])
    frames_grid = create_image_grid(cur_frames, grid_size=grid_size)  # HWC

    fps = video_handles[0].fps  # use the fps of first video
    out_size = frames_grid.shape[0:2]  # HWC to HW
    out_size = out_size[::-1]  # reverse HW to WH, as VideoWriter need that format
    video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   100,
                                   out_size)

    for n in range(least_frames - 1):
        if n % 100 == 0:
            print(f'{n}: {len(cur_frames)} frames merge into grid with size={frames_grid.shape}')
        video_writer.write(frames_grid)

        cur_frames = np.array([cv2.resize(next(g), least_size) for g in generators])
        frames_grid = create_image_grid(cur_frames, grid_size=grid_size)

    video_writer.release()
    print(f'Output video saved... {video_out}')
