from __future__ import absolute_import

import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2
import av
from dataloaders import decoder
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def get_video_decoding_kwargs(container, num_frames, target_fps,
                              num_clips=None, clip_idx=None,
                              sampling_strategy="rand",
                              safeguard_duration=False, video_max_pts=None):
    if num_clips is None:
        three_clip_names = ["start", "middle", "end"]  # uniformly 3 clips
        assert sampling_strategy in ["rand", "uniform"] + three_clip_names
        if sampling_strategy == "rand":
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,
                num_frames=num_frames,
                clip_idx=-1,  # random sampling
                num_clips=None,  # will not be used when clip_idx is `-1`
                target_fps=target_fps
            )
        elif sampling_strategy == "uniform":
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,  # will not be used when clip_idx is `-2`
                num_frames=num_frames,
                clip_idx=-2,  # uniformly sampling from the whole video
                num_clips=1,  # will not be used when clip_idx is `-2`
                target_fps=target_fps  # will not be used when clip_idx is `-2`
            )
        else:  # in three_clip_names
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,
                num_frames=num_frames,
                clip_idx=three_clip_names.index(sampling_strategy),
                num_clips=3,
                target_fps=target_fps
            )
    else:  # multi_clip_ensemble, num_clips and clip_idx are only used here
        assert clip_idx is not None
        # sampling_strategy will not be used, as uniform sampling will be used by default.
        # uniformly sample `num_clips` from the video,
        # each clip sample num_frames frames at target_fps.
        decoder_kwargs = dict(
            container=container,
            sampling_rate=1,
            num_frames=num_frames,
            clip_idx=clip_idx,
            num_clips=num_clips,
            target_fps=target_fps,
            safeguard_duration=safeguard_duration,
            video_max_pts=video_max_pts
        )
    return decoder_kwargs

def extract_frames_from_video_binary(
        in_mem_bytes_io, target_fps=3, num_frames=3, num_clips=None, clip_idx=None,
        multi_thread_decode=False, sampling_strategy="rand",
        safeguard_duration=False, video_max_pts=None):
    """
    Args:
        in_mem_bytes_io: binary from read file object
            >>> with open(video_path, "rb") as f:
            >>>     input_bytes = f.read()
            >>> frames = extract_frames_from_video_binary(input_bytes)
            OR from saved binary in lmdb database
            >>> env = lmdb.open("lmdb_dir", readonly=True)
            >>> txn = env.begin()
            >>> stream = io.BytesIO(txn.get(str("key").encode("utf-8")))
            >>> frames = extract_frames_from_video_binary(stream)
            >>> from torchvision.utils import save_image
            >>> save_image(frames[0], "path/to/example.jpg")  # save the extracted frames.
        target_fps: int, the input video may have different fps, convert it to
            the target video fps before frame sampling.
        num_frames: int, number of frames to sample.
        multi_thread_decode: bool, if True, perform multi-thread decoding.
        sampling_strategy: str, how to sample frame from video, one of
            ["rand", "uniform", "start", "middle", "end"]
            `rand`: randomly sample consecutive num_frames from the video at target_fps
                Note it randomly samples a clip containing num_frames at target_fps,
                not uniformly sample from the whole video
            `uniform`: uniformly sample num_frames of equal distance from the video, without
                considering target_fps/sampling_rate, etc. E.g., when sampling_strategy=uniform
                and num_frames=3, it samples 3 frames at [0, N/2-1, N-1] given a video w/ N frames.
                However, note that when num_frames=1, it will sample 1 frame at [0].
                Also note that `target_fps` will not be used under `uniform` sampling strategy.
            `start`/`middle`/`end`: first uniformly segment the video into 3 clips, then sample
                num_frames from the corresponding clip at target_fps. E.g., num_frames=3, a video
                w/ 30 frames, it samples [0, 1, 2]; [9, 10, 11]; [18, 19, 20] for start/middle/end.
            If the total #frames at target_fps in the video/clip is less than num_frames,
            there will be some duplicated frames
        num_clips: int,
        clip_idx: int
        safeguard_duration:
        video_max_pts: resue it to improve efficiency

    Returns:
        torch.uint8, (T, C, H, W)
    """
    try:
        # Add `metadata_errors="ignore"` to ignore metadata decoding error.
        # When verified visually, it does not seem to affect the extracted frames.
        video_container = av.open(in_mem_bytes_io, metadata_errors="ignore")
    except Exception as e:
        logger.info(f"Exception in loading video binary: {e}")
        return None, None

    if multi_thread_decode:
        # Enable multiple threads for decoding.
        video_container.streams.video[0].thread_type = "AUTO"
    # (T, H, W, C), channels are RGB
    # see docs in decoder.decode for usage of these parameters.
    decoder_kwargs = get_video_decoding_kwargs(
        container=video_container, num_frames=num_frames,
        target_fps=target_fps, num_clips=num_clips, clip_idx=clip_idx,
        sampling_strategy=sampling_strategy,
        safeguard_duration=safeguard_duration, video_max_pts=video_max_pts)
    frames, video_max_pts = decoder.decode(**decoder_kwargs)
    # (T, H, W, C) -> (T, C, H, W)
    if frames is not None:
        frames = frames.permute(0, 3, 1, 2)
    return frames, video_max_pts


class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2