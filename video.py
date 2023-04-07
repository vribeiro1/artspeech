import numpy as np
import pydicom
import torch
import torchaudio

from operator import itemgetter


class Video:
    def __init__(
            self,
            frames_filepaths,
            audio_filepath,
            framerate=50,
            max_diff=0.0025,
        ):
        audio, self.sample_rate = torchaudio.load(audio_filepath)
        audio = torch.mean(audio, dim=0).squeeze(dim=0)

        self.num_samples, = audio.shape
        audio_duration = self.num_samples / self.sample_rate

        self.framerate = framerate
        self.num_frames = len(frames_filepaths)
        video_duration = self.num_frames / self.framerate

        diff = abs(video_duration - audio_duration)
        if diff > max_diff:
            raise ValueError(f"Difference in duration of audio and video is too large ({diff})")
        self.duration = video_duration

        self.audio = audio
        self.frames_filepaths = frames_filepaths

    @staticmethod
    def load_frame(filepath):
        ds = pydicom.dcmread(filepath)
        frame = torch.tensor(ds.pixel_array.astype(np.float))
        return frame

    def get_audio_interval(self, start, end):
        time = np.linspace(0., self.duration, self.num_samples)
        ge_start, = np.where(time >= start)  # Greater than or equal to the start
        lt_end, = np.where(time < end)  # Lower than the end
        indices = sorted(set(ge_start) & set(lt_end))
        audio_interval = self.audio[indices]
        return torch.tensor(time[indices], dtype=torch.float), audio_interval

    def get_frames_interval(self, start, end, load_frames=False):
        time = np.linspace(0., self.duration, self.num_frames)
        ge_start, = np.where(time >= start)  # Greater than or equal to the start
        lt_end, = np.where(time < end)  # Lower than the end
        indices = list(set(ge_start) & set(lt_end))

        if len(indices) == 0:
            return torch.tensor([], dtype=torch.float), []

        frames_filepaths = itemgetter(*indices)(self.frames_filepaths)
        if isinstance(frames_filepaths, str):
            frames_filepaths = [frames_filepaths]
        frames_filepaths = sorted(frames_filepaths)

        if load_frames:
            frames = torch.stack([self.load_frame(fp) for fp in frames_filepaths])
        else:
            frames = frames_filepaths

        return torch.tensor(time[indices], dtype=torch.float), frames

    def get_interval(self, start, end, load_frames=False):
        _, audio_interval = self.get_audio_interval(start, end)
        _, frames_interval = self.get_frames_interval(start, end, load_frames)

        return audio_interval, frames_interval
