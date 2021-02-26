import funcy
import numpy as np
import pydicom
import torch
import torchaudio


class Video:
    def __init__(self, frames_filepaths, audio_filepath, framerate=50, stereo=False):
        audio, self.sample_rate = torchaudio.load(audio_filepath)
        if stereo:
            raise NotImplementedError("Model not implemented for stereo audio.")
        else:
            audio = audio[0]

        self.num_samples, = audio.shape
        audio_duration = self.num_samples / self.sample_rate

        self.framerate = framerate
        self.num_frames = len(frames_filepaths)
        video_duration = self.num_frames / self.framerate

        max_err = 0.05
        diff = abs(video_duration - audio_duration)
        if diff > max_err:
            raise ValueError(f"Difference in duration of audio and video is too large ({diff})")
        self.duration = video_duration

        time = np.linspace(0., self.duration, self.num_samples)
        self.audio = list(zip(time, audio))

        time = np.linspace(0., self.duration, self.num_frames)
        self.frames_filepaths = list(zip(time, frames_filepaths))

    @staticmethod
    def load_frame(filepath):
        ds = pydicom.dcmread(filepath)
        frame = torch.tensor(ds.pixel_array.astype(np.float))
        return frame

    def get_audio_interval(self, start, end):
        on_interval = filter(lambda d: start <= d[0] < end, self.audio)
        audio_interval = torch.tensor([d[1] for d in on_interval])

        return audio_interval

    def get_frames_interval(self, start, end, load_frames=False):
        on_interval = filter(lambda d: start <= d[0] < end, self.frames_filepaths)
        frames_filepaths = [d[1] for d in on_interval]

        if load_frames:
            frames = torch.stack([self.load_frame(fp) for fp in frames_filepaths])
        else:
            frames = frames_filepaths

        return frames

    def get_interval(self, start, end, load_frames=False):
        audio_interval = self.get_audio_interval(start, end)
        frames_interval = self.get_frames_interval(start, end, load_frames)

        return audio_interval, frames_interval
