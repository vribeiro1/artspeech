import pdb
import logging
import torchaudio
import os

from glob import glob
from tempfile import NamedTemporaryFile
from tgt.io import read_textgrid
from tqdm import tqdm

from video import Video
from settings import BASE_DIR, ArtSpeechConfig, GottingenConfig


class DatabaseCollector:
    sentence_tier = "SentenceTier"
    word_tier = "WordTier"
    phoneme_tier = "PhonTier"

    def __init__(self, datadir, save_audio_dir=None):
        self.datadir=datadir
        self.save_audio_dir = save_audio_dir

    @staticmethod
    def _has_all_articulators(frame_ids, required_articulators):
        if required_articulators is None:
            return True

        has_all_required_articulators = all([
            all([
                os.path.exists(
                    os.path.join(
                        sequence_dir,
                        "inference_contours",
                        f"{frame_id}_{articulator}.npy"
                    )
                ) for articulator in required_articulators
            ]) for frame_id in frame_ids
        ])
        return has_all_required_articulators

    def get_sequence_dir(self, subject, sequence):
        raise NotImplementedError

    def get_wav_filepath(self, subject, sequence):
        raise NotImplementedError

    def get_textgrid_filepath(self, subject, sequence):
        raise NotImplementedError

    def get_frame_ids(self, subject, sequence):
        sequence_dir = self.get_sequence_dir(subject, sequence)
        # Collect all of the articulators files
        articulators_filepaths = glob(os.path.join(sequence_dir, "inference_contours", "*.npy"))
        # Since each frame have more than one articulator, we extract the frame ids and remove
        # repetitions.
        articulators_basenames = map(os.path.basename, articulators_filepaths)
        articulators_filenames = map(lambda s: s.split(".")[0], articulators_basenames)
        frame_ids = sorted(set(map(lambda s: s.split("_")[0], articulators_filenames)))
        return frame_ids

    def _save_sentence_audio_interval(self, video, sentence_interval):
        _, sentence_audio_interval = video.get_audio_interval(
            sentence_interval.start_time,
            sentence_interval.end_time
        )
        sentence_audio_interval = sentence_audio_interval.unsqueeze(dim=0)

        with NamedTemporaryFile(dir=self.save_audio_dir, suffix=".wav", delete=False) as f:
            sentence_wav_filepath = f.name
            torchaudio.save(
                f.name,
                sentence_audio_interval,
                video.sample_rate,
            )

        return sentence_wav_filepath

    def collect_data(self, sequences, required_articulators=None):
        data = []
        for subject, sequence in tqdm(sequences, desc="Collecting data"):
            sequence_dir = self.get_sequence_dir(subject, sequence)
            frame_ids = self.get_frame_ids(subject, sequence)
            if len(frame_ids) == 0:
                logging.warning(f"Skipping {subject}/{sequence} - Empty frame sequence")
                continue

            textgrid_filepath = self.get_textgrid_filepath(subject, sequence)
            if not os.path.isfile(textgrid_filepath):
                logging.warning(f"Skipping {subject}/{sequence} - Missing textgrid")
                continue
            textgrid = read_textgrid(textgrid_filepath)
            phone_tier = textgrid.get_tier_by_name(self.phoneme_tier)
            sentence_tier = textgrid.get_tier_by_name(self.sentence_tier)

            wav_filepath = self.get_wav_filepath(subject, sequence)
            video = Video(
                frames_filepaths=frame_ids[self.dataset_config.SYNC_SHIFT:],
                audio_filepath=wav_filepath,
                framerate=self.dataset_config.FRAMERATE,
                max_diff=1.0
            )

            for sentence_interval in sentence_tier.intervals:
                sentence_wav_filepath = wav_filepath
                if self.save_audio_dir is not None:
                    sentence_wav_filepath = self._save_sentence_audio_interval(
                        video,
                        sentence_interval
                    )

                def phone_is_in_interval(phone):
                    return (
                        phone.start_time >= sentence_interval.start_time and
                        phone.end_time <= sentence_interval.end_time
                    )

                sentence_phone_intervals = filter(phone_is_in_interval, phone_tier)
                sentence_phone_intervals = sorted(
                    sentence_phone_intervals,
                    key=lambda interval: interval.start_time
                )
                sentence_phonemes_with_time = []
                sentence_phonemes = []
                sentence_frame_ids = []
                for phone_interval in sentence_phone_intervals:
                    _, phoneme_frame_ids = video.get_frames_interval(
                        phone_interval.start_time,
                        phone_interval.end_time
                    )
                    repeated_phoneme = [phone_interval.text] * len(phoneme_frame_ids)
                    sentence_frame_ids.extend(phoneme_frame_ids)
                    sentence_phonemes.extend(repeated_phoneme)
                    # Since the sentences are split into smaller wav files, the phoneme onset
                    # and offset need to be adjusted
                    sentence_phonemes_with_time.append((
                        phone_interval.text,
                        phone_interval.start_time - sentence_interval.start_time,
                        phone_interval.end_time - sentence_interval.start_time
                    ))

                start_str = "%.04f" % sentence_interval.start_time
                end_str = "%.04f" % sentence_interval.end_time
                sentence_name = f"{subject}_{sequence}-{start_str}_{end_str}"

                data.append({
                    "subject": subject,
                    "sequence": sequence,
                    "sentence_name": sentence_name,
                    "wav_filepath": sentence_wav_filepath,
                    "audio_duration": sentence_interval.end_time - sentence_interval.start_time,
                    "textgrid_filepath": textgrid_filepath,
                    "n_frames": len(sentence_frame_ids),
                    "frame_ids": sentence_frame_ids,
                    "phonemes_with_time": sentence_phonemes_with_time,
                    "phonemes": sentence_phonemes,
                    "has_all": self._has_all_articulators(sentence_frame_ids, required_articulators)
                })

        return data


class ArtSpeechDatabase2Collector(DatabaseCollector):
    dataset_config = ArtSpeechConfig
    long_sentence_tier = "LongSentenceTier"
    short_sentence_tier = "ShortSentenceTier"

    def __init__(self, datadir, sentence_tier="long"):
        super().__init__(datadir)

        self.sentence_tier = (
            self.short_sentence_tier if sentence_tier == "short"
            else self.long_sentence_tier
        )

    def get_sequence_dir(self, subject, sequence):
        return os.path.join(self.datadir, subject, sequence)

    def get_wav_filepath(self, subject, sequence):
        sequence_dir = self.get_sequence_dir(subject, sequence)
        wav_filepath = os.path.join(sequence_dir, f"vol_{subject}_{sequence}.wav")
        return wav_filepath

    def get_textgrid_filepath(self, subject, sequence):
        sequence_dir = self.get_sequence_dir(subject, sequence)
        textgrid_filepath = os.path.join(sequence_dir, f"vol_{subject}_{sequence}.TextGrid")
        return textgrid_filepath

    def get_frame_ids(self, subject, sequence):
        sequence_dir = self.get_sequence_dir(subject, sequence)
        # Collect all of the articulators files
        frame_filepaths = glob(os.path.join(sequence_dir, "NPY_MR", "*.npy"))
        frame_ids = sorted(map(lambda s: s.split(".")[0], map(os.path.basename, frame_filepaths)))
        return frame_ids


class GottingenDatabaseCollector(DatabaseCollector):
    dataset_config = GottingenConfig

    def get_sequence_dir(self, subject, sequence):
        return os.path.join(self.datadir, subject, sequence)

    def get_wav_filepath(self, subject, sequence):
        sequence_dir = self.get_sequence_dir(subject, sequence)
        wav_filepath = os.path.join(sequence_dir, f"vol_{subject}_{sequence}.wav")
        return wav_filepath

    def get_textgrid_filepath(self, subject, sequence):
        sequence_dir = self.get_sequence_dir(subject, sequence)
        textgrid_filepath = os.path.join(sequence_dir, f"vol_{subject}_{sequence}.textgrid")
        return textgrid_filepath


DATABASE_COLLECTORS = {
    "artspeech2": ArtSpeechDatabase2Collector,
    "gottingen": GottingenDatabaseCollector,
}
