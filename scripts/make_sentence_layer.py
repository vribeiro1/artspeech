import pdb

import os

from glob import glob
from tgt.core import TextGrid, IntervalTier, Interval
from tgt.io3 import read_textgrid, write_to_file
from tqdm import tqdm

EMPTY = ""
SIL = "#"
LONG_SIL_MAX_LENGTH = 1.5
SHORT_SIL_MAX_LENGTH = 0.6

datadir = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/vsouzaribeiro/datasets/TextGridCorrigesJustine2022"
textgrid_filepaths = sorted(glob(os.path.join(datadir, "*", "S*", "vol_*_*.TextGrid")))

for textgrid_filepath in tqdm(textgrid_filepaths):
    textgrid = read_textgrid(textgrid_filepath, include_empty_intervals=True)
    word_tier = textgrid.get_tier_by_name("WordTier")
    phon_tier = textgrid.get_tier_by_name("PhonTier")
    sentence_tier = textgrid.get_tier_by_name("SentenceTier")

    merged_word_tier = word_tier.get_copy_with_same_intervals_merged()
    merged_phon_tier = phon_tier.get_copy_with_same_intervals_merged()

    new_textgrid = TextGrid(filename=textgrid.filename)
    long_sentence_tier = IntervalTier(
        start_time=sentence_tier.start_time,
        end_time=sentence_tier.end_time,
        name="LongSentenceTier"
    )
    short_sentence_tier = IntervalTier(
        start_time=sentence_tier.start_time,
        end_time=sentence_tier.end_time,
        name="ShortSentenceTier"
    )
    long_word_tier = IntervalTier(
        start_time=word_tier.start_time,
        end_time=word_tier.end_time,
        name="LongWordTier"
    )
    new_word_tier = IntervalTier(
        start_time=word_tier.start_time,
        end_time=word_tier.end_time,
        name="WordTier"
    )
    new_phon_tier = IntervalTier(
        start_time=phon_tier.start_time,
        end_time=phon_tier.end_time,
        name="PhonTier"
    )

    for i, interval in enumerate(merged_word_tier):
        if i == 0 or i == len(merged_word_tier) - 1:
            new_word_tier.add_interval(interval)
            continue

        if interval.text != SIL:
            new_word_tier.add_interval(interval)
            continue

        sil_length = interval.end_time - interval.start_time
        if sil_length <= SHORT_SIL_MAX_LENGTH:
            new_word_tier.add_interval(interval)
            continue

        interval_1 = Interval(
            start_time=interval.start_time,
            end_time=interval.start_time + sil_length / 3,
            text=SIL
        )
        interval_2 = Interval(
            start_time=interval_1.end_time,
            end_time=interval_1.end_time + sil_length / 3,
            text=EMPTY
        )
        interval_3 = Interval(
            start_time=interval_2.end_time,
            end_time=interval.end_time,
            text=SIL
        )
        new_word_tier.add_intervals([
            interval_1,
            interval_2,
            interval_3
        ])

    for i, interval in enumerate(merged_word_tier):
        if i == 0 or i == len(merged_word_tier) - 1:
            long_word_tier.add_interval(interval)
            continue

        if interval.text != SIL:
            long_word_tier.add_interval(interval)
            continue

        sil_length = interval.end_time - interval.start_time
        if sil_length <= LONG_SIL_MAX_LENGTH:
            long_word_tier.add_interval(interval)
            continue

        interval_1 = Interval(
            start_time=interval.start_time,
            end_time=interval.start_time + sil_length / 3,
            text=SIL
        )
        interval_2 = Interval(
            start_time=interval_1.end_time,
            end_time=interval_1.end_time + sil_length / 3,
            text=EMPTY
        )
        interval_3 = Interval(
            start_time=interval_2.end_time,
            end_time=interval.end_time,
            text=SIL
        )
        long_word_tier.add_intervals([
            interval_1,
            interval_2,
            interval_3
        ])

    for i, interval in enumerate(merged_phon_tier):
        if i == 0 or i == len(merged_phon_tier) - 1:
            new_phon_tier.add_interval(interval)
            continue

        if interval.text != SIL:
            new_phon_tier.add_interval(interval)
            continue

        sil_length = interval.end_time - interval.start_time
        if sil_length <= SHORT_SIL_MAX_LENGTH:
            new_phon_tier.add_interval(interval)
            continue

        interval_1 = Interval(
            start_time=interval.start_time,
            end_time=interval.start_time + sil_length / 3,
            text=SIL
        )
        interval_2 = Interval(
            start_time=interval_1.end_time,
            end_time=interval_1.end_time + sil_length / 3,
            text=EMPTY
        )
        interval_3 = Interval(
            start_time=interval_2.end_time,
            end_time=interval.end_time,
            text=SIL
        )
        new_phon_tier.add_intervals([
            interval_1,
            interval_2,
            interval_3
        ])

    sentence_intervals = []
    for i, interval in enumerate(new_word_tier):
        if interval.text == EMPTY:
            if len(sentence_intervals) > 0:
                text = " ".join([
                    interval.text for interval in sentence_intervals
                    if interval.text != SIL
                ]).strip()
                new_interval = Interval(
                    start_time=sentence_intervals[0].start_time,
                    end_time=sentence_intervals[-1].end_time,
                    text=text
                )
                short_sentence_tier.add_interval(new_interval)
                sentence_intervals = []
            short_sentence_tier.add_interval(interval)
        else:
            sentence_intervals.append(interval)

    if len(sentence_intervals) > 0:
        text = " ".join([
            interval.text for interval in sentence_intervals
            if interval.text != SIL
        ]).strip()
        new_interval = Interval(
            start_time=sentence_intervals[0].start_time,
            end_time=sentence_intervals[-1].end_time,
            text=text
        )
        short_sentence_tier.add_interval(new_interval)

    sentence_intervals = []
    for i, interval in enumerate(long_word_tier):
        if interval.text == EMPTY:
            if len(sentence_intervals) > 0:
                text = " ".join([
                    interval.text for interval in sentence_intervals
                    if interval.text != SIL
                ]).strip()
                new_interval = Interval(
                    start_time=sentence_intervals[0].start_time,
                    end_time=sentence_intervals[-1].end_time,
                    text=text
                )
                long_sentence_tier.add_interval(new_interval)
                sentence_intervals = []
            long_sentence_tier.add_interval(interval)
        else:
            sentence_intervals.append(interval)

    if len(sentence_intervals) > 0:
        text = " ".join([
            interval.text for interval in sentence_intervals
            if interval.text != SIL
        ]).strip()
        new_interval = Interval(
            start_time=sentence_intervals[0].start_time,
            end_time=sentence_intervals[-1].end_time,
            text=text
        )
        long_sentence_tier.add_interval(new_interval)

    new_textgrid.add_tier(long_sentence_tier)
    new_textgrid.add_tier(short_sentence_tier)
    new_textgrid.add_tier(new_word_tier)
    new_textgrid.add_tier(new_phon_tier)

    new_textgrid_filepath = textgrid_filepath.replace(
        "TextGridCorrigesJustine2022",
        "TextGridCorrigesJustine2022_Adjusted"
    )
    if not os.path.exists(os.path.dirname(new_textgrid_filepath)):
        os.makedirs(os.path.dirname(new_textgrid_filepath))

    write_to_file(new_textgrid, new_textgrid_filepath, format="long")
