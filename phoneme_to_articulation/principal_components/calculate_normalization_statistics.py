import argparse
import numpy as np
import os
import torch
import yaml

from tqdm import tqdm
from vt_shape_gen.helpers import load_articulator_array
from vt_tools import *

from helpers import sequences_from_dict
from phoneme_to_articulation.principal_components.dataset import collect_data
from phoneme_to_articulation.tail_clipper import TailClipper
from settings import DatasetConfig


def prepare_articulator_array(datadir, subject, sequence, frame_number, articulator):
    fp_articulator = os.path.join(
        datadir, subject, sequence, "inference_contours", f"{frame_number}_{articulator}.npy"
    )

    articulator_array = load_articulator_array(
        fp_articulator,
        norm_value=DatasetConfig.RES
    )

    tail_clip_refs = {}
    for reference in TailClipper.TAIL_CLIP_REFERENCES:
        fp_reference = os.path.join(
            datadir, subject, sequence, "inference_contours", f"{frame_number}_{reference}.npy"
        )

        reference_array = load_articulator_array(
            fp_reference,
            norm_value=DatasetConfig.RES
        )

        tail_clip_refs[reference.replace("-", "_")] = reference_array

    tail_clip_method_name = f"clip_{articulator.replace('-', '_')}_tails"
    tail_clip_method = getattr(TailClipper, tail_clip_method_name, None)

    if tail_clip_method:
        articulator_array = tail_clip_method(articulator_array, **tail_clip_refs)

    articulator_array = articulator_array.T

    fp_coord_system_reference = os.path.join(
        datadir, subject, sequence, "inference_contours", f"{frame_number}_{UPPER_INCISOR}.npy"
    )

    coord_system_reference_array = load_articulator_array(
        fp_coord_system_reference,
        norm_value=DatasetConfig.RES
    )
    coord_system_reference = coord_system_reference_array.T[:, -1]
    coord_system_reference = coord_system_reference.unsqueeze(dim=-1)

    articulator_array = articulator_array - coord_system_reference
    articulator_array[0, :] = articulator_array[0, :] + 0.3
    articulator_array[1, :] = articulator_array[1, :] + 0.3

    return articulator_array


def main(cfg):
    datadir = cfg["datadir"]
    sequences_dict = cfg["sequences_dict"]
    sequences = sequences_from_dict(sequences_dict)
    articulators = cfg["articulators"]
    save_to = cfg["save_to"]

    sentence_data = collect_data(datadir, sequences, DatasetConfig.SYNC_SHIFT, DatasetConfig.FRAMERATE)
    data = []
    for sentence in sentence_data:
        for frame_id, phoneme in zip(sentence["frame_ids"], sentence["phonemes"]):
            data.append({
                "subject": sentence["subject"],
                "sequence": sentence["sequence"],
                "frame_id": frame_id,
                "phoneme": phoneme
            })

    for articulator in articulators:
        articulator_arrays = torch.zeros(0, 2, 50)
        for item in tqdm(data):
            subject = item["subject"]
            sequence = item["sequence"]
            frame_id = item["frame_id"]

            articulator_array = prepare_articulator_array(
                datadir, subject, sequence, frame_id, articulator
            ).unsqueeze(dim=0)

            articulator_arrays = torch.concat([articulator_arrays, articulator_array], dim=0)

        articulator_mean = articulator_arrays.mean(dim=0).numpy()
        articulator_std = articulator_arrays.mean(dim=0).numpy()

        np.save(os.path.join(save_to, f"{articulator}_mean.npy"), articulator_mean)
        np.save(os.path.join(save_to, f"{articulator}_std.npy"), articulator_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(cfg)
