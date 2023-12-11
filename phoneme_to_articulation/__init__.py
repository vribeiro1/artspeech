####################################################################################################
#
# Code base for the phoneme-to-articulation experiments
#
# Packages:
#
# phoneme_wise_mean_contour - Codebase for the mean contour phoneme-to-articulation
# encoder_decoder - Codebase for the model-free phoneme-to-articulation
# principal_components - Codebase for the autoencoder-based phoneme-to-articulation
# transformer - Codebase for the model-free phoneme-to-articulation with a transformer network
#               (not included in the thesis)
#
####################################################################################################
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from enum import Enum
from functools import lru_cache
from vt_tools import (
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    TONGUE,
    UPPER_LIP,
    UPPER_INCISOR,
)
from vt_tools import UPPER_INCISOR
from vt_tools.bs_regularization import regularize_Bsplines
from vt_shape_gen.helpers import load_articulator_array

from phoneme_to_articulation.tail_clipper import TailClipper
from tract_variables import calculate_vocal_tract_variables

REQUIRED_ARTICULATORS_FOR_TVS = [
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    TONGUE,
    UPPER_LIP,
    UPPER_INCISOR,
]


class RNNType(Enum):
    LSTM = nn.LSTM
    GRU = nn.GRU


@lru_cache(maxsize=None)
def cached_load_articulator_array(filepath, norm_value):
    return torch.from_numpy(load_articulator_array(filepath, norm_value)).type(torch.float)


class InputLoaderMixin:
    @staticmethod
    def prepare_articulator_array(
        datadir,
        subject,
        sequence,
        frame_id,
        articulator,
        dataset_config,
        normalize_fn=None,
        clip_tails=True
    ):
        fp_articulator = os.path.join(
            datadir, subject, sequence, "inference_contours", f"{frame_id}_{articulator}.npy"
        )
        articulator_array = cached_load_articulator_array(
            fp_articulator,
            norm_value=dataset_config.RES
        )

        if clip_tails:
            tail_clip_refs = {}
            tail_clipper = TailClipper(dataset_config)
            for reference in TailClipper.TAIL_CLIP_REFERENCES:
                fp_reference = os.path.join(
                    datadir, subject, sequence, "inference_contours", f"{frame_id}_{reference}.npy"
                )
                reference_array = cached_load_articulator_array(
                    fp_reference,
                    norm_value=dataset_config.RES
                )
                tail_clip_refs[reference.replace("-", "_")] = reference_array

            tail_clip_method_name = f"clip_{articulator.replace('-', '_')}_tails"
            tail_clip_method = getattr(tail_clipper, tail_clip_method_name, None)
            if tail_clip_method:
                articulator_array = tail_clip_method(articulator_array, **tail_clip_refs)

        fp_coord_system_reference = os.path.join(
            datadir, subject, sequence, "inference_contours", f"{frame_id}_{UPPER_INCISOR}.npy"
        )
        coord_system_reference_array = cached_load_articulator_array(
            fp_coord_system_reference,
            norm_value=dataset_config.RES
        )
        coord_system_reference = coord_system_reference_array.T[:, -1]
        coord_system_reference = coord_system_reference.unsqueeze(dim=-1)

        coord_system_reference_array = coord_system_reference_array.T
        coord_system_reference_array = coord_system_reference_array - coord_system_reference
        coord_system_reference_array[0, :] = coord_system_reference_array[0, :] + 0.3
        coord_system_reference_array[1, :] = coord_system_reference_array[1, :] + 0.3

        articulator_array = articulator_array.T
        articulator_array = articulator_array - coord_system_reference
        articulator_array[0, :] = articulator_array[0, :] + 0.3
        articulator_array[1, :] = articulator_array[1, :] + 0.3

        if normalize_fn is not None:
            articulator_array = normalize_fn(articulator_array)

        return articulator_array, coord_system_reference_array


def save_outputs(
    sentences_ids,
    frame_ids,
    outputs,
    targets,
    lengths,
    phonemes,
    articulators,
    save_to,
    regularize_out
):
    """
    Args:
        sentences_ids (str): Unique id of each sentence to save the results.
        frame_ids (List)
        outputs (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
        targets (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
        lengths (List): List with the length of each sentence in the batch.
        phonemes (List): List with the sequence of phonemes for each sentence in the batch.
        articulators (List[str]): List of articulators.
        save_to (str): Path to the directory to save the results.
        regularize_out (bool): If should apply bspline regularization or not.
    """
    for (
        sentence_id,
        sentence_outs,
        sentence_targets,
        length,
        sentence_phonemes,
        sentence_frames
    ) in zip(
        sentences_ids,
        outputs,
        targets,
        lengths,
        phonemes,
        frame_ids
    ):
        phoneme_data = []
        saves_sentence_dir = os.path.join(save_to, sentence_id)
        if not os.path.exists(os.path.join(saves_sentence_dir, "contours")):
            os.makedirs(os.path.join(saves_sentence_dir, "contours"))

        for (
            out,
            target,
            phoneme,
            frame
        ) in zip(
            sentence_outs[:length],
            sentence_targets[:length],
            sentence_phonemes,
            sentence_frames
        ):
            phoneme_data.append({
                "sentence": sentence_id,
                "frame": frame,
                "phoneme": phoneme
            })

            for i_art, art in enumerate(sorted(articulators)):
                pred_art_arr = out[i_art].numpy()
                true_art_arr = target[i_art].numpy()

                if regularize_out:
                    resX, resY = regularize_Bsplines(pred_art_arr.transpose(1, 0), 3)
                    pred_art_arr = np.array([resX, resY])

                pred_npy_filepath = os.path.join(saves_sentence_dir, "contours", f"{frame}_{art}.npy")
                with open(pred_npy_filepath, "wb") as f:
                    np.save(f, pred_art_arr)

                true_npy_filepath = os.path.join(saves_sentence_dir, "contours", f"{frame}_{art}_true.npy")
                with open(true_npy_filepath, "wb") as f:
                    np.save(f, true_art_arr)

            df_filepath = os.path.join(saves_sentence_dir, "phonemes.csv")
            pd.DataFrame(phoneme_data).to_csv(df_filepath, index=False)


def tract_variables(
    sentences_ids,
    frame_ids,
    outputs,
    targets,
    lengths,
    phonemes,
    articulators,
    save_to
):
    """
    Args:
        sentences_ids (str): Unique id of each sentence to save the results.
        frame_ids (List)
        outputs (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
        targets (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
        lengths (List): List with the length of each sentence in the batch.
        phonemes (List): List with the sequence of phonemes for each sentence in the batch.
        articulators (List[str]): List of articulators.
        save_to (str): Path to the directory to save the results.
    """
    for (
        sentence_id,
        sentence_outs,
        sentence_targets,
        length,
        sentence_frames,
        sentence_phonemes
    ) in zip(
        sentences_ids,
        outputs,
        targets,
        lengths,
        frame_ids,
        phonemes
    ):
        saves_sentence_dir = os.path.join(save_to, sentence_id)
        if not os.path.exists(saves_sentence_dir):
            os.makedirs(saves_sentence_dir)

        TVs_data = []
        for (
            out,
            target,
            frame,
            phoneme
        ) in zip(
            sentence_outs[:length],
            sentence_targets[:length],
            sentence_frames,
            sentence_phonemes
        ):
            pred_input_dict = {
                art: tensor.T for art, tensor in zip(articulators, out)
            }
            pred_TVs = calculate_vocal_tract_variables(pred_input_dict)

            target_input_dict = {
                art: tensor.T for art, tensor in zip(articulators, target)
            }
            target_TVs = calculate_vocal_tract_variables(target_input_dict)

            item = {
                "sentence": sentence_id,
                "frame": frame,
                "phoneme": phoneme
            }

            for TV, TV_dict in target_TVs.items():
                if TV_dict is None:
                    continue

                item.update({
                    f"{TV}_target": TV_dict["value"],
                    f"{TV}_target_poc_1_x": TV_dict["poc_1"][0].item(),
                    f"{TV}_target_poc_1_y": TV_dict["poc_1"][1].item(),
                    f"{TV}_target_poc_2_x": TV_dict["poc_2"][0].item(),
                    f"{TV}_target_poc_2_y": TV_dict["poc_2"][1].item()
                })

            for TV, TV_dict in pred_TVs.items():
                if TV_dict is None:
                    continue

                item.update({
                    f"{TV}_pred": TV_dict["value"],
                    f"{TV}_pred_poc_1_x": TV_dict["poc_1"][0].item(),
                    f"{TV}_pred_poc_1_y": TV_dict["poc_1"][1].item(),
                    f"{TV}_pred_poc_2_x": TV_dict["poc_2"][0].item(),
                    f"{TV}_pred_poc_2_y": TV_dict["poc_2"][1].item()
                })

            TVs_data.append(item)

        filepath = os.path.join(saves_sentence_dir, "tract_variables.csv")
        df = pd.DataFrame(TVs_data)
        df.to_csv(filepath, index=False)
