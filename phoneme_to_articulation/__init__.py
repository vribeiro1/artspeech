import numpy as np
import os
import pandas as pd

from vt_tools import (
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    TONGUE,
    UPPER_LIP,
    UPPER_INCISOR,
)
from vt_tools.bs_regularization import regularize_Bsplines

from tract_variables import calculate_vocal_tract_variables

REQUIRED_ARTICULATORS_FOR_TVS = [
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    TONGUE,
    UPPER_LIP,
    UPPER_INCISOR,
]


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
