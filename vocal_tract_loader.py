import os
import torch

from functools import lru_cache
from vt_shape_gen.helpers import load_articulator_array
from vt_tools import UPPER_INCISOR

from phoneme_to_articulation.tail_clipper import TailClipper


@lru_cache(maxsize=None)
def cached_load_articulator_array(filepath, norm_value):
    return torch.from_numpy(load_articulator_array(filepath, norm_value)).type(torch.float)


class VocalTractShapeLoader:
    def __init__(self, datadir, articulators, num_samples, dataset_config, clip_tails=True):
        self.datadir = datadir
        self.articulators = articulators
        self.num_articulators = len(articulators)
        self.num_samples = num_samples
        self.x_center = 0.3
        self.y_center = 0.3
        self.dataset_config = dataset_config
        self.clip_tails = clip_tails

    def get_frame_coordinate_system_reference(self, subject, sequence, frame_id):
        fp_coord_system_reference = os.path.join(
            self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{UPPER_INCISOR}.npy"
        )

        coord_system_reference_array = cached_load_articulator_array(
            fp_coord_system_reference,
            norm_value=self.dataset_config.RES
        ).T

        return coord_system_reference_array

    def prepare_articulator_array(self, subject, sequence, frame_id, articulator):
        fp_articulator = os.path.join(
            self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{articulator}.npy"
        )

        articulator_array = cached_load_articulator_array(
            fp_articulator,
            norm_value=self.dataset_config.RES
        )

        if self.clip_tails:
            tail_clip_refs = {}
            tail_clipper = TailClipper(self.dataset_config)
            for reference in TailClipper.TAIL_CLIP_REFERENCES:
                fp_reference = os.path.join(
                    self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{reference}.npy"
                )

                reference_array = cached_load_articulator_array(
                    fp_reference,
                    norm_value=self.dataset_config.RES
                )
                tail_clip_refs[reference.replace("-", "_")] = reference_array

            tail_clip_method_name = f"clip_{articulator.replace('-', '_')}_tails"
            tail_clip_method = getattr(tail_clipper, tail_clip_method_name, None)

            if tail_clip_method:
                articulator_array = tail_clip_method(articulator_array, **tail_clip_refs)

        articulator_array = articulator_array.T

        return articulator_array

    def load_vocal_tract_shapes(self, subject, sequence, frame_ids, skip_missing=False):
        sentence_targets = torch.zeros(size=(0, self.num_articulators, 2, self.num_samples))
        sentence_references = torch.zeros(size=(0, 2, self.num_samples))
        for frame_id in frame_ids:
            coord_system_reference_array = self.get_frame_coordinate_system_reference(
                subject, sequence, frame_id
            )
            coord_system_reference = coord_system_reference_array[:, -1]
            coord_system_reference = coord_system_reference.unsqueeze(dim=-1)

            coord_system_reference_array = coord_system_reference_array - coord_system_reference
            coord_system_reference_array[0, :] = coord_system_reference_array[0, :] + self.x_center
            coord_system_reference_array[1, :] = coord_system_reference_array[1, :] + self.y_center

            try:
                frame_targets = torch.stack(
                    [
                        self.prepare_articulator_array(
                            subject,
                            sequence,
                            frame_id,
                            articulator
                        )
                        for articulator in self.articulators
                    ],
                    dim=0
                ).unsqueeze(dim=0)
            except FileNotFoundError as e:
                if skip_missing:
                    continue
                else:
                    raise e

            frame_targets = frame_targets - coord_system_reference
            frame_targets[..., 0, :] = frame_targets[..., 0, :] + 0.3
            frame_targets[..., 1, :] = frame_targets[..., 1, :] + 0.3

            sentence_targets = torch.cat([sentence_targets, frame_targets], dim=0)
            coord_system_reference_array = coord_system_reference_array.unsqueeze(dim=0)
            sentence_references = torch.cat([sentence_references, coord_system_reference_array], dim=0)

        sentence_length = len(frame_ids)
        sentence_targets = sentence_targets.type(torch.float)
        sentence_references = sentence_references.type(torch.float)

        return sentence_targets, sentence_references, sentence_length
