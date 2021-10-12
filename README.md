# Towards the prediction of the vocal tract shape from the sequence of phonemes to be articulated

In this work, we address the prediction of speech articulators’ temporal geometric position from the sequence of phonemes to be articulated. We start from a set of real-time MRI sequences uttered by a female French speaker. The contours of five articulators were tracked automatically in each of the frames in the MRI video. Then, we explore the capacity of a bidirectional GRU to correctly predict each articulator’s shape and position given the sequence of phonemes and their duration. We propose a 5-fold cross-validation experiment to evaluate the generalization capacity of the model. In a second experiment, we evaluate our model’s data efficiency by reducing training data. We evaluate the point-to-point Euclidean distance and the Pearson’s correlations along time between the predicted and the target shapes. We also evaluate produced shapes of the critical articulators of specific phonemes. We show that our model can achieve good results with minimal data, producing very realistic vocal tract shapes.

Published at <a href="https://www.isca-speech.org/archive/interspeech_2021/ribeiro21b_interspeech.html">Interspeech 2021</a>

Cite as: Ribeiro, V., Isaieva, K., Leclere, J., Vuissoz, P.-A., Laprie, Y. (2021) Towards the Prediction of the Vocal Tract Shape from the Sequence of Phonemes to be Articulated. Proc. Interspeech 2021, 3325-3329, doi: 10.21437/Interspeech.2021-184

```bibtex
@inproceedings{ribeiro21b_interspeech,
  author={Vinicius Ribeiro and Karyna Isaieva and Justine Leclere and Pierre-André Vuissoz and Yves Laprie},
  title={{Towards the Prediction of the Vocal Tract Shape from the Sequence of Phonemes to be Articulated}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={3325--3329},
  doi={10.21437/Interspeech.2021-184}
}
```
