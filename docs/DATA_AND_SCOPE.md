# Data, evaluation, and project scope

The course material for ML for Health Lab 3 (Politecnico di Torino, 2025/26) provided 54 labeled JPEG images: 11 filenames beginning with `low_risk_`, 16 with `medium_risk_`, and 27 with `melanoma_`. These category names are inherited dataset identifiers **not** inferred by this project. The uploaded archive also contained two additional altered images.

Six JPEGs are included here as small reproducibility examples. The full supplied image set is not republished in the GitHub package. The original file `lab3/moles.zip` from the user's private course archive can be extracted locally and passed to `--input-dir` to experiment with the full set.

Because the dataset was distributed as part of a course and no public reuse license was provided, verify redistribution rights before publishing the image files or derived photographic demonstration figures. It is possible to publish the code and processing diagram without redistributing the photographs.

The pipeline outputs *heuristic candidate region masks*. These images are not medical predictions. No pixel-level ground-truth masks were provided; a non-empty mask is not equivalent to an accurate segmentation. No sensitivity, specificity, Dice, or IoU against expert annotations is claimed.

The original experiment scripts are preserved without rewriting their content. The modular pipeline and its automated tests are supplemental engineering work for clearer reproducibility and portfolio presentation.
