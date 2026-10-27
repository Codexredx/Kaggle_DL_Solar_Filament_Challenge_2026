# Solar Filament Segmentation Using Deep Learning

This project uses **PyTorch and U-Net** to identify solar filaments in grayscale H-alpha telescope images. It learns which image pixels belong to filaments, converts those predictions into **separate filament instances**, evaluates their quality, and exports predictions in the format expected by the Kaggle Solar Filament Segmentation Challenge 2026.

## How the system works

```text
Grayscale solar image (2048 x 2048)
             |
             v
Image normalization and training crops       COCO polygon annotations
             |                                         |
             |                              Polygon-to-mask conversion
             |                                         |
             +------------ Training ------------------+
                              |
                              v
                       U-Net network
                              |
                              v
                   Pixel probability map
                              |
                              v
           Thresholding and instance separation
                              |
                              v
              Individual filament masks
                       /             \
                      v               v
           Validation PQ       Compressed COCO RLE
                                      |
                                      v
                               submission.csv
```

The neural network performs **semantic segmentation**: it predicts foreground (filament) versus background for each pixel. The post-processing stage then performs **instance separation**, assigning different IDs to distinct filaments. This distinction matters because identifying the correct foreground pixels does not necessarily mean the model correctly separates neighboring filaments or preserves a filament's continuity.

## Dataset and annotations

The input images are **2048 x 2048 grayscale JPEGs** from GONG H-alpha observations. Training annotations are stored in a COCO-style JSON file. Each annotated filament has a polygon describing its shape and may also have a bounding box, spine, and category.

The code uses the polygon boundaries to create training masks. The current implementation does **not** train a separate classifier for the Left, Right, Unidentifiable, and Ambiguous annotation categories. It also does **not** use the annotated spines as separate training targets.

Expected dataset layout:

```text
MAGFiLO_1.0_Kaggle_2026/
|-- train/
|   |-- train_images/
|   `-- MAGFiLO_1.0_Annotations_kaggle2026_train.json
`-- test/
    `-- test_images/
```

Test images do not contain ground-truth annotations. Predictions for these images are generated only after training on the labeled training data.

### Mask generation

`data.py` reads the JSON and converts every filament polygon into a binary mask using `pycocotools`. The masks belonging to an annotated image entry are combined into a single **instance label map**:

- `0`: background
- `1`: first annotated filament
- `2`: second annotated filament
- Additional integer IDs: additional filaments

These masks are cached as lossless PNG files under `masks/` so that training does not need to rasterize polygons repeatedly. `manifest.csv` records image paths, mask paths, observation IDs, acquisition times, instrument identifiers, and whether each entry belongs to the train or test collection.

Multiple annotators can label the same underlying observation. The code separates training and validation at the **observation level** using `time_block_split()`, which keeps annotation versions of one observation together and excludes nearby observations around selected validation blocks. The `--split-seed` option controls this split and should remain unchanged when comparing model variants.

### Image preprocessing and augmentation

`load_image()` converts the image into a floating-point grayscale array. It uses the 1st and 99.5th percentiles to normalize intensities approximately into `[0, 1]`, reducing the effect of unusually dark or bright observations.

Training uses smaller crops, typically `512 x 512`, rather than processing an entire `2048 x 2048` image at once. `random_crop()` preferentially samples crops containing filament pixels, because the background occupies most of each solar image. `augment()` applies matching flips and right-angle rotations to images and masks, followed by image-only changes in brightness, contrast, blur, and noise. These augmentations improve the variety of examples presented to the model.

## Neural networks (`models.py`)

Two models are supported:

| Model | Implementation | Purpose |
| --- | --- | --- |
| `smp` | U-Net with an optional ImageNet-pretrained ResNet34 encoder, using `segmentation-models-pytorch` | Main segmentation model |
| `light` | Smaller U-Net implemented directly in PyTorch | Lightweight baseline or student model for distillation |

A U-Net contains an **encoder** that extracts image features at progressively coarser resolutions and a **decoder** that reconstructs a full-resolution prediction. **Skip connections** transfer spatial detail from encoder layers to decoder layers. This is useful for thin filament structures that might otherwise be lost during downsampling.

Both models take a one-channel image as input and output **one logit per pixel**. Applying a sigmoid converts each logit into a foreground probability between 0 and 1.

`predict_prob()` processes full-size images in overlapping tiles. Tile predictions are averaged where they overlap, helping keep GPU memory requirements manageable while producing a full-image probability map.

## Loss functions (`losses.py`)

The training objective combines losses that measure different aspects of segmentation:

| Loss | What it measures |
| --- | --- |
| Binary cross-entropy (BCE) | Whether each pixel is correctly classified as filament or background |
| Dice loss | How strongly the predicted foreground overlaps the annotated foreground |
| clDice loss | How well the predicted thin structures overlap along their approximate skeletons |
| Distillation loss | How closely a smaller student model matches the soft predictions of a teacher model |

The default baseline is:

```text
Loss = BCE + Dice
```

When `--cldice 0.5` is enabled, the training objective becomes:

```text
Loss = BCE + Dice + 0.5 * clDice
```

clDice is an optional experiment intended to improve filament continuity. It does not guarantee a better score and should be evaluated against the same validation split as the baseline.

## Training (`train.py`)

`FilamentDS` loads an image and its cached instance mask, converts the instance mask to **binary foreground/background labels**, selects a crop, applies augmentation, and returns PyTorch tensors.

Each training iteration performs these operations:

1. Pass the image batch through U-Net to obtain pixel logits.
2. Compare logits with ground-truth foreground masks using the selected loss.
3. Use backpropagation to calculate gradients for the model weights.
4. Update the weights with the AdamW optimizer.
5. Update the learning rate using a short warmup followed by cosine decay.

On a CUDA GPU, training uses automatic mixed precision to reduce memory consumption and potentially improve speed.

After every `--val-every` epochs, the model predicts on the validation images. Those predictions are converted into instances and evaluated with Panoptic Quality. The checkpoint with the highest validation PQ is saved as:

```text
runs/<experiment>/seed<seed>/best.pt
```

The corresponding `summary.json` contains the training history, best validation PQ, model size, training duration, and configuration. The saved checkpoint is a model state dictionary; loading it requires the matching architecture configuration.

**Training seed versus split seed:** `--seed` changes the training initialization and sampling randomness. `--split-seed` controls which observations are used for validation. Keeping the split seed fixed makes experiments more directly comparable.

## Turning predictions into individual filaments (`postprocess.py`)

U-Net predicts one foreground probability map, not one mask per individual filament. `prob_to_instances()` separates these predictions through several steps:

1. **Hysteresis thresholding:** retain high-confidence pixels and lower-confidence pixels connected to them, helping recover faint filament extensions.
2. **Morphological closing:** fill small gaps in the foreground mask, potentially repairing broken structures.
3. **Small-object filtering:** remove predicted regions below `min_area`, reducing isolated false positives.
4. **Connected components:** assign separate integer labels to disconnected filament regions.
5. **Optional watershed:** attempt to split touching foreground regions into separate instances.

The parameters `low`, `high`, `close_px`, `min_area`, and optional watershed settings control the trade-off between missed filament pixels, fragmentation, and incorrectly merged filaments.

`tune()` searches combinations of post-processing settings using labeled validation images. Post-processing should be selected using validation performance, not the hidden test labels.

## Evaluating segmentation (`pq_metric.py`)

The key local metric is **Panoptic Quality (PQ)**, an instance-level metric. A predicted instance is treated as matched to a ground-truth filament when their intersection over union (IoU) exceeds `0.5`.

```text
IoU = overlap area / union area

PQ = sum(IoU of matched pairs) / (TP + 0.5 * FP + 0.5 * FN)
```

- **TP:** predicted instances matched to ground-truth instances
- **FP:** unmatched predicted instances
- **FN:** ground-truth instances without a match
- **SQ:** average IoU of matched instances
- **RQ:** how successfully instances were recognized
- **PQ:** combined segmentation and recognition quality, equal to `SQ * RQ`

The metric also reports diagnostic counts for fragmentation and merging. These diagnostics help explain why a foreground mask may look reasonable while its instance-level score is low.

The local metric is a standard PQ implementation with additional diagnostics. Its numerical results should be checked against the competition's official evaluation method before assuming identical Kaggle scores.

## Predicting test images (`infer.py`)

Inference loads a trained checkpoint and performs the following process for every test image:

1. Load and normalize the grayscale image.
2. Predict foreground probabilities using overlapping tiles.
3. Optionally average predictions from flipped images using test-time augmentation (TTA).
4. Convert the probability map into separately labeled filament instances.
5. Encode each nonempty filament mask as compressed COCO run-length encoding (RLE).
6. Write one CSV row for each predicted filament.

The output file has this schema:

```csv
filament_id,segmentation_rle
```

A filament ID combines the observation identifier and a sequential filament number, such as `20160920230134Lh_1`. The `segmentation_rle` column contains the compressed mask representation, not raw pixel values or ordinary start-length pairs.

`submission_check.py` verifies the CSV column names, uniqueness and structure of IDs, and whether a sample of encoded masks can be decoded successfully. This validates the local file structure but does not replace validation by Kaggle.

## Model compression (`compress.py`)

The project tests whether a trained model can use fewer resources while preserving segmentation quality:

- **Magnitude pruning:** zero a chosen fraction of small convolution weights, optionally fine-tune, and re-evaluate PQ.
- **FP16 evaluation:** compare model storage size, network speed, and PQ using half-precision weights on supported GPUs.
- **Knowledge distillation:** train `LightUNet` to match a trained teacher's outputs, using `train.py --teacher` and `--model light`.

The compression comparison records model parameters, sparsity, estimated storage size, network-only inference time, and validation PQ. **Unstructured pruning does not automatically reduce dense inference time or checkpoint size**, so performance must be measured rather than inferred from the sparsity percentage.

## Code map

| File | Role |
| --- | --- |
| `data.py` | Image loading, normalization, COCO polygon conversion, masks, splits and augmentation |
| `models.py` | Pretrained U-Net, lightweight U-Net and tiled full-image prediction |
| `losses.py` | BCE, Dice, clDice and distillation losses |
| `train.py` | Training loop, learning-rate schedule, validation and checkpoints |
| `postprocess.py` | Turn foreground predictions into separate filament instances |
| `pq_metric.py` | Local PQ scoring and split/merge diagnostics |
| `infer.py` | Test-image inference, optional tuning and Kaggle CSV creation |
| `submission_check.py` | Validate the generated prediction CSV |
| `compress.py` | Pruning and FP16 comparisons |
| `timing_probe.py` | Estimate training and inference time on the available hardware |

## Generated outputs

| Output | Description |
| --- | --- |
| `masks/` | Cached instance-labeled training masks |
| `manifest.csv` | Image locations, observation information and train/test membership |
| `runs/.../best.pt` | Best trained model weights for each experiment and seed |
| `runs/.../summary.json` | Recorded training configuration, history, PQ and timing |
| `submission.csv` | Individual filament predictions encoded for Kaggle |
| `submission.metadata.json` | Inference configuration and runtime information |
| `compression_results.csv` | Compression and efficiency comparison |

The complete pipeline runs from **annotated solar observations**, through **segmentation learning and instance separation**, to **measured performance and encoded filament predictions**.
