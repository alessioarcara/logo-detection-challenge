# The Logo Detector Challenge
 
> This repository is my solution to a computer vision challenge organized by **[eyecan.ai](https://www.eyecan.ai)**. The goal: detect and localize a specific logo in real-world images, with no training data provided — just the logo itself.
 
## Approach
 
The challenge was tackled along two complementary tracks:
 
### Generator
 
- Cut-and-paste onto backgrounds with affine/perspective geometric transformations to force invariance to the logo's position, scale, and viewpoint.
- Color and brightness transformations to make the model invariant to color shifts and lighting differences.
- Additionally, to push the model to be invariant to cut-and-paste artifacts, multiple versions of each background/transformation seed were generated with different blending types, forcing invariance to possible artifacts.
- Use of COCO 2017 val and DTD (Describable Textures Dataset) to make the model invariant to the background: the former for more realistic images, the latter specifically for the possible textures behind the logo.
- To avoid false positives, a configurable number of examples is generated without the logo, making the model less prone to false detections.
- Added transformations to compensate for resolution drops, sensor noise, and motion blur, together with Dropout to encourage a more distributed use of the logo's features rather than reliance on a single part.
### Detector
 
- Loading DINOv2 patch14 (small/base) weights into a ViT: learning a feature extractor from scratch on a synthetic dataset is hard, while pre-trained weights allow faster convergence and better performance. DINO was chosen specifically because features learned through self-supervised training are more robust and generalize better than a zero-shot model, in addition to outperforming ImageNet features. Moreover, DINOv2 uses a patch embedding of 14 instead of 16 (as in DINOv1), yielding higher resolution.
- The feature extractor is kept frozen.
- To guarantee sufficient spatial context, an FPN is added — specifically a simplified version of DPT.
- To get more supervision signal than regressing just two values (x, y), a heatmap head with a CenterNet-based loss was chosen: the model predicts a heatmap and the location with the highest logit/probability is taken. This provides denser supervision and thus faster convergence.
### Validation
 
- For the validation set, the logo was printed and photographed multiple times in different indoor/outdoor scenarios, varying perspective and scale to measure the model's real-world performance.
- This small dataset was labeled using LabelStudio, with a custom script to convert the labels into the Underfolder format. The dataset is available in the `data` folder.
### Metrics
 
Two dedicated metrics were implemented:
 
- **LocalizationAccuracy**: counts how many logos have a localization error < 10%.
- **LogoPresenceAccuracy**: measures accuracy both when the logo is present and when it is absent, in order to estimate false positives and false negatives.
---
 
## References (papers read during the challenge)
 
- *Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection*
- *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*
- *Deep Learning Logo Detection with Data Expansion by Synthesising Context*
- *On Pre-Trained Image Features and Synthetic Images for Deep Learning*
## Usage
 
Once the dependencies are installed:
 
```
uv sync  # or: pip install .
```
 
you can use the `scripts/eval.py` script to:
 
- evaluate the model on a specific folder
- or directly compute the metrics on a test dataset defined in the config
## Note
 
> This project uses [`ezconfy`](https://github.com/alessioarcara/EzConfy), a library of **mine** recently created to manage YAML configuration files. Here it is used to validate the config against `configs/schema.yaml` and to perform object instantiation directly from YAML via `_target_type_` and `_init_args_`.
 
## Repository structure
 
```
├── checkpoints/        # Saved model checkpoints
├── configs/            # YAML experiment configurations
│   ├── base.yaml       # Main configuration
│   └── schema.yaml     # Configuration schema
├── data/               # Datasets and assets used by the project
│   ├── raw/            # Data exported from LabelStudio
│   ├── training/       # Assets for synthetic generation
│   │   ├── backgrounds/ # Backgrounds used by the generator
│   │   └── logos/      # Source logos
│   └── validation/     # Real validation set in Underfolder format
├── notebooks/
│   └── visualize_generator.ipynb  # Notebook to visualize the generator
├── scripts/            # Entry points and utilities
│   ├── convert_labelstudio_data_to_underfolder.py  # LabelStudio conversion
│   ├── download_backgrounds.sh  # Background download
│   ├── eval.py         # Evaluation script
│   └── train.py        # Training script
├── src/                # Main project code
│   ├── data.py         # Dataset, dataloaders
│   ├── generator/      # Cut-and-paste generation pipeline
│   ├── models/         # Detector architectures
│   ├── training/       # Training loop, losses, metrics and callbacks
│   └── utils/          # Shared utilities
```
