# CNN vs Vision Transformer in Automatic License Plate Recognition

This project compares two types of deep learning models — a classic convolutional neural network (ResNet-50) and a newer Vision Transformer (ViT-Small) to check which one works better for reading license plates and identifying vehicle characteristics from camera images.

## What does this project include?

Imagine a parking lot camera that needs to automatically identify every car that drives in. This system does exactly that, it looks at a photo of a vehicle and tries to:

1. **Find the license plate** in the image
2. **Read the characters** on the plate (e.g. "WR 12345")
3. **Identify the vehicle** — what color it is, what type of body it has (sedan, SUV, hatchback...), and what brand it is

The key question this project answers is: **which type of neural network does this job better — the traditional CNN or the newer Transformer-based approach?**

## Why does this comparison matter?

For years, convolutional neural networks (CNNs) have been the standard tool for image recognition tasks. In 2021, a new approach called Vision Transformer (ViT) was introduced, borrowed from natural language processing. ViT works very differently, instead of scanning an image with small filters, it splits the image into patches and looks at relationships between all of them at once.

Both approaches have their advantages, and it is not obvious which one is better for real-world applications like license plate recognition. This project runs a series of experiments to find out.

## What was tested?

Five experiments were run, each answering a different question:

- **E1 — How accurate are the models in ideal conditions?** Both models were trained on the full dataset and tested on clean, high-quality images.
- **E2 — How do they hold up when images are degraded?** The models were tested on blurry, dark, rainy, and partially obscured images — without any retraining.
- **E3 — What if there is not much training data?** The models were trained on only 10%, 20%, 30%, 50%, and 80% of the available data to see how quickly they learn.
- **E4 — Do the models overfit?** Training and validation accuracy curves were tracked across epochs to check whether the models memorize the training data or actually learn to generalize.
- **E5 — How fast are they?** Inference speed was measured in frames per second (FPS), both for the model alone and for the full processing pipeline including detection.

A statistical analysis using the Wilcoxon signed-rank test was also performed to check whether any observed differences between the two models are statistically significant or just random variation.

## What were the results?

In short — both models are very good, but they have different strengths:

- **Accuracy**: both models read license plates correctly about 99% of the time under ideal conditions. For vehicle attribute classification, both achieved around 91.6% mean accuracy.
- **Robustness**: when images are blurry, ViT-Small is much more resilient — it loses only 22 percentage points of accuracy, while ResNet-50 drops by 51 percentage points.
- **Learning efficiency**: with only 10% of training data, ViT-Small already achieves 87% plate accuracy, while ResNet-50 manages only 52%. ViT learns more from less data.
- **Speed**: both models are fast enough for real-time use. ResNet-50 is slightly faster when measured in isolation, but in the full pipeline ViT-Small is actually faster.
- **Statistical significance**: the Wilcoxon test found no statistically significant difference between the two models at α=0.05, meaning the differences — while consistent — are small enough to be within the range of random variation.

The bottom line is that ViT-Small is the better choice when conditions are difficult or training data is limited, while ResNet-50 is a solid option when computational resources are tight.

## Datasets used

| Dataset | What it contains | Used for |
|---------|-----------------|---------|
| CCPD 2019 | 200,000 photos of Chinese license plates with annotations encoded in filenames | License plate OCR |
| CompCars | ~31,000 car photos annotated with make and body type | Vehicle type & make classification |
| VehicleColor (VCoR) | ~9,000 car photos in 15 color categories | Vehicle color classification |

## Datasets

| Dataset | Task | Link |
|---------|------|------|
| CCPD 2019 | License plate OCR | [GitHub](https://github.com/detectRecog/CCPD) |
| CompCars | Vehicle type & make | [CUHK](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) |
| VehicleColor (VCoR) | Vehicle color | [Kaggle](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset) |

If you use these datasets in your work, please cite the original papers:

- **CCPD**: Xu et al., *Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline*, ECCV 2018
- **CompCars**: Yang et al., *A Large-Scale Car Dataset for Fine-Grained Categorization and Verification*, CVPR 2015  
- **VehicleColor**: Kezebou et al., *Artificial Intelligence for Text-Based Vehicle Search, Recognition, and Continuous Localization in Traffic Videos*, AI 2021

## Project structure

```
alpr_thesis/
├── models/              # Neural network architectures (ResNet, ViT, heads)
├── training/scripts/    # Scripts to train the models
├── evaluation/          # Scripts to run experiments and generate plots
├── utils/               # Dataset loaders
└── scripts/             # Detection, visualization, statistical analysis
```

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.12 and an NVIDIA GeForce RTX 3070 Ti GPU.

## Quick start

```bash
> **Note:** Trained model checkpoints are not included in this repository.
> You need to train the models yourself following the steps below.

# 1. Download datasets and place them in data/raw/
# 2. Preprocess CCPD images
python scripts/preprocess_ccpd.py

# 3. Train the OCR model
python training/scripts/train_ocr_cnn.py --arch resnet50 --workers 4 --output-dir outputs/ocr_cnn/E1/resnet50

# 4. Run all experiments
python evaluation/evaluate.py --task all --experiment all

# 5. Generate plots
python evaluation/plot_results.py
```

## License

This project is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to share and adapt this work for non-commercial purposes,
as long as you give appropriate credit to the original author.

© 2025 Dawid Jordan
