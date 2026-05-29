# DU-AP_Purifier

**DU-AP_Purifier** is a defence tool developed within the **CoEvolution Hub** for improving the robustness of LiDAR-based semantic segmentation systems against adversarial attacks.

The tool implements **DU-AP — Deep Unrolled Adversarial Purification**, a lightweight model-based purification module designed for LiDAR perception pipelines that operate on **2D range-view representations**. DU-AP is placed before the segmentation model and removes adversarial perturbations from the input range image before inference.

Unlike heavy generative defences, DU-AP follows an optimisation-inspired deep unrolling strategy. Each layer of the network corresponds to a purification step composed of a **data consistency update** and a **learnable denoising module**. This makes the defence compact, interpretable, and suitable for deployment on embedded automotive platforms.

## Defence Objective

The objective of DU-AP_Purifier is to restore reliable semantic segmentation under adversarial conditions.

```text
Adversarial LiDAR Range Image
        │
        ▼
DU-AP Purification Module
        │
        ▼
Purified LiDAR Range Image
        │
        ▼
Semantic Segmentation Network
        │
        ▼
Robust Segmentation Output
```

## Design → Train → Deploy Flow

```text
┌──────────────┐
│    DESIGN    │
└──────┬───────┘
       │
       ▼
Formulate adversarial purification as a structured denoising problem
for 2D LiDAR range-view data.

       │
       ▼
┌──────────────┐
│    TRAIN     │
└──────┬───────┘
       │
       ▼
Train the deep-unrolled purification module using adversarially
perturbed range images and segmentation-based recovery objectives.

       │
       ▼
┌──────────────┐
│    DEPLOY    │
└──────┬───────┘
       │
       ▼
Insert DU-AP before the segmentation model as a lightweight online
defence layer for robust LiDAR perception.
```

## Main Capabilities

* Purifies adversarially perturbed LiDAR range images.
* Supports 2D range-view LiDAR segmentation pipelines.
* Operates as a plug-in pre-processing defence module.
* Uses a deep-unrolled, optimisation-inspired architecture.
* Preserves geometric consistency in the LiDAR range image.
* Improves segmentation robustness under adversarial attacks.
* Adds minimal computational overhead compared with large generative defences.
* Supports CoEvolution Hub workflows for attack assessment, defence validation, and trustworthy AI deployment.

## CoEvolution Hub Role

Within the **CoEvolution Hub**, DU-AP_Purifier supports the validation of secure and robust AI perception systems. It can be used to evaluate how LiDAR segmentation models behave under adversarial perturbations and how effectively a lightweight defence module can restore reliable predictions.

The tool is especially relevant for autonomous vehicle and robotic perception scenarios where LiDAR segmentation is safety-critical and must remain robust under manipulated or degraded sensor inputs.
