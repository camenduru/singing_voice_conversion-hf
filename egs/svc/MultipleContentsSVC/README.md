# Leveraging Content-based Features from Multiple Acoustic Models for Singing Voice Conversion

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2310.11160)
[![demo](https://img.shields.io/badge/SVC-Demo-red)](https://www.zhangxueyao.com/data/MultipleContentsSVC/index.html)

<br>
<div align="center">
<img src="../../../imgs/svc/MultipleContentsSVC.png" width="85%">
</div>
<br>

This is the official implementation of the paper "[Leveraging Content-based Features from Multiple Acoustic Models for Singing Voice Conversion](https://arxiv.org/abs/2310.11160)" (NeurIPS 2023 Workshop on Machine Learning for Audio). Specially,

- The muptile content features are from [Whipser](https://github.com/wenet-e2e/wenet) and [ContentVec](https://github.com/auspicious3000/contentvec).
- The acoustic model is based on Bidirectional Non-Causal Dilated CNN (called `DiffWaveNetSVC` in Amphion), which is similar to [WaveNet](https://arxiv.org/pdf/1609.03499.pdf), [DiffWave](https://openreview.net/forum?id=a-xFK8Ymz5J), and [DiffSVC](https://ieeexplore.ieee.org/document/9688219).
- The vocoder is [BigVGAN](https://github.com/NVIDIA/BigVGAN) architecture and we fine-tuned it in over 120 hours singing voice data.

There are four stages in total:

1. Data preparation
2. Features extraction
3. Training
4. Inference/conversion

> **NOTE:** You need to run every command of this recipe in the `Amphion` root path:
> ```bash
> cd Amphion
> ```

## 1. Data Preparation

### Dataset Download

By default, we utilize the five datasets for training: M4Singer, Opencpop, OpenSinger, SVCC, and VCTK. How to download them is detailed [here](../../datasets/README.md).

### Configuration

Specify the dataset paths in  `exp_config.json`. Note that you can change the `dataset` list to use your preferred datasets.

```json
    "dataset": [
        "m4singer",
        "opencpop",
        "opensinger",
        "svcc",
        "vctk"
    ],
    "dataset_path": {
        // TODO: Fill in your dataset path
        "m4singer": "[M4Singer dataset path]",
        "opencpop": "[Opencpop dataset path]",
        "opensinger": "[OpenSinger dataset path]",
        "svcc": "[SVCC dataset path]",
        "vctk": "[VCTK dataset path]"
    },
```

## 2. Features Extraction

### Content-based Pretrained Models Download

By default, we utilize the Whisper and ContentVec to extract content features. How to download them is detailed [here](../../../pretrained/README.md).

### Configuration

Specify the dataset path and the output path for saving the processed data and the training model in `exp_config.json`:

```json
    // TODO: Fill in the output log path. The default value is "Amphion/ckpts/svc"
    "log_dir": "ckpts/svc",
    "preprocess": {
        // TODO: Fill in the output data path. The default value is "Amphion/data"
        "processed_dir": "data",
        ...
    },
```

### Run

Run the `run.sh` as the preproces stage (set  `--stage 1`).

```bash
sh egs/svc/MultipleContentsSVC/run.sh --stage 1
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "1"`.

## 3. Training

### Configuration

We provide the default hyparameters in the `exp_config.json`. They can work on single NVIDIA-24g GPU. You can adjust them based on you GPU machines.

```json
"train": {
        "batch_size": 32,
        ...
        "adamw": {
            "lr": 2.0e-4
        },
        ...
    }
```

### Run

Run the `run.sh` as the training stage (set  `--stage 2`). Specify a experimental name to run the following command. The tensorboard logs and checkpoints will be saved in `Amphion/ckpts/svc/[YourExptName]`.

```bash
sh egs/svc/MultipleContentsSVC/run.sh --stage 2 --name [YourExptName]
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "0,1,2,3"`.

## 4. Inference/Conversion

### Pretrained Vocoder Download

We fine-tune the official BigVGAN pretrained model with over 120 hours singing voice data. The benifits of fine-tuning has been investigated in our paper (see this [demo page](https://www.zhangxueyao.com/data/MultipleContentsSVC/vocoder.html)). The final pretrained singing voice vocoder is released [here](../../../pretrained/README.md#amphion-singing-bigvgan) (called `Amphion Singing BigVGAN`).

### Run

For inference/conversion, you need to specify the following configurations when running `run.sh`:

| Parameters                                          | Description                                                                                                                                | Example                                                                                                                                                                            |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--infer_expt_dir`                                  | The experimental directory which contains `checkpoint`                                                                                     | `Amphion/ckpts/svc/[YourExptName]`                                                                                                                                                 |
| `--infer_output_dir`                                | The output directory to save inferred audios.                                                                                              | `Amphion/ckpts/svc/[YourExptName]/result`                                                                                                                                          |
| `--infer_source_file` or `--infer_source_audio_dir` | The inference source (can be a json file or a dir).                                                                                        | The `infer_source_file` could be `Amphion/data/[YourDataset]/test.json`, and the `infer_source_audio_dir` is a folder which includes several audio files (*.wav, *.mp3 or *.flac). |
| `--infer_target_speaker`                            | The target speaker you want to convert into. You can refer to `Amphion/ckpts/svc/[YourExptName]/singers.json` to choose a trained speaker. | For opencpop dataset, the speaker name would be `opencpop_female1`.                                                                                                                |
| `--infer_key_shift`                                 | How many semitones you want to transpose.                                                                                                  | `"autoshfit"` (by default), `3`, `-3`, etc.                                                                                                                                        |

For example, if you want to make `opencpop_female1` sing the songs in the `[Your Audios Folder]`, just run:

```bash
sh egs/svc/MultipleContentsSVC/run.sh --stage 3 --gpu "0" \
	--infer_expt_dir Amphion/ckpts/svc/[YourExptName] \
	--infer_output_dir Amphion/ckpts/svc/[YourExptName]/result \
	--infer_source_audio_dir [Your Audios Folder] \
	--infer_target_speaker "opencpop_female1" \
	--infer_key_shift "autoshift"
```

## Citations

```bibtex
@article{zhang2023leveraging,
  title={Leveraging Content-based Features from Multiple Acoustic Models for Singing Voice Conversion},
  author={Zhang, Xueyao and Gu, Yicheng and Chen, Haopeng and Fang, Zihao and Zou, Lexiao and Xue, Liumeng and Wu, Zhizheng},
  journal={Machine Learning for Audio Worshop, NeurIPS 2023},
  year={2023}
}
```
