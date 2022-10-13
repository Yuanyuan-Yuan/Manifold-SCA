![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg) ![Pytorch 1.4](https://img.shields.io/badge/pytorch-1.4-green.svg)
# Automated Side Channel Analysis of Media Software with Manifold Learning

Official implementation of **USENIX Security 2022** paper: *Automated Side Channel Analysis of Media Software with Manifold Learning*.

This artifact passed the artifact evaluation of USENIX Security 2022 and got the
**Available**, **Functional**, and **Reproduced** Badges.

Paper link: https://www.usenix.org/conference/usenixsecurity22/presentation/yuan 

Extended version: https://arxiv.org/pdf/2112.04947.pdf

Artifact appendix: [here](https://github.com/Yuanyuan-Yuan/Manifold-SCA/blob/main/files/artifact.pdf)

Slides: [here](https://github.com/Yuanyuan-Yuan/Manifold-SCA/blob/main/files/slides.pdf)

## Note

**Warning**: This repo is provided as-is and is only for research purposes. Please use it only on test systems with no sensitive data. You are responsible for protecting yourself, your data, and others from potential risks caused by this repo. 

## Updates

- 13 Oct. 2022. Update the artifact paper and citation bib.

- 9 Oct. 2021. Released data, code and trained models.

## Requirements

- To build from source code, install following requirements.

    ```setup
    pip install torch==1.4.0
    pip install torchvision==0.5.0
    pip install numpy==1.18.5
    pip install pillow==7.2.0
    pip install opencv-python
    pip install scipy==1.5.0
    pip install matplotlib==3.2.2
    pip install librosa==0.7.2
    pip install progressbar
    ```
  
  Then type
  
  ```bash
  git clone https://github.com/Yuanyuan-Yuan/Manifold-SCA
  cd Manifold-SCA
  export MANIFOLD_SCA=$PWD
  ```
  
- If you would like to build this repo from docker, see [DOCKER](https://github.com/Yuanyuan-Yuan/Manifold-SCA/blob/main/DOCKER.md) and skip the following steps except for [6.1](https://github.com/Yuanyuan-Yuan/Manifold-SCA#61-prepare-data).

## 0. Output

We provide data and our trained models [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/Eo-muH8ZPihOkXEXKF6TQo4BUzgpGHWdV3X2n9N9B9sExg?e=ekIFpx).

Run the following scripts to produce outputs from these data samples

```bash
python output.py --dataset="CelebA"
```

```bash
python output_blind.py --dataset="CelebA"
```

```bash
python output_noise.py --dataset="CelebA"
```

You can choose `dataset` from `["CelebA", "ChestX-ray", "SC09", "Sub-URMP", "COCO", "DailyDialog"]`. Results will be saved in `output`.

We also provide all our [outputs](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/Eo-muH8ZPihOkXEXKF6TQo4BUzgpGHWdV3X2n9N9B9sExg?e=ekIFpx).

## 1. Datasets

We provide sampels of our processed data [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/Eo-muH8ZPihOkXEXKF6TQo4BUzgpGHWdV3X2n9N9B9sExg?e=ekIFpx).

### CelebA

Download the CelebA dataset from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). We use the `Align&Cropped Images` version.

After downloading the dataset, go to `tool`. Then run

```bash
python crop_celeba.py --input_dir="/path/to/unzipped_images" --output_dir="/path/to/cropped_images"
```

to crop all images to size of `128*128`. We provide several examples in `data/CelebA_crop128/image`.

### ChestX-ray

Download the ChestX-ray dataset from [here](https://stanfordmlgroup.github.io/competitions/chexpert/).

After downloading the dataset, go to `tool`. Then run

```bash
python resize_chest.py --input_dir="/path/to/unzipped_images" --output_dir="/path/to/resized_images"
```

to convert all images to JPEG format and resize them to size of `128*128`. We provide several examples in `data/ChestX-ray_jpg128/image`.

### SC09 & Sub-URMP

Download the SC09 dataset from [here](https://www.kaggle.com/rahulbhalley/sc09-spoken-numbers) and Sub-URMP dataset [here](https://www.cs.rochester.edu/~cxu22/d/vagan/).

We process audios in the Log-amplitude of Mel Spectrum (LMS) form, which is a 2D representation. Once the dataset is downloaded, go to `tool` and run

```bash
python audio2lms.py --dataset="{SC09} or {Sub-URMP}" --input_dir="/path/to/audios" --output_dir="/path/to/lms"
```

to covert all audios to their LMS representations. Several examples are provided in `data/SC09/lms` and `data/Sub-URMP/lms` respectively.

### COCO-Caption & DailyDialog

Download COCO captions from [here](https://cocodataset.org/#download). We use the `2014 Train/Val annotations`. After downloading you need to extract captions from `captions_train2014.json` and `captions_val2014.json`. We provide several examples in `data/COCO/text/train.json` and `data/COCO/text/val.json`.

Download DailyDialog dataset from [here](http://yanran.li/dailydialog.html). After downloading you will have `dialogues_train.txt` and `dialogues_test.txt`. We suggest you store the sentences in `json` files. Several examples are given in `data/DailyDialog/text/train.json` and `data/DailyDialog/text/test.json`.

Once the sentences are prepared, you need to build the corresponding vocabulary. Go to `tool` and run

```bash
python build_vocab.py input_path="/path/to/sentences.json" --output_path="/path/to/vocabulary.json" --freq=minimal_word_frequency
```

to build the vocabulary. We provide our vocabularies in `data/COCO/text/word_dict_freq5.json` and `data/DailyDialog/text/word_dict_freq5.json`.

## 2. Target Software

Install [libjpeg](https://github.com/libjpeg-turbo/libjpeg-turbo), [hunspell](https://github.com/hunspell/hunspell) and [ffmpeg](https://github.com/FFmpeg/FFmpeg).

We already set up the three software in [DOCKER](https://github.com/Yuanyuan-Yuan/Manifold-SCA/blob/main/DOCKER.md).


## 3. Side Channel Attack

We analyze three common side channels, namely, cache bank, cache line and page tables.

### 3.1. Prepare Data

We use [Intel Pin](https://software.intel.com/content/www/us/en/develop/articles/pin-a-dynamic-binary-instrumentation-tool.html) (Ver. 3.11) to collect the accessed memory addresses of the target software when processing media data. We already set up the Pin in [DOCKER](https://github.com/Yuanyuan-Yuan/Manifold-SCA/blob/main/DOCKER.md).

We provide our pintool in `pin/pintool/mem_access.cpp`. Download Pin from [here](https://software.intel.com/content/www/us/en/develop/articles/pin-a-binary-instrumentation-tool-downloads.html) and unzip it to `PIN_ROOT` (specify this path by yourself).

To prepare accessed memory addresses of libjpeg when processing CelebA images, first put `pin/pintool/mem_access.cpp` into `/PIN_ROOT/source/tools/ManualExamples/` and run

```bash
make obj-intel64/mem_access.so TARGET=intel64
```

to compile the pintool. Before collect the memory address, remember to run

```bash
setarch $(uname -m) -R /bin/bash
```

in your bash to disable ASLR. In fact, the ASLR does not affect our approach, so you can also drop the randomized bits of collected memory address.

Then put `pin/prep_celeba.py` into `/PIN_ROOT/source/tools/ManualExamples/` and set the following variables:
- `input_dir` - Directory of media data.
- `npz_dir` - Directory where the accessed memory addresses will be saved. Addresses of each media data will be saved in `.npz` format.
- `raw_dir` - Directory where the raw output of our pintool will be saved. These files will be used for localize side channel vulnerabilities.
- `libjpeg_path` - Path to the executable file of libjpeg.

You can speed up the progress by running multiple processes. Go to `/PIN_ROOT/source/tools/ManualExamples/` and simply set variable `total_num` in `*.py` to the number of processes and run

```bash
python prep_celeba.py --ID=id_starting_from_1
```

to prepare data. Follow the same procedure for other datasets.

We provide our collected side channel records of all datasets [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/Eo-muH8ZPihOkXEXKF6TQo4BUzgpGHWdV3X2n9N9B9sExg?e=ekIFpx).

### 3.2. Map Memory Addresses to Side Channels

We map the collected memory addresses *addr* to side channels according to the following table.

| CPU Cache Bank Index | CPU Cache Line Index | OS Page Table Index |
|  :----:  | :----: | :----: |
| *addr >> 2* | *addr >> 6* | *addr >> 12* |

Set the following variables in `tool/addr2side.py`.
- `input_dir` - Directory of collected `.npz` files recording accessed memory addresses.
- `cachebank_dir` - Directory of converted cache bank indexes.
- `cacheline_dir` - Directory of converted cache line indexes.
- `pagetable_dir` - Directory of converted page table indexes.

Then run

```bash
python addr2side.py
```

to get the side channels records. You can also speed up the progress by running multiple processes.

### 3.3. Reconstruct Private Media Data

You need to first customize following data directories in `code/data_path.json`.

```json
{ 
    "dataset_name": {
        "media": "/path/to/media_data/",
        "cachebank": "/path/to/cache_bank/",
        "cacheline": "/path/to/cache_line/",
        "pagetable": "/path/to/page_table/",
        "split": ["train", "test"]
        },
}
```

To approximate the manifold of face photos from cache line indexes, go to `code` and run

```bash
python recons_image.py --exp_name="CelebA_cacheline" --dataset="CelebA" --side="cacheline" 
```

The `recons_image.py` script approximates manifold using the `train` split of CelebA dataset and ends within 24 hours on one Nvidia GeForce RTX 2080 GPU. Outputs (e.g., trained models, logs) will by default be saved in `output/CelebA_cacheline`. You can customize the output directory by setting `--output_root="/path/to/output/"`. The procedure is same for other media data (i.e., audio, text).

Once the desired manifold is constructed, run

```bash
python output.py --exp_name="CelebA_cacheline" --dataset="CelebA" --side="cacheline"
```

to reconstruct unknown face photos (i.e., the `test` split). The reconstructed face photos will by default be saved in `output/CelebA_cacheline/recons/`. This procedure is also same for audio and text data.

We use [Face++](https://www.faceplusplus.com/) to assess the similarity of IDs between reconstructed and reference face photos. The online service is free at the time of writing, so you can register your own account. Then set the `key` and `secret` variables in `code/face_similarity.py` and run

```bash
python face_similarity.py --recons_dir="../output/CelebA_cacheline/recons/" --target_dir="../output/CelebA_cacheline/target/" --output_path="../output/CelebA_cacheline/simillarity.txt"
```

The results will by default be saved in `output/CelebA_cacheline/simillarity.txt`.

For ChestX-ray images, we use this [tool](https://github.com/jfhealthcare/Chexpert) to check the consistency between disease information of reconstructed reference images.

You can also evaluate the similarity between reconstructed and reference images by running `tool/SSIM.py`.

```bash
python SSIM.py --K=1 --N=100 --recons_dir="../output/CelebA_cacheline/recons/" --target_dir="../output/CelebA_cacheline/target/" --output_path="../output/CelebA_cacheline/SSIM.txt"
```

The evaluation methods of audio data and text data are implemented in `code/recons_audio.py` and `code/recons_text.py` respectively. Note that the reconstructed audios are in the LMS representation, to get the raw audio (i.e., `.wav` format), run

```bash
python lms2audio.py --input_dir="/path/to/lms" --output_dir="/path/to/wav"
```

If you want to use your customrized dataset, write your dataset class in `code/data_loader.py`.

We also provide our trained [models](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/Eo-muH8ZPihOkXEXKF6TQo4BUzgpGHWdV3X2n9N9B9sExg?e=ekIFpx).

## 4. Program Point Localization

Once you successfully perform side channel attacks on the target softwares, you can localize the side channel vulnerabilities.

First customize the following variables in `code/data_path.json`.

```json
{ 
    "dataset_name": {
        "pin": "/path/to/pintool_output/",
        },
}
```

Then go to `code` and run.

```bash
python localize.py --exp_name="CelebA_cacheline" --dataset="CelebA" --side="cacheline"
```

The output `.json` file will be saved in `output/CelebA_cacheline/localize`. The results are organized as

```json
{
    "function_name; assmbly instruction; instruction address": "count",
}
```

The results of media software (e.g., libjpeg) processing different data (e.g., CelebA and ChestX-ray) are mostly consistent. We provide our localized [vulnerabilities](https://github.com/Yuanyuan-Yuan/Manifold-SCA/tree/main/appendix/vulnerability).

## 5. Perception Blinding

We provid the blinded media data [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/Eo-muH8ZPihOkXEXKF6TQo4BUzgpGHWdV3X2n9N9B9sExg?e=ekIFpx).

To blind media data, go to `code` and run

```bash
python blind_add.py --meida="{image} or {audio} or {text}" --mask_weight=0.9 --mask="{mask_word} or {/path/to/mask_image_or_audio}" --input_dir="{/path/to/text.json} or {/foler/of/image_or_audio/}" --output_dir="{/path/to/text.json} or {/foler/of/image_or_audio/}"
```

To unblind media data, run

```bash
python blind_subtract.py --meida="{image} or {audio} or {text}" --mask_weight=0.9 --mask="{mask_word} or {/path/to/mask_image_or_audio}" --input_dir="{/path/to/text.json} or {/foler/of/image_or_audio/}" --output_dir="{/path/to/text.json} or {/foler/of/image_or_audio/}"
```

Run

```bash
python output_blind.py --dataset="CelebA"
# or
python output_blind.py --dataset="ChestX-ray"
```

to see reconstructed media data from side channels corresponding to blinded data.

## 6. Attack with Prime+Probe

We use [Mastik](https://cs.adelaide.edu.au/~yval/Mastik/) (Ver. 0.02) to launch Prime+Probe on L1 cache of Intel Xeon CPU and AMD Ryzen CPU. We provide our scripts in `prime_probe/Mastik/demo`. After downloading Mastik, you can put our scripts into the `demo` folder and run `make` in the root folder to compile our scripts. We highly recommend you to *set the cache miss threshold* in these scripts according to your machines. 

The *Prime+Probe* is launched in Linux OS. You need first to install **taskset** and **cpuset**.

### 6.1. Prepare Data

We assume *victim* and *spy* are on the same CPU core and no other process is runing on this CPU core. To attack libjpeg, you need to first customize the following variables in `code/prime_probe/coord_image.py`

- `pp_exe` - Path to the executable file of our prime+probe script.
- `input_dir` - Directory of media data.
- `side_dir` - Directory where the collected cache set accesses will be saved.
- `libjpeg_path` - Path to the executable file of libjpeg.
- `TRY_NUM` - Repeating times of processing one media using the target software.
- `PAD_LEN` - The length that the collected trace will be padded to.

The script `coord_image.py` is the coordinator which runs spy and victim on the same CPU core simultaneously and saves the collected cache set access.

Then run

```bash
sudo cset shield --cpu {cpu_id}
```

to isolate one CPU core. Once the CPU core is isolated, you can run

```bash
sudo cset shield --exec python run_image.py -- {cpu_id} {segment_id}
```

The script `run_image.py` will run `coord_image.py` using **taskset**. Note that we seperate the media data into several segments to speed up the side channel collection. The `segment_id` starts from 0. The procedure is same for other media data.

We provide our logged [side channels](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/Eo-muH8ZPihOkXEXKF6TQo4BUzgpGHWdV3X2n9N9B9sExg?e=ekIFpx). 

### 6.2. Reconstruct Private Media Data

First customize the following variables in `code/data_path.json`.

```json
{ 
    "dataset_name": {
        "pp-intel-dcache": "/path/to/intel_l1_dcache",
        "pp-intel-icache": "/path/to/intel_l1_icache",
        "pp-amd-dcache": "/path/to/amd_l1_dcache",
        "pp-amd-icache": "/path/to/amd_l1_icache",
        },
}
```

Then run

```bash
python pp_image.py --exp_name="CelebA_pp" --dataset="CelebA" --cpu="intel" --cache="dcache"
```

to approximate the manifold. To reconstruct unknonw images from the collected cache set accesses, uncomment

```python
engine.load_model("/path/to/model.pth")
engine.inference(test_loader, "test")
```

in `pp_image.py`. The reconstructed images will be saved in `output/CelebA_pp/recons`. Follow the same procedure for other media data.

We release our [trained models](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/Eo-muH8ZPihOkXEXKF6TQo4BUzgpGHWdV3X2n9N9B9sExg?e=ekIFpx) and all SC09 [audios](https://github.com/Yuanyuan-Yuan/Manifold-SCA/tree/main/appendix/audio) reconstructed from side channels collected by *Prime+Probe* on Intel L1 D cache.

## 7. Noise Resilience

We have the following noise insertion schemes (see more details in our paper).

| Pin logged trace | Prime+Probe logged trace |
|  :----:  | :----: |
| Gaussian | Leave out |
| Shifting | False hit/miss |
| Removal | Wrong order |

To insert the "shifting" noise into pin logged trace, go to `code` and run

```bash
python output_noise.py --exp_name="CelebA_cacheline" --dataset="CelebA" --side="cacheline" --noise_op="shift" --noise_k=100
```

Images reconstructed from noisy cache line records will be saved in `output/CelebA_cacheline/recons_noise` by default.

To insert the "wrong order" noise into prime+probe logged trace, you need to modify `code/pp_image.py` as

```python
# test_dataset = RealSideDataset(args, split=args.data_path[args.dataset]["split"][1])
test_dataset = NoisyRealSideDataset(args, split=args.data_path[args.dataset]["split"][1])
```

and uncomment

```python
engine.load_model("/path/to/model.pth")
engine.inference(test_loader, "test")
```

to reconstruct unknown images from noisy side channel records.

The procedure is same for other media data. Note that in order to assess the noise resilience, you should **NOT** approximate manifold (i.e., train the model) using the noisy side channel.

## 8. Customization

All parameters are set in `code/params.py`. You can customize the hyper-parameters for approximating manifold.

All datasets are implemented with OOP manner in `code/data_loader.py`. You can modify the dataset class to support your own data.

All models are also implemented with OOP manner in `code/model.py`. You can build a new framework from new models.

## Copyrights

The Prime+Probe scripts in `prime_probe/Mastik/demo` are implemented based on [Mastik](https://cs.adelaide.edu.au/~yval/Mastik/), so you should follow the license of Mastik.

For all other original parts, we allow free distribution of the code under the MIT license.


## Citation

```bibtex
@inproceedings{yuan2022automated,
  title={Automated side channel analysis of media software with manifold learning},
  author={Yuan, Yuanyuan and Pang, Qi and Wang, Shuai},
  booktitle={31st USENIX Security Symposium (USENIX Security 22)},
  pages={4419--4436},
  year={2022}
}
```

If you have any questions, feel free to contact with me (<yyuanaq@cse.ust.hk>).

