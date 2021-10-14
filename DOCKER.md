# Build on Docker

We provide a docker container [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyuanaq_connect_ust_hk/EXEpJpl7VIhJmFbaXo3Th1YBjuCzO9ht2LEN0WKP3tQ_2A?e=4Fg3RB).

## Set Up

Import the downloaded container.

```bash
cat Manifold-SCA.tar | docker import - <image_name>
```

Run the container.

```bash
docker run -it <image_name> /bin/bash
```

Go to the working directory.

```bash
cd /home/Manifold-SCA
export MANIFOLD_SCA=$PWD
```

The working directory is organized as:
```python
📂Manifold-SCA
 ┣📂blinded_data # blinded data
 ┃ ┗📂{dataset_name}
 ┣📂code # implmentation of our framework
 ┣📂data # processed data and logged side channels
 ┃ ┗📂{dataset_name}
 ┃   ┣📂{media_data}
 ┃   ┣📂pin # side channel records collected using Pin
 ┃   ┗📂pp # side channel records collected using Prime+Probe
 ┣📂models # our trained models
 ┣📂output # experiment outputs will be save here
 ┣📂pin # our pintool and scripts for collecting side channel
 ┣📂prime_probe # scripts for launching Prime+Probe
 ┣📂target # compiled target software
 ┗📂tool # scripts for data processing
```

## Experiment

### Prepare Data

We already provide data and side channels in the `data` folder.

You can also collect data using our scripts.

```bash
cd target/pin-3.11/source/tools/ManualExamples
python3 prep_celeba.py
# or [prep_chest.py, prep_coco.py, prep_dialog.py, prep_sc09.py, prep_urmp.py] 
```

### Side Channel Attack

Run following scripts to approximate manifold.
```bash
python3 recons_image.py --exp_name="CelebA_cacheline" --dataset="CelebA" --side="cacheline"
# or
python3 recons_audio.py --exp_name="SC09_cacheline" --dataset="SC09" --side="cacheline"
# or
python3 recons_text.py --exp_name="DailyDialog_cacheline" --dataset="DailyDialog" --side="cacheline"
```

Reconstruct media from unknown (i.e., the test split) side channels using our trained models.

```bash
python3 output.py --dataset="CelebA"
```

You can choose `dataset` in `["CelebA", "ChestX-ray", "SC09", "Sub-URMP", "COCO", "DailyDialog"]`. Reconstructed media data will be saved in `output/recons_{dataset}`.

### Localize Vulnerabilities

Go to `target/pin-3.11/source/tools/ManualExamples/` and run `python3 prep_celeba.py` to prepare some instrumenting outputs of our pintool. The output will be saved in `data/CelebA_crop128/pin/raw/`.

Once the models have been trained, run this script.

```bash
python3 localize.py --exp_name="CelebA_cacheline" --dataset="CelebA" --side="cacheline"
```

Or set the model path to our trained models.

```python
model_path = (args.ckpt_root + 'final.pth')
# -->
ROOT = os.environ.get('MANIFOLD_SCA')
model_path = ROOT + '/models/pin/CelebA_cacheline/final.pth'
```

Uncomment the following lines to localize other software.

```python
media_dataset = CelebaDataset(
                    img_dir=args.data_path[args.dataset]['media'], 
                    npz_dir=args.data_path[args.dataset][args.side],
                    ID_path=args.data_path[args.dataset]['ID_path'],
                    split=args.data_path[args.dataset]['split'][1],
                    trace_c=args.trace_c,
                    trace_w=args.trace_w,
                    image_size=args.image_size,
                    side=args.side
                )

# media_dataset = SC09Dataset(
#                 lms_dir=args.data_path[args.dataset]['media'],
#                 npz_dir=args.data_path[args.dataset][args.side],
#                 split=args.data_path[args.dataset]['split'][1],
#                 trace_c=args.trace_c,
#                 trace_w=args.trace_w,
#                 max_db=args.max_db,
#                 min_db=args.min_db,
#                 side=args.side
#             )

engine.image_localize(media_loader)
# engine.audio_localize(media_loader)
```

### Perception Blinding

Run the following script to see reconstructed images from side channels of blinded images.

```bash
python3 output_blind.py --dataset="CelebA"
# or
python3 output_blind.py --dataset="ChestX-ray"
```

The reconstructed images will be saved in `output/blinded_{dataset}`.

### Noise Resilence

Run the following scripts to investigate how nosie of different types affect the quality of reconstructed media data.

```bash
python3 output_noise.py --dataset="CelebA" --noise_op="shift" --noise_k="100"
```

You can choose `noise_op` from `["shift", "delete", "noise", "zero"]`. The reconstructed media data will be saved in `output/noise_{dataset}`.

### Prime+Probe

Run the following scripts to approximate manifold from side channels logged by Prime+Probe.

```bash
python3 pp_image.py --exp_name="CelebA_intel_dcache" --dataset="CelebA" --cpu="intel" --cache="dcache"
# or
python3 pp_audio.py --exp_name="SC09_intel_dcache" --dataset="SC09" --cpu="intel" --cache="dcache"
# or
python3 pp_text.py --exp_name="DailyDialog_intel_dcache" --dataset="DailyDialog" --cpu="intel" --cache="dcache"
```

To reconstruct media data from unknown side channels, you can uncomment the following lines.

```python
# Part B: for reconstructing media data
# B1. use our trained model
ROOT = '..' if os.environ.get('MANIFOLD_SCA') is None else os.environ.get('MANIFOLD_SCA')
engine.load_model(ROOT + '/models/pp/SC09_intel_dcache/final.pth')

# B2. use your model
engine.load_model(args.ckpt_root + 'final.pth')

engine.inference(test_loader)
```
