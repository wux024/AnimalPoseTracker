# Animal Pose Estimation with AnimalVitPose

Aanimal pose estimation aims to detect the keypoints of different species. It provides detailed behavioral analysis for neuroscience, medical and ecology applications. Some results are shown below.![](https://s3.bmp.ovh/imgs/2024/07/27/e1b49c32bd1cccbf.jpg)
## Installation

1. Create a conda virtual environment and activate it.

```
conda create -n animalvitpose python=3.8
conda activate animalvitpose
```

2. Install PyTorch (rqeuired version: 2.1.0) and torchvision following the [official instructions](https://pytorch.org/).

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

3. Install the base dependencies.

```
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
mim install "mmdet>=3.1.0"
mim install 'mmpretrain>=1.0.0'
```

4. Clone the mmpose repository and install the required dependencies.

```
git clone https://github.com/wux024/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -r requirements/albu.txt
pip install -v -e .
```
## Testing
You could download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1S1l5ohbf0x7Nm60k86K6OUqZFsCFg-sQ?usp=sharing).

```
CONFIG_FILE=configs\animal_2d_keypoint\animalvitpose\ap10k\animalvitpose-small_8xb64-210e_ap10k-256x256.py
CHECKPOINT_FILE=path/to/AnimalViTPose-S.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```
| ARGS                                  | Description                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | The path to the config file.                                                                                                                                        |
| `CHECKPOINT_FILE`                     | The path to the checkpoint file (It can be a http link, and you can find checkpoints [here](https://MMPose.readthedocs.io/en/latest/model_zoo.html)).               |
| `--work-dir WORK_DIR`                 | The directory to save the file containing evaluation metrics.                                                                                                       |
| `--out OUT`                           | The path to save the file containing evaluation metrics.                                                                                                            |
| `--dump DUMP`                         | The path to dump all outputs of the model for offline evaluation.                                                                                                   |
| `--cfg-options CFG_OPTIONS`           | Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file. If the value to be overwritten is a list, it should be of the form of either `key="[a,b]"` or `key=a,b`. The argument also allows nested list/tuple values, e.g. `key="[(a,b),(c,d)]"`. Note that quotation marks are necessary and that no white space is allowed. |
| `--show-dir SHOW_DIR`                 | The directory to save the result visualization images.                                                                                                              |
| `--show`                              | Visualize the prediction result in a window.                                                                                                                        |
| `--interval INTERVAL`                 | The interval of samples to visualize.                                                                                                                               |
| `--wait-time WAIT_TIME`               | The display time of every window (in seconds). Defaults to 1.                                                                                                       |
| `--launcher {none,pytorch,slurm,mpi}` | Options for job launcher.     

More details refer to [Testing](https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html#test-your-model).
## Training

1. The datasets (e.g. AP10K) used AnimalVitPose could be downloaded by contacting the corresponding author and me (<EMAIL>wux024@nenu.edu.cn). Extract the dataset to the `data` folder.
```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── ap10k
    |—— other datasets
```

2. Run the following command to train the model:
```
bash tools/train.sh --dataset ap10k --mode animalvitpose
```

3. The training log and checkpoints will be saved in the `work_dirs` folder.

4. Test the trained model:
``` 
bash tools/test.sh --dataset ap10k --mode animalvitpose
```

More details refer to [Training](https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html#launch-training).


