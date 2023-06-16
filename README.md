# Vocos

## Training

Prepare a filelist of audio files for the training and validation set:

```bash
find $TRAIN_DATASET_DIR -name *.wav > filelist.train
find $VAL_DATASET_DIR -name *.wav > filelist.val
```

Fill a config file, e.g. [vocos.yaml](configs%2Fvocos.yaml), with your filelist paths and start training with:

```bash
conda create -n vocos python=3.10
conda activate vocos

pip install -r requirements-all.txt

# train from the ground
python train.py -c configs/vocos.yaml

# using a checkpoint
python train.py -c configs/vocos.yaml --trainer.resume_from_checkpoint /home/yehor/Work/github/vocos/logs/lightning_logs/version_0/checkpoints/last.ckpt

# run tensorboard to see the metrics
tensorboard --logdir logs/lightning_logs/version_1/

conda activate my
conda remove -n vocos --all
```
