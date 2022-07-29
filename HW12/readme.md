# HW12 Da-Qi Ren


In this work, I used colab template for training Conformer-Transducer BPE Small on LibriSpeech from scratch for 10-20 epochs. 

### Files and Folders

(1) Notebook 1 : Preparation and system setup The config file for all Conformer, set it to small.
(2) Notebook 2 : Traning ASR with Tansducers\

-ipynb notebooks\
-Folder scripts\
-Folder datasets\
-Experiments/Transducer-Model\
-tokenizers




### Training Process:

  | Name              | Type                              | Params
------------------------------------------------------------------------
0 | preprocessor      | AudioToMelSpectrogramPreprocessor | 0     
1 | encoder           | ConvASREncoder                    | 2.3 M 
2 | decoder           | RNNTDecoder                       | 35.4 K
3 | joint             | RNNTJoint                         | 47.3 K
4 | loss              | RNNTLoss                          | 0     
5 | spec_augmentation | SpectrogramAugmentation           | 0     
6 | wer               | RNNTBPEWER                        | 0     
------------------------------------------------------------------------
2.3 M     Trainable params
0         Non-trainable params
2.3 M     Total params
9.380     Total estimated model params size (MB)
Sanity Checking: 0it [00:00, ?it/s]
[NeMo W 2022-07-29 06:57:05 nemo_logging:349] /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 16 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))
    
Training: 0it [00:00, ?it/s]
[NeMo W 2022-07-29 06:57:19 nemo_logging:349] /usr/local/lib/python3.7/dist-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
      warn(NumbaPerformanceWarning(msg))
    
[NeMo W 2022-07-29 06:57:20 nemo_logging:349] /usr/local/lib/python3.7/dist-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
      warn(NumbaPerformanceWarning(msg))
    
[NeMo W 2022-07-29 06:57:21 nemo_logging:349] /usr/local/lib/python3.7/dist-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
      warn(NumbaPerformanceWarning(msg))
    
[NeMo W 2022-07-29 06:57:22 nemo_logging:349] /usr/local/lib/python3.7/dist-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
      warn(NumbaPerformanceWarning(msg))
    
[NeMo W 2022-07-29 06:57:22 nemo_logging:349] /usr/local/lib/python3.7/dist-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
      warn(NumbaPerformanceWarning(msg))
    
[NeMo W 2022-07-29 06:57:29 nemo_logging:349] /usr/local/lib/python3.7/dist-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
      warn(NumbaPerformanceWarning(msg))
    
[NeMo W 2022-07-29 06:57:30 nemo_logging:349] /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 16 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))
    
Validation: 0it [00:00, ?it/s]
Epoch 9, global step 600: 'val_wer' reached 0.43855 (best 0.43855), saving model to '/content/gdrive/MyDrive/HW12/experiments/Transducer-Model/2022-07-29_06-56-24/checkpoints/Transducer-Model--val_wer=0.4386-epoch=9.ckpt' as top 3
Validation: 0it [00:00, ?it/s]
Epoch 19, global step 1200: 'val_wer' reached 0.22898 (best 0.22898), saving model to '/content/gdrive/MyDrive/HW12/experiments/Transducer-Model/2022-07-29_06-56-24/checkpoints/Transducer-Model--val_wer=0.2290-epoch=19.ckpt' as top 3
Validation: 0it [00:00, ?it/s]
Epoch 29, global step 1800: 'val_wer' reached 0.14618 (best 0.14618), saving model to '/content/gdrive/MyDrive/HW12/experiments/Transducer-Model/2022-07-29_06-56-24/checkpoints/Transducer-Model--val_wer=0.1462-epoch=29.ckpt' as top 3
Validation: 0it [00:00, ?it/s]
Epoch 39, global step 2400: 'val_wer' reached 0.09573 (best 0.09573), saving model to '/content/gdrive/MyDrive/HW12/experiments/Transducer-Model/2022-07-29_06-56-24/checkpoints/Transducer-Model--val_wer=0.0957-epoch=39.ckpt' as top 3
Validation: 0it [00:00, ?it/s]
Epoch 49, global step 3000: 'val_wer' reached 0.08668 (best 0.08668), saving model to '/content/gdrive/MyDrive/HW12/experiments/Transducer-Model/2022-07-29_06-56-24/checkpoints/Transducer-Model--val_wer=0.0867-epoch=49.ckpt' as top 3
[NeMo W 2022-07-29 07:07:52 nemo_logging:349] /usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py:2029: LightningDeprecationWarning: `Trainer.training_type_plugin` is deprecated in v1.6 and will be removed in v1.8. Use `Trainer.strategy` instead.
      "`Trainer.training_type_plugin` is deprecated in v1.6 and will be removed in v1.8. Use"
    
Restoring states from the checkpoint path at /content/gdrive/MyDrive/HW12/experiments/Transducer-Model/2022-07-29_06-56-24/checkpoints/Transducer-Model--val_wer=0.0867-epoch=49.ckpt
Restored all states from the checkpoint file at /content/gdrive/MyDrive/HW12/experiments/Transducer-Model/2022-07-29_06-56-24/checkpoints/Transducer-Model--val_wer=0.0867-epoch=49.ckpt



### Results: 


Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:02<00:17,  2.54s/sample]
Beam search progress::  25%|██▌       | 2/8 [00:02<00:07,  1.26s/sample]
Beam search progress::  38%|███▊      | 3/8 [00:05<00:08,  1.80s/sample]
Beam search progress::  62%|██████▎   | 5/8 [00:07<00:04,  1.41s/sample]
Beam search progress::  75%|███████▌  | 6/8 [00:07<00:02,  1.06s/sample]
Beam search progress::  88%|████████▊ | 7/8 [00:08<00:00,  1.20sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:08<00:00,  1.03s/sample]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:01,  3.90sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:00<00:02,  2.33sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:01<00:01,  2.83sample/s]
Beam search progress::  50%|█████     | 4/8 [00:01<00:01,  2.82sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:01<00:01,  2.96sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:02<00:00,  4.07sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:02<00:00,  3.50sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:01,  4.66sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:00<00:01,  3.07sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:00<00:01,  3.37sample/s]
Beam search progress::  50%|█████     | 4/8 [00:01<00:01,  3.71sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:01<00:00,  4.52sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:01<00:00,  4.63sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:01<00:00,  3.41sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:02<00:00,  3.80sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:02,  3.14sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:00<00:01,  3.44sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:00<00:01,  4.06sample/s]
Beam search progress::  50%|█████     | 4/8 [00:01<00:01,  3.73sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:01<00:00,  4.10sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:01<00:00,  3.53sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:01<00:00,  3.41sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:02<00:00,  3.69sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:01,  4.48sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:00<00:01,  5.46sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:00<00:01,  4.56sample/s]
Beam search progress::  50%|█████     | 4/8 [00:00<00:00,  4.06sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:01<00:01,  2.10sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:02<00:00,  2.52sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:02<00:00,  2.92sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:02<00:00,  3.02sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:01<00:08,  1.20s/sample]
Beam search progress::  25%|██▌       | 2/8 [00:01<00:03,  1.53sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:01<00:02,  1.84sample/s]
Beam search progress::  50%|█████     | 4/8 [00:02<00:01,  2.50sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:02<00:00,  3.01sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:02<00:00,  3.74sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:02<00:00,  4.22sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:02<00:00,  2.85sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:02,  3.21sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:00<00:01,  4.11sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:00<00:01,  3.69sample/s]
Beam search progress::  50%|█████     | 4/8 [00:01<00:01,  3.88sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:01<00:00,  3.52sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:01<00:00,  4.00sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:02<00:00,  3.79sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:03,  2.03sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:00<00:02,  2.38sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:01<00:01,  2.82sample/s]
Beam search progress::  50%|█████     | 4/8 [00:01<00:01,  2.05sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:02<00:01,  2.02sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:02<00:00,  2.04sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:03<00:00,  2.36sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:03<00:00,  2.20sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:02,  3.45sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:01<00:03,  1.61sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:01<00:02,  2.48sample/s]
Beam search progress::  50%|█████     | 4/8 [00:01<00:01,  2.66sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:01<00:01,  2.67sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:02<00:00,  3.06sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:02<00:00,  3.19sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:02<00:00,  2.96sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:01,  4.56sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:00<00:01,  3.36sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:00<00:01,  2.95sample/s]
Beam search progress::  50%|█████     | 4/8 [00:01<00:01,  3.48sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:01<00:00,  3.49sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:01<00:00,  3.66sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:02<00:00,  3.20sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:02<00:00,  3.38sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:02,  2.89sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:00<00:02,  2.42sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:00<00:01,  3.43sample/s]
Beam search progress::  50%|█████     | 4/8 [00:01<00:01,  3.47sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:01<00:00,  4.00sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:01<00:00,  4.34sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:01<00:00,  3.71sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:02<00:00,  3.33sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:01,  4.34sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:00<00:01,  3.41sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:00<00:01,  3.38sample/s]
Beam search progress::  50%|█████     | 4/8 [00:01<00:00,  4.15sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:01<00:00,  4.21sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:01<00:00,  4.00sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:01<00:00,  4.42sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:01,  5.84sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:01<00:03,  1.79sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:01<00:03,  1.50sample/s]
Beam search progress::  50%|█████     | 4/8 [00:02<00:02,  1.95sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:02<00:01,  2.25sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:03<00:01,  1.34sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:03<00:00,  1.70sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:05<00:00,  1.59sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:01,  4.85sample/s]
Beam search progress::  38%|███▊      | 3/8 [00:01<00:03,  1.64sample/s]
Beam search progress::  50%|█████     | 4/8 [00:02<00:02,  1.93sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:02<00:01,  2.44sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:03<00:01,  1.44sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:05<00:00,  1.04sample/s]
Beam search progress:: 100%|██████████| 8/8 [00:05<00:00,  1.53sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:00<00:01,  4.48sample/s]
Beam search progress::  25%|██▌       | 2/8 [00:02<00:06,  1.14s/sample]
Beam search progress::  38%|███▊      | 3/8 [00:03<00:05,  1.20s/sample]
Beam search progress::  50%|█████     | 4/8 [00:03<00:03,  1.21sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:05<00:03,  1.14s/sample]
Beam search progress::  75%|███████▌  | 6/8 [00:07<00:02,  1.46s/sample]
Beam search progress:: 100%|██████████| 8/8 [00:07<00:00,  1.07sample/s]

Beam search progress::   0%|          | 0/8 [00:00<?, ?sample/s]
Beam search progress::  12%|█▎        | 1/8 [00:02<00:17,  2.48s/sample]
Beam search progress::  25%|██▌       | 2/8 [00:02<00:07,  1.19s/sample]
Beam search progress::  38%|███▊      | 3/8 [00:03<00:03,  1.29sample/s]
Beam search progress::  50%|█████     | 4/8 [00:03<00:02,  1.78sample/s]
Beam search progress::  62%|██████▎   | 5/8 [00:04<00:01,  1.53sample/s]
Beam search progress::  75%|███████▌  | 6/8 [00:04<00:01,  1.72sample/s]
Beam search progress::  88%|████████▊ | 7/8 [00:06<00:01,  1.07s/sample]
Beam search progress:: 100%|██████████| 8/8 [00:06<00:00,  1.17sample/s]

Beam search progress::   0%|          | 0/2 [00:00<?, ?sample/s]
Beam search progress::  50%|█████     | 1/2 [00:02<00:02,  2.27s/sample]
Beam search progress:: 100%|██████████| 2/2 [00:02<00:00,  1.28s/sample]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        test_wer            0.07891332358121872
 
