# Tensorflow 2 implementation of ABC-GAN.

Based on [CycleGAN-Tensorflow-2](https://github.com/LynnHo/CycleGAN-Tensorflow-2)

Paper: [ABC-GAN... ADD LINK]()

Author: [Dimitry Mindlin](https://github.com/dimitrymindlin) *et al.*

## Main Contribution

We propose a novel counterfactual generation method for location specific counterfactuals.


<p align="center"> <img src="./pics/Linear-ABC-GAN-Mura.svg" width="70%" /> </p>

### Experiments with MURA dataset

abnormal -> normal counterfactuals. Green boxes indicate valid counterfactuals, red boxes indicate invalid

<p align="center"> <img src="./pics/ABC-GAN-Comparison-MURA.drawio.svg" width="90%" /> </p>

Check out the paper for more results.

# TODO: Usage

- Environment

    - Python 3.6

    - TensorFlow 2.2, TensorFlow Addons 0.10.0

    - OpenCV, scikit-image, tqdm, oyaml

    - *we recommend [Anaconda](https://www.anaconda.com/distribution/#download-section)
      or [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers), then you can create the
      TensorFlow 2.2 environment with commands below*

        ```console
        conda create -n tensorflow-2.2 python=3.6

        source activate tensorflow-2.2

        conda install scikit-image tqdm tensorflow-gpu=2.2

        conda install -c conda-forge oyaml

        pip install tensorflow-addons==0.10.0
        ```

    - *NOTICE: if you create a new conda environment, remember to activate it before any other command*

        ```console
        source activate tensorflow-2.2
        ```

### Dataset to replicate the experiments

- Download the MURA dataset here: [MURA](https://stanfordmlgroup.github.io/competitions/mura/)
- Download the RSNA dataset here: [RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/)

#### Prepare Mura dataset

We use Histogram Equalization to preprocess the MURA Wrist images. This is done in the
mura_data/data/transform_dataset.py file.

run the following command to create the transformed dataset:

```console 
python transform_dataset.py --dataset_dir <path to the MURA dataset> --output_dir <path to the output directory>
```

### Training the classification model

ABC-GAN and GANterfactual require a classification model to be trained before the training of the GANs.
We used the inception v3 model for the mura dataset and an alexnet for the RSNA data as originally proposed in the
[GANterfactual paper](https://www.frontiersin.org/articles/10.3389/frai.2022.825565/full).

The training script for the inception model on the MURA dataset can be found at
classifiers/MURA/train_mura_classifier.py

The training script for the alexnet model on the RSNA dataset can be found at the original repository of
[GANterfactual](https://github.com/hcmlab/GANterfactual/blob/main/GANterfactual/train_alexNet.py)

### Training ABC-GAN

Once the classifier is trained, the ABC-GAN and GANterfactual models can be trained. train.py is responsibel to do that
with the provided arguments.
Change the following arguments to replicate the experiments from the paper:

- dataset: mura or rsna
- counterfactual_loss_weight: 1 to include it for ABC-GAN and GANterfactual, 0 for normal CycleGAN
- identity_loss_weight: Check in paper which versions use this loss (=1) and which don't (=0)
- cyclegan_mode: abc-gan or ganterfactual or cycle-gan
- clf_name: inception for mura, alexnet for rsna
- clf_ckp_name: name of the checkpoint of the classifier that was trained before
- clf_input_channels: 3 for inception (mura), 1 for alexnet (rsna)
- start_attention_epoch: defines when the attention strategy should be applied. Experiments showed that
  pretraining without attention for 10 epochs and then with attention yield the best results.
- discriminator: whether to use the normal patchgan discriminator that acts on the whole image or the "attentive"
  version
  where only the attended area is passed to the discriminator.

# Changes from the original CycleGAN Repository

- Introducing abc_gan.py in attention_strategies folder to do the training step with the attention mechanism
- With that, attention_maps.py and image_segmentation.py help to create and store information about the image such as
  background, foreground and attention map
- classifiers folder that contains the training scripts for the classifiers
- rsna_data folder that contains the scripts to preprocess the rsna dataset
- mura_data folder that contains the scripts to preprocess the mura dataset
- 
