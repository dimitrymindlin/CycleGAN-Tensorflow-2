# Tensorflow 2 implementation of ABC-GAN.

Based on [CycleGAN-Tensorflow-2](https://github.com/LynnHo/CycleGAN-Tensorflow-2)

Paper: [ABC-GAN... ADD LINK from ARXIV]()

Author: [Dimitry Mindlin](https://github.com/dimitrymindlin) *et al.*

## Main Contribution

#TODO: Include the main contribution of the paper as image

## Exemplar results

### TODO: MURA dataset

row 1: summer -> winter -> reconstructed summer, row 2: winter -> summer -> reconstructed winter

<p align="center"> <img src="./pics/summer2winter.jpg" width="100%" /> </p>

### TODO: RSNA dataset

row 1: horse -> zebra -> reconstructed horse, row 2: zebra -> horse -> reconstructed zebra

<p align="center"> <img src="./pics/horse2zebra.jpg" width="100%" /> </p>

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

- TODO: Dataset
    - Download the mura dataset ... ?
    
            ```console
            sh ./download_dataset.sh mura
            ```

    - download the summer2winter dataset

        ```console
        sh ./download_dataset.sh summer2winter_yosemite
        ```

    - download the horse2zebra dataset

        ```console
        sh ./download_dataset.sh horse2zebra
        ```

    - see [download_dataset.sh](./download_dataset.sh) for more datasets

- Example of training

    ```console
    CUDA_VISIBLE_DEVICES=0 python train.py --dataset summer2winter_yosemite
    ```

    - tensorboard for loss visualization

        ```console
        tensorboard --logdir ./output/summer2winter_yosemite/summaries --port 6006
        ```

- Example of testing

    ```console
    CUDA_VISIBLE_DEVICES=0 python test.py --experiment_dir ./output/summer2winter_yosemite
    ```

# Changes from the original CycleGAN

- Introducing abc_gan.py in attention_strategies folder to do the training step with the attention mechanism
- With that, attention_maps.py and image_segmentation.py help to create and store information about the image such as
  background, foreground and attention map
- 
