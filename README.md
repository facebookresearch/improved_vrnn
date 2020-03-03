# Introduction

improved_vrnn is the accompanying code repository for the paper: 
"Improved Conditional VRNNs for Video Prediction" by  Lluis Castrejon, Nicolas Ballas and Aaron Courville.
> Lluis Castrejon, Nicolas Ballas and Aaron Courville, "Improved Conditional VRNNs for Video Prediction," ICCV 2019. [Official ICCV version](http://openaccess.thecvf.com/content_ICCV_2019/papers/Castrejon_Improved_Conditional_VRNNs_for_Video_Prediction_ICCV_2019_paper.pdf) [arxiv version](https://arxiv.org/abs/1904.12165)

If you use this code for your research, please cite the paper.

## Getting Started

### Setup Enviroment

We use the following packages in our environment:

* python=3.6
* ffmpeg=4.0.2
* imageio=2.4.1
* joblib=0.12.5
* matplotlib=3.0.1
* numpy=1.15.4
* opencv=3.4.2
* pillow=5.3.0
* python=3.6.6
* pytorch=0.4.1
* scikit-image=0.14.0
* scipy=1.1.0
* torchvision=0.2.1
* tqdm=4.28.1


### Preparing datasets:

#### Stochastic Moving MNIST

For Stochastic Moving MNIST, download and use the dataloader found on [SVG](https://github.com/edenton/svg/).

#### BAIR Push

For the BAIR Push Dataset, follow the steps in [SVG](https://github.com/edenton/svg/) to download the data and to process the `tfrecords` files.


### Model Training

`main.py` provides the common training pipeline for all datasets.

Example commands:

```
python main.py --out_dir OUTPUT_DIR --exp_name mnist --dataset stochastic --model vrnn --rec_loss bce --n_ctx 10 --n_steps 10 
python main.py --out_dir OUTPUT_DIR --exp_name bair_push --dataset pushbair --model vrnn --n_ctx 2 --n_steps 10 
```  

### Sampling

`sample.py` generates multiple samples for different example using a trained model.

Example commands:

```
python sample.py --checkpoint PATH_TO_MODEL_CHECKPOINT --n_seqs NUMBER_OF_EXAMPLES --n_samples NUMBER_OF_SAMPLES_PER_EXAMPLE
```

# License

Improved VRNNs is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.

# Citation

Please cite it as follows:

```
@InProceedings{Castrejon_2019_ICCV,
author = {Castrejon, Lluis and Ballas, Nicolas and Courville, Aaron},
title = {Improved Conditional VRNNs for Video Prediction},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```
