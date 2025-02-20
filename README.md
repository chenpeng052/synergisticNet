<h2 align="center">A Synergistic CNN-Transformer Network with Pooling Attention Fusion for Hyperspectral Image Classification</h2>

<p align="center">
<b>Peng Chen, Wenxuan He, Feng Qian, Guangyao Shi, Jingwen Yan</b>  
<br>
<a align="center">College of Engineering, Shantou University, Shantou, 515063, Guangdong, China</a>
</p>

This repository contains the code and models for our paper:  

*Chen, Peng et al. "A Synergistic CNN-Transformer Network with Pooling Attention Fusion for Hyperspectral Image Classification."*  
**Digital Signal Processing**, 2025, Elsevier.  

Feel free to contact us if there is anything we can help. Thanks for your support!
chenpeng052@gmail.com


![fig1](https://github.com/user-attachments/assets/6583a7ba-db8f-4478-bf93-8c01a9718af5)

## Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

2. Prepare your dataset:  
Place the dataset in the `data` folder with the appropriate structure. For example:

```
/data
├── paviaU.mat
├── paviaU_gt.mat
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Train the model:

```bash
python train.py
```

5. Evaluate the model:

```bash
python test.py
```


## Citation

If you find this work helpful, please cite:

```bibtex
@article{chen2025synergistic,
  title={A Synergistic CNN-Transformer Network with Pooling Attention Fusion for Hyperspectral Image Classification},
  author={Chen, Peng and He, Wenxuan and Qian, Feng and Shi, Guangyao and Yan, Jingwen},
  journal={Digital Signal Processing},
  pages={105070},
  year={2025},
  publisher={Elsevier}
}

