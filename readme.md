# (CVPR 2024) CGI-DM: Digital Copyright Authentication for Diffusion Models via Contrasting Gradient Inversion

## Requirements:
Run following code to install requirements:

```
pip install -r requirements.txt
```

## Training:
We provide sample code for fine-tuning. Run following code to finetune a model from vangogh's paintings and from sampled dog's images:

```
python Trainer.py

```
The model checkpoints will be saved in path 'db_prior'.

## CGI-DM:
Run following code the remove and reconstruct partial information of CGI-DM:

```
python Extractor.py
```

The partial representation and the reconstructed images are saved in "Recovered_Samples".

## Validating:
To measure copyright authentication effects of CGI-DM, run following code:

```
python Validator.py
```

The terminal will show the Acc. and AUC of CGI-DM.

## Citation:

```
@inproceedings{liang2023adversarial,
  title={Adversarial example does good: Preventing painting imitation from diffusion models via adversarial examples},
  author={Liang, Chumeng and Wu, Xiaoyu and Hua, Yang and Zhang, Jiaru and Xue, Yiming and Song, Tao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  booktitle={International Conference on Machine Learning},
  pages={20763--20786},
  year={2023},
  organization={PMLR}
}
```