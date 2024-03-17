# CVPR 7124: CGI-DM: Digital Copyright Authentication for Diffusion Models via Contrasting Gradient Inversion

## Requirements:
Run following code to install requirements:

`
pip install -r requirements.txt
`

## Training:
We provide sample code for fine-tuning. Run following code to finetune a model from vangogh's paintings and from sampled dog's images:

`
python Trainer.py
`

The model checkpoints will be saved in path 'db_prior'.

## CGI-DM:
Run following code the remove and reconstruct partial information of CGI-DM:

`
python Extractor.py
`

The partial representation and the reconstructed images are saved in "Recovered_Samples".

## Validating:
To measure copyright authentication effects of CGI-DM, run following code:

`
python Validator.py
`

The terminal will show the Acc. and AUC of CGI-DM.