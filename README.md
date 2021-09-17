
---

<div align="center">

# An Ensemble Approach for Patient Prognosis of Head and Neck Tumor Using Multimodal Data

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description
Accurate prognosis of a tumor can help doctors provide a proper course of treatment and, therefore, save the lives of many. Traditional machine learning algorithms have been eminently useful in crafting prognostic models in the last few decades. Recently, deep learning algorithms have shown significant advancement and improvement when developing diagnosis and prognosis solutions to different healthcare problems. However, most of these solutions rely solely on either imaging or clinical data. Utilizing patient tabular data such as demographics and patient medical history along side imaging data in a multimodal approach to solve a prognosis task has started to gain more interest recently and has the potential to create more accurate solutions. The main issue when using clinical and imaging data to train a deep learning model is to decide on how to combine the information from these sources. We propose a multimodal network that ensembles deep mutli-task logistic regression (MTLR) and Cox proportional hazard model (CoxPH) models to predict prognostic outcomes for patients with head and neck tumors using patient's clinical and imaging (CT and PET) data. Features from CT and PET scans are fused and combined with patient's electronic health records for the prediction. The proposed model is trained and tested on 224 and 110 patient records respectively. Experimental results show that our proposed ensemble solution achieves C-Index of 0.72. 

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
bash bash/setup_conda.sh

# install requirements
pip install -r requirements.txt
```

Train model with default configuration
```yaml
# data
create the data folder and copy the downloaded data to it. Also, update all the data paths in the config files. 

# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```yaml
python run.py experiment=experiment_name
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Authors
```yaml
Numan Saeed (numanai), Ikboljon Sobirov (ikboljon) and Roba Al Majzoub (musk007)
```

Credits: Hydra-template

<br>
