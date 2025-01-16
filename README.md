#Hierarchal DreamRec

This repository is based on *Reshaping Sequential Recommendation via Guided Diffusion* presented at NeurIPS 2023 ([arXiv link](https://arxiv.org/abs/2310.20453)) by Zhengyi Yang, Jiancan Wu, Zhicai Wang, Xiang Wang, Yancheng Yuan, and Xiangnan He. In their work, they introduce a guided diffusion recommender system called DreamRec. 

In this project, we extend FreamRec by introducing a decoder, which can be found in the `DreamRec_Movies.py` file. By encoding genre spaces, we created a latent space that enhances the main recommender's ability to predict users' next preferred movie. We refer to our model as *Hierarchical-DreamRec*.

