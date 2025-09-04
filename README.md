# Wasserstein GAN for Anime Face Generation  

---

## Brief Overview  
In this project, I have learned and implemented a **Wasserstein GAN with Gradient Penalty** in **PyTorch**. 

The aim of this work is to:  
- Explore **stable WGAN training** using Wasserstein loss with Gradient Penalty.
- Display the generator and critic architectures using torchview  
- Generate visually realistic anime faces from noise vectors of dim=100.  
- Evaluate quality using **Fréchet Inception Distance (FID)**.  
- Provide an **interactive demo website** for trying the model with streamlit.  

I developed the project **end-to-end**:  
- Designed **Generator & Critic** architectures  
- Built the **training loop** with visualization & checkpoints  
- Added **evaluation pipeline** with FID scoring  
- Deployed a **Streamlit app** for real-time face generation  

**Try it yourself here:** [Live]([https://your-demo-link.com](https://pradumns-001-gan-project-app-covwvj.streamlit.app/))  

---

##  Output  

Generated anime faces after training (FID ≈ **23.9**):  

<p align="center">
  <img src="results/generated (3).png" width="100"/>
</p>  

---

## Project Structure 
``` 
WGAN-AnimeFaces/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── app.py
├── model.py
│
├── notebooks/
│   ├── WGAN-GP_trainer.ipynb
│   ├── WGAN-GP_evaluator.ipynb
│
├── results/
│   ├── generated (3).png
│   ├── generated (2).png
│   ├── generated (1).png
│   ├── generated.png
│   ├── generator_architecture.png
│   ├── critic_architecture.png
│
├── models/
│   ├── discriminator31.pth
│   ├── generator31.pth
│   ├── filter.pth
```
---

## Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/PradumnS-001/GAN-Project
cd WGAN-AnimeFaces
pip install -r requirements.txt
```
---

### Train Model
```bash
jupyter notebook notebooks/trainer.ipynb
```
### Evaluate Model
```bash
jupyter notebook notebooks/evaluator.ipynb
```
### Run Interactive App
```bash
streamlit run app/app.py
```
---
**Dataset**: Anime Faces Dataset (Kaggle)
Link: https://www.kaggle.com/datasets/splcher/animefacedataset

Note: Dataset is NOT included in this repo. Please download separately.
