# Computer Vision Nanodegree Program, Notebooks

This repository contains code exercises and materials for Udacity's [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891) program. Follow the instructions below to create a local environment with the necesary dependencies to run the notebooks.  



## 1. Create the Environment

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/udacity/CVND_Exercises.git
cd CVND_Exercises
```

2. Create (and activate) a new environment, named `cv` with Python 3.6.
	```
	conda create -n cv python=3.6
	source activate cv
	```


3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	```
	conda install pytorch torchvision -c pytorch 
	```


6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```
