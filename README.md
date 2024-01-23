# Fault Detection ResNet
Given some defective iPhone cases we trained a binary image classifier using ResNet-50 as a base model to detect defective iPhone cases. 

### Setting up Instructions

First clone the repository for getting testing codes and dataset images

```
git clone https://github.com/gargsid/fault-detection-cv.git
```

We used PyTorch and Python for this project. You can create the environment using `environment.yml` file

```
conda env create -f environment.yml
conda activate env
```

### Testing the model

To test the model, run `predict.py` file. It will load the trained model and run it on the test images. By running, we the following results

```
python predict.py

# Image: 03.jpg Label: Defective Predicted: Defective
# Image: 39.jpg Label: Defective Predicted: Defective
# Image: 82.jpg Label: Defective Predicted: Defective
# Image: 48.jpg Label: Not Defective Predicted: Not Defective
# Image: 49.jpg Label: Not Defective Predicted: Defective
# Image: 66.jpg Label: Not Defective Predicted: Not Defective
# Image: 11.jpg Label: Not Defective Predicted: Not Defective
# Image: 04.jpg Label: Not Defective Predicted: Not Defective
# Image: 56.jpg Label: Not Defective Predicted: Not Defective
# Image: 96.jpg Label: Not Defective Predicted: Not Defective
# Image: 86.jpg Label: Not Defective Predicted: Not Defective
# Image: 46.jpg Label: Not Defective Predicted: Not Defective
```

Great! We are able to identify all the defective cases and got only 1 False positive where it predictive a non-defective case as faulty. This makes our model a really good classifier with high recall and precision. 

| Metric  | Validation  | 
|---|---|
| Recall  | 1  | 
| F1  | 0.876  | 
|---|---|


### Training the model

For training the model, you can edit the `config` dictionary in `main.py` file for changing the hyper-parameters. The model will save the weights in `saved_models` directory. To run the training use 

```
python main.py
```

