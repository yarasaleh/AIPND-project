## AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Requirements   
- PyTorch 0.4 (or 0.3)   
- Aconda 

  
 #### To run file train.py    
   run `python3 train.py flowers` you can add optinal arguments , to choose architecture: `--arch "vgg16"` and set hyperparameters: `--learning_rate 0.01 --hidden_units 512 --epochs 20` to use GPU for training : `--gpu`   
 #### To run file predict.py    
   run `python3 predict.py flowers/test/1/image_06743.jpg checkpoint` you can add optinal arguments to return top K most likely classes: `--top_k 3` to use GPU for inference: `--gpu`

