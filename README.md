# Image-Classifier

## Project basics
This project contains a full Deep learning botanical Image classifier with over 100 identification categories,
built using python and pytorch.

## The model
The application has a built in command line tool used to pass parameters for the model.
The user is able to choose between 3 different Neural Network models: VGG11/Alexnet/Densenet121

The classifier will use the model to train and later predict the image passed to the classifier.

## Dependencies
* Python >= 3.7
* Anaconda >= 4.8
* PyTorch >= 1.5

## Classifier Usage
Basic instructions
1. After cloning the repo, you can run this command to train the model: `python train.py --save_dir saving --arch densenet121 --hidden_units 300 --epochs 1 --gpu`
train.py is the file to train the model. 
### Training
* `--save_dir` is the folder where your model will be saved for the prediction step,
* `--arch` is one of 3 architectures you can choose,
* `--hidden_units` amount you want to use from the entry layer to hidden layer, do not use values above 900 as the model with with less entry units has aprox 1000 units,
* `--epochs` is how many times you want to train the model, do not choose a high number or it will take too long, you can get very good results between 2-5, 
* `--gpu` is the option to use the GPU for calculations, if available, calculations are much faster with the GPU.

* You can find the complete list of arguments in get_input_args.py file

### Inference
2. Prediction, after training the model you can use the prediction function to identify the image category.
example of command line to identify an image category `python predict.py flowers/test/1/image_06743.jpg checkpoint --top_k 3 --category_names cat_to_name.json --gpu`
* First argument is the image file path
* Second argument is the path where you saved your checkpoint, if no path was provided during training it will be saved in the same folder of the program, and you only need to pass 
`checkpoint`, if you choose like in the example, savinf folder the path will be `saving/checkpoint`
* `--top_k` amount of categories you want returned
* `--category_names` file mapping the category index with their names
