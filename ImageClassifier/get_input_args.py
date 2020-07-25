import argparse


def get_input_args():

    """
    Retrieves and parse the command line arguments provided by the user
    when they run the program from a terminal window.
    This function uses argparse module to create and define these arguments.
    If the user fails to provide the arguments, default values will be used for 
    the missing arguments
        1. Directory where the images to train the model are as --data_dir 
        1. Directory where the training checkpoint will be saved as --save_dir
        2. Model architecture name to be used to construct the model for as --arch  eg.'vgg11, alexnet, densenet'
        3. Model learning rate to train the model as --learning_rate
        4. Model hidden units to train the model as --hidden_units
        5. Model epochs to train the model as --epochs
        6. Option to use GPU for training as --gpu 
        
    This function returns these arguments as an ArgumentParser object.
    """

    parser = argparse.ArgumentParser()

    #Create the command line arguments mentioned above

    parser.add_argument('--data_dir', type=str, default='flowers/', help = 'path to the folder containing the training images')
    parser.add_argument('--save_dir', type=str, default='ImageClassifier/', help = 'path to the folder where the checkpoint file will be saved')
    parser.add_argument('--arch', type=str, default='vgg11', help = 'Covolutional Neural Network model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help = 'Learning rate value used for the gradient descent')
    parser.add_argument('--hidden_units', type=int, default=256, help = 'Number of hidden units used to create the classifier')
    parser.add_argument('--epochs', type=int, default=5, help = 'Number of periods to loop through, training the model')
    parser.add_argument('--gpu', action='store_true', help = 'Option to use gpu for training, if not specified, it will use the cpu')
    
    return parser.parse_args()


def get_prediction_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("files",nargs="*")
    parser.add_argument('--top_k', type=int, default=5, help = 'Number of the most commom classes returned')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help = 'File with the mapping from index for categories')
    parser.add_argument('--gpu', action='store_true', help = 'Option to use gpu for training, if not specified, it will use the cpu')
    return parser.parse_args()


    
