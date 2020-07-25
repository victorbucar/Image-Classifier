"""
This will be the file where get_inpu_args.py will be called to get the arguments 
for the network.
"""
from get_input_args import get_input_args


def main():
    
    #get the arguments from the user
    in_args = get_input_args()
    
    print(in_args)
    
    
# call for main function    
main()