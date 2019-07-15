# PROGRAMMER: Yara Saleh
# Date: 4 July 2019

import argparse
import json
import numpy as np
import torch
from predict_functions import *
def main():
    args = get_arguments()
    cuda = args.cuda
    model = load_checkpoint('checkpoint.pth', cuda)
    model.idx_to_class = dict([[v,k] for k, v in model.class_to_idx.items()])

    with open(args.categories , 'r') as f:
        cat_to_name = json.load(f)
    probabilities = predict(args.input,model,args.cuda,topk=int(args.top_K))
    #print([cat_to_name[x] for x in classes])
    #print(cat_to_name[classes[0, x]] for x in range(4))
    a = np.array(probabilities[0][0])
    b = [cat_to_name[str(index+1)] for index in np.array(probabilities[1][0])]
    for key in range(len(a)) :
        print(a[key],b[key])



main()
