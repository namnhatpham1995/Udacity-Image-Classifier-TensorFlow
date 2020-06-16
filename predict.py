import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
import numpy as np
import json

from img_Process import process_image, get_class_names, load_model
import argparse

def predict(image_path, model_path, top_k, all_class_names):
    top_k = int(top_k)
    
    print("\nLoad Trained Model")
    model = load_model(model_path)
    print("\nOpen Image and import to variable")
    img = Image.open(image_path)
    test_image = np.asarray(img)
    print("\nProcessing the image")
    processed_test_image = process_image(test_image)
    

    print("\nFetching prediction probabilities: ")
    print(processed_test_image.shape, np.expand_dims(processed_test_image,axis=0).shape)
    prob_preds = model.predict(np.expand_dims(processed_test_image,axis=0))
    prob_preds = prob_preds[0].tolist()
    
    #Finding class with highest probability
    best_pred_class_id = model.predict_classes(np.expand_dims(processed_test_image,axis=0))
    best_pred_class_prob = prob_preds[best_pred_class_id[0]]
    best_pred_class_name = all_class_names[str(best_pred_class_id[0])]
    print("\n\nThe Flower class with highest probability to the test image:",
          "\n +Class id :",best_pred_class_id, 
          "\n +Class name :", best_pred_class_name, 
          "\n +Class probability :", best_pred_class_prob)
    
    values, indices= tf.math.top_k(prob_preds, k = top_k)
    probs_topk = values.numpy().tolist()#[0]
    classes_topk = indices.numpy().tolist()#[0]
    class_labels = [all_class_names[str(i)] for i in classes_topk]
    class_prob_dict = dict(zip(class_labels, probs_topk))       
    print("\nTop {0} classes with highest probabilities : \n".format(top_k))    
    for count in range(top_k):
        print(" +{0} : {1}\n".format(class_labels[count], probs_topk[count]))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("image_path", help="Test Image Folder", default="")
    parser.add_argument("saved_model", help="Trained Model Folder", default="")
    parser.add_argument("--top_k", help="Fetch top k predictions", required = False, default = 3)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()
    print(args)
    
    all_class_names = get_class_names(args.category_names)
    print("\nAvailable class names:\n",all_class_names)

    print("Begin prediction: \n")
    predict(args.image_path, args.saved_model, args.top_k, all_class_names)
