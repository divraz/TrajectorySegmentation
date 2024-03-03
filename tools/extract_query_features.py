import torch
import os.path
from tqdm import tqdm
import jsonlines
import numpy as np
from transformers import AutoTokenizer, CLIPTextModel

DATA_PATH = "/home/local/ASUAD/draj5/rlbench_data/"
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def extract_features (qid, query):

    path = f"{DATA_PATH}dataset_random/features/clip_text_features/"
    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.isfile (f"{path}qid{qid}.npz"):
        return
    inputs = tokenizer (query, padding = True, return_tensors = "pt")
    outputs = model (**inputs)
    lsh = torch.squeeze (outputs.last_hidden_state.detach (), 0)
    po = torch.squeeze (outputs.pooler_output.detach (), 0)
    #print (lsh.shape, po.shape)
    np.savez (f"{path}qid{qid}", last_hidden_state = lsh, pooler_output = po)

def extract_files (dir_name):
    files = os.listdir (dir_name)
    for name in tqdm (files, desc = dir_name):
        reader = jsonlines.open (f"{dir_name}{name}")
        for data in reader:
            extract_features (data['qid'], data['query'])

if __name__ == '__main__':
    files = os.listdir (f"{DATA_PATH}dataset_random/json/")
    for folder in files:
        extract_files (f"{DATA_PATH}dataset_random/json/{folder}/")
