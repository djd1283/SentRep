"""This file only exists to allow me to experiment with BERT models in Pytorch."""
import torch
from transformers import *

print('About to test BERT architecture')

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model = BertModel.from_pretrained(pretrained_weights)

input_ids = torch.tensor([tokenizer.encode('this is a test of the BERT tokenizer.')])

result = model(input_ids)[-2:]

parameters = [p for p in model.parameters() if p.requires_grad]

import pdb; pdb.set_trace()















