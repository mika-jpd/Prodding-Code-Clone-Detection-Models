import random

from transformers import AutoTokenizer, AutoModel
import json
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm.auto import tqdm

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
device = torch.device('cpu')

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(1536, 768)
        self.dropout = nn.Dropout(0.5)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead()
        #self.args = args

    def forward(self, input_ids=None, labels=None):
        #input_ids = input_ids.view(-1, 514)
        #output_id1 = self.encoder(input_ids=id1, attention_mask=id1.ne(1))
        #output_id1 = output_id1[0]

        #output_id2 = self.encoder(input_ids=id2, attention_mask=id2.ne(1))
        #output_id2 = output_id2[0]
        output = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        #logits = self.classifier(torch.cat((output_id1, output_id2), dim=2))
        logits = self.classifier(output)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class CloneDataset(Dataset):  # dataset
    # here x will be a list of tuples
    def __init__(self, x1: list, x2:list, label:list):

        #convert to tuple of list
        self.code1 = x1
        self.code2 = x2
        self.label = label

        self.length = len(self.code1)

    def __getitem__(self, idx):
        return self.code1[idx], self.code2[idx], self.label[idx]

    def __len__(self):
        return self.length

def test(urls_to_code:dict, indx_test:list, model_path:str):
    #load the model
    codebert = AutoModel.from_pretrained("microsoft/codebert-base")

    model = Model(
        encoder=codebert,
        tokenizer=tokenizer
    )

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    test_data = {}
    pb = tqdm(total=2000)
    with torch.no_grad():
        for (url1, url2, label) in indx_test:
            # test for each percentage kept
            percentage_keep = [0]

            #original_url1 = urls_to_code[url1]['original'][None, :]
            #original_url2 = urls_to_code[url2]['original'][None, :]

            original_url1 = torch.tensor(urls_to_code[url1]['original'])[None, :]
            original_url2 = torch.tensor(urls_to_code[url2]['original'])[None, :]

            output = model(torch.cat((original_url1, original_url2), dim=0)).to(torch.float32)

            test_data[f'({url1}, {url2}, {label})'] = {}
            test_data[f'({url1}, {url2}, {label})']['original'] = output
            test_data[f'({url1}, {url2}, {label})']['label'] = label
            pass
            pb.update(1)

            for p in percentage_keep:
                permutations_url1 = [torch.tensor(i)[None,:] for i in urls_to_code[url1]['shuffled'][f'{p}']]
                permutations_url2 = [torch.tensor(i)[None,:] for i in urls_to_code[url2]['shuffled'][f'{p}']]

                label_pair = [label for i in range(len(permutations_url1))]
                test_data[f'({url1}, {url2}, {label})'][f'{p}'] = []

                model.eval()
                # here loop through the different permutations and load f1 score in the test data array
                dataset = CloneDataset(permutations_url1, permutations_url2, label_pair)
                data_loader = DataLoader(dataset, batch_size=1)

                fp = 0
                tp = 0
                fn = 0
                tn = 0

                count_val = 0
                for _, data_val in enumerate(data_loader):
                    # check what dataval is
                    c1, c2, l = data_val
                    l = int(l)
                    c1 = torch.squeeze(c1, dim=0)
                    c2 = torch.squeeze(c2, dim=0)

                    outputs = model(torch.cat((c1, c2), dim=0))
                    output_indice = int(outputs.topk(1, dim=1).indices)

                    tp = 0
                    tn = 0
                    fp = 0
                    fn = 0

                    if l == 1:
                        if output_indice == 1:
                            tp +=1
                        else:
                            fp +=1
                    else:
                        if output_indice == 0:
                            tn += 1
                        else:
                            fn += 1
                    count_val += 1

                precision = None
                recall = None
                f1 = None
                try:
                    precision = tp/(tp + fp)
                except:
                    continue
                try:
                    recall = tp/(tp + fn)
                except:
                    continue
                try:
                    f1 = (2*(precision*recall))/(precision + recall)
                except:
                    continue
                accuracy = (tp + tn)/count_val
                test_data[f'({url1}, {url2}, {label})'][f'{p}']['accuracy'].append(accuracy)
                test_data[f'({url1}, {url2}, {label})'][f'{p}']['tp'].append(tp)
                test_data[f'({url1}, {url2}, {label})'][f'{p}']['tn'].append(tn)
                test_data[f'({url1}, {url2}, {label})'][f'{p}']['fp'].append(fp)
                test_data[f'({url1}, {url2}, {label})'][f'{p}']['precision'].append(precision)
                test_data[f'({url1}, {url2}, {label})'][f'{p}']['recall'].append(recall)
                test_data[f'({url1}, {url2}, {label})'][f'{p}']['f1'].append(f1)

        pb.close()
    return test_data

def load_data_indx (path_url_code_short:str, path_indx_test:str):
    # get the url to code data
    with open(path_url_code_short, 'r') as f:
        data = r'' + f.read()

        url_to_code = json.loads(data)

    # get the indx
    indx_test = []
    with open(path_indx_test) as f:
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if label == '0':
                label = 0
            else:
                label = 1
            indx_test.append((url1, url2, label))
    return url_to_code, indx_test

path_url_code_short = "D:\\School\\Winter 2022\\Comp 599\\json_data.json"
path_indx_test = "D:\\School\\Winter 2022\\Comp 599\\indx_test.txt"
model_path = "D:\\School\\Winter 2022\\Comp 599\\model.bin"

url_to_code, indx_test = load_data_indx(path_url_code_short, path_indx_test)

labels = [l for (c1, c2, l) in indx_test]
random.shuffle(labels)

indx_test = [(c1, c2, l) for ((c1, c2, x), l) in zip(indx_test, labels)]
t = torch.cuda.is_available()

data_test = test(
    urls_to_code=url_to_code,
    indx_test=indx_test,
    model_path=model_path
)

with open('D:\\School\\Winter 2022\\Comp 599\\json_experiment.json', 'w') as outfile:
    json.dump(data_test, outfile)