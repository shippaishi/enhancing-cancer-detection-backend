# make sure model 'enhance_cancer.pth' is stored in same directory

from transformers import RobertaModel
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from flask import Flask, jsonify, request

app = Flask(__name__)

device = ('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')) #determines whether CUDA GPU or CPU is used
MAX_LEN = 256 #max length for input data
model_name = 'roberta-base' #model name

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe['text']
        self.targets = dataframe['label']
        self.max_len = max_len

    def __len__(self):
      return len(self.comment_text)

    def __getitem__(self, index):
        # split and rejoin sentence to standardize whitespace
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text, #text data in dataframe
            None, #no second input
            add_special_tokens=True, #increases accuracy
            max_length=self.max_len, #max input
            # truncation=True,
            pad_to_max_length=True, #standardizes input length
            return_token_type_ids=True #returns token ids
        )

        '''
        inputs returns
        input_ids: list of token ids that represent input text
        attention_mask: determine which tokens are input and which are padding
        token_type_ids: token type IDs that differentiate between different segments of text
        '''

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # return dictionary with tensor arrays containing ids, mask, token types, and targets (values)
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
class RoBERTaClass(torch.nn.Module):
    def __init__(self):
        super(RoBERTaClass, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base") #layer 1 has roberta model
        self.drop = torch.nn.Dropout(0.3) #layer 2 deactivates (drops out) .30 of neurons while training
        self.linear = torch.nn.Linear(768, 11) #weight matrix and bias vector

    def forward(self, ids, mask, token_type_ids): #run model
        _, output_1 = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        output_2 = self.drop(output_1)
        output = self.linear(output_2)
        return output

model = RoBERTaClass()
model.to(device)

# load model and run inference
model = RoBERTaClass().to(device)
# model.load_state_dict(torch.load('enhance_cancer.pth'))
model = AutoModel.from_pretrained("your_username/my-awesome-model")
tokenizer = AutoTokenizer.from_pretrained(model_name)

keys = {
    '1': 'Sustaining proliferative signaling (PS)',
    '2': 'Evading growth suppressors (GS)',
    '3': 'Resisting cell death (CD)',
    '4': 'Enabling replicative immortality (RI)',
    '5': 'Inducing angiogenesis (A)',
    '6': ' Activating invasion & metastasis (IM)',
    '7': 'Genome instability & mutation (GI)',
    '8': 'Tumor-promoting inflammation (TPI)',
    '9': 'Deregulating cellular energetics (CE)',
    '10': 'Avoiding immune destruction (ID)'
}

# inference code
def inference(symptom):
    inference_dataset = pd.DataFrame({'text': [symptom], 'label': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]})
    inference_set = CustomDataset(inference_dataset, tokenizer=tokenizer, max_len=MAX_LEN)
    inference_loader = DataLoader(inference_set, batch_size=1, shuffle=False)
    
    # run model on input data
    model.eval()
    with torch.no_grad():
        for data in inference_loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            output = model(ids, mask, token_type_ids)

    # process output
    predictions = torch.sigmoid(output).detach().cpu().numpy()[0]
    print(f'predictions: {predictions}') #debugging
    threshold = .5 #accept inputs over this value
    hallmarks = []
    for lID, prob in enumerate(predictions):
        if(prob > threshold):
            hallmarks.append(lID)
    print(f'hallmarks: {hallmarks}') #debugging

    hallmarksDesc = ""
    for lID in hallmarks:
        hallmarksDesc += keys[str(lID)]

    print(hallmarksDesc) #debugging
    return hallmarksDesc

@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = data['symptom']
    result = inference(input_data)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)