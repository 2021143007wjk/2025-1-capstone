import torch
import pandas as pd
import numpy as np
from jinja2.filters import do_lower
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm
from transformers.models.paligemma.convert_paligemma_weights_to_hf import device

GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("using device:", device)

data_path = "new_combdata.csv"
df = pd.read_csv(data_path, encoding="utf-8")
data_X = list(df['title_comment'].values)

print(len(data_X))

tokenizer = MobileBertTokenizer.from_pretrained("mobilebert-uncased", do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs["input_ids"]
attention_mask = inputs['attention_mask']

batch_size = 8
test_inputs = torch.tensor(input_ids)
test_masks = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_masks)
test_sampler = torch.utils.data.RandomSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model = MobileBertForSequenceClassification.from_pretrained("mobliebert_custom_model_imdb_sample5.pt")
model.to(device)

model.eval()

test_pred = []

for batch in tqdm(test_dataloader, desc="Inferencing Full dataset"):
    batch_ids, batch_mask = batch

    batch_ids = batch_ids.to(device)
    batch_mask = batch_mask.to(device)

    with torch.no_grad():
        output = model(batch_ids, attention_mask= batch_mask)
    logits = output.logits
    pred = torch.argmax(logits, dim=1)
    test_pred.extend(pred.cpu().numpy())

# 예측 결과 저장
result_df = pd.DataFrame({
    'text': data_X,
    'predicted_label': test_pred
})

result_csv_path = "unlabeled_prediction_results3.csv"
result_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")

print(f"예측 결과가 {result_csv_path}에 저장되었습니다.")

