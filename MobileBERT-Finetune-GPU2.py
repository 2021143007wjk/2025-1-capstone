import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer, get_linear_schedule_with_warmup, logging
import matplotlib.pyplot as plt

# 0. GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)

# 1. 경고 제거
logging.set_verbosity_error()

# 2. 데이터 로딩 및 클래스 불균형 오버샘플링
path = 'new_label_data.csv'
df = pd.read_csv(path, encoding='utf-8')
df_all = pd.DataFrame({'text': df['title_comment'], 'target': df['target']})
print(df_all['target'].value_counts())
# 클래스 불균형 확인 및 오버샘플링
df_majority = df_all[df_all['target'] == 0]
df_minority = df_all[df_all['target'] == 1]

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

print("리샘플링 후 클래스 분포:\n", df_balanced['target'].value_counts())

data_X = df_balanced['text'].tolist()
labels = df_balanced['target'].tolist()

# 3. 토큰화
tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
input_ids = inputs['input_ids'].tolist()
attention_mask = inputs['attention_mask'].tolist()

# 4. 학습/검증 데이터 분할
train_ids, val_ids, train_y, val_y = train_test_split(input_ids, labels, test_size=0.2, random_state=2025)
train_masks, val_masks, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2025)

# 5. 데이터로더 설정
batch_size = 16
train_data = TensorDataset(torch.tensor(train_ids), torch.tensor(train_masks), torch.tensor(train_y))
val_data = TensorDataset(torch.tensor(val_ids), torch.tensor(val_masks), torch.tensor(val_y))

train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# 6. 모델 및 옵티마이저 설정
model = MobileBertForSequenceClassification.from_pretrained('mobilebert-uncased', num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 12
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*epochs)

# 7. 클래스 가중치 설정
label_counts = Counter(labels)
total = sum(label_counts.values())
weights = [total / label_counts[i] for i in range(2)]
class_weights = torch.tensor(weights).float().to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# 8. 학습 루프

train_losses = []
train_accuracies = []
val_accuracies = []
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        b_ids, b_mask, b_labels = [x.to(device) for x in batch]
        model.zero_grad()
        outputs = model(b_ids, attention_mask=b_mask)
        loss = loss_fn(outputs.logits, b_labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)

    # 평가 (학습/검증 정확도)
    def evaluate(loader, desc):
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                b_ids, b_mask, b_labels = [x.to(device) for x in batch]
                logits = model(b_ids, attention_mask=b_mask).logits
                pred = torch.argmax(logits, dim=1)
                preds.extend(pred.cpu().numpy())
                truths.extend(b_labels.cpu().numpy())
        acc = np.mean(np.array(preds) == np.array(truths))
        return acc

    train_acc = evaluate(train_loader, f"Evaluating Train Epoch {epoch+1}")
    val_acc = evaluate(val_loader, f"Evaluating Val Epoch {epoch+1}")
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    print(f"\nEpoch {epoch+1} -- Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n")

# 9. 모델 저장
model.save_pretrained('mobliebert_custom_model_imdb_sample5.pt')
print("모델 저장 완료.")

# 10. 📈 Accuracy 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', marker='x')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(range(1, epochs + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()

# 11. 📋 표로 정리
metrics_df = pd.DataFrame({
    'Epoch': list(range(1, epochs + 1)),
    'Train Loss': train_losses,
    'Train Accuracy': train_accuracies,
    'Validation Accuracy': val_accuracies
})
print("\n=== Epoch별 학습 결과 ===")
print(metrics_df.to_string(index=False))

# 저장도 가능
metrics_df.to_csv("training_metrics.csv", index=False)

