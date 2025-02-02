import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader


model = SentenceTransformer("arielen/fine-tuned-mpnet-v3")
df = pd.read_csv("fine_tune.csv", sep=",", encoding="utf-8")

train_examples = [
    InputExample(
        texts=[row["Товар поставщика"], row["Товар в магазине"]],
        label=float(row["correct_match"]),
    )
    for _, row in df.iterrows()
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.ContrastiveLoss(model)

sentences1 = df["Товар поставщика"].tolist()
sentences2 = df["Товар в магазине"].tolist()
scores = df["correct_match"].astype(float).tolist()

evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=10,
    warmup_steps=int(len(train_dataloader) * 0.1),
    evaluation_steps=100,
    save_best_model=True,
    output_path="fine_tuned_mpnet_v4",
)

model.save("fine_tuned_mpnet_v4")
print("✅ Модель сохранена в fine_tuned_mpnet_v31")
