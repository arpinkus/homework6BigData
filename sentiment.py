# This is the old version the correct one is the ipynb file
# Load transformers pipeline - Part 1
from transformers import pipeline

# Load the dataset (IMDB sentiment analysis) - Part 2
import torch
from datasets import load_dataset
imdb = load_dataset("imdb")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # speed up training with padding
# Load the model - Part 2
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
import numpy as np
import evaluate
from huggingface_hub import notebook_login
notebook_login()
from transformers import TrainingArguments, Trainer


# Built in sentiment analysis pipeline - Part 1
#model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_name = "finiteautomata/bertweet-base-sentiment-analysis"
classifier = pipeline("sentiment-analysis", model=model_name)
results = classifier(["I love you", "I hate you"])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 9)}")

# Load the dataset (IMDB sentiment analysis) - Part 2
torch.cuda.is_available()
small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

# Prepare the text inputs for the model by using the map method - Part 2
def preprocess_function(examples):
   return tokenizer(examples["text"], truncation=True)
 
tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

# Load the model - Part 2
def compute_metrics(eval_pred):
    load_metric = evaluate.load_metric("accuracy")
    load_f1 = evaluate.load_metric("f1")
  
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}
 
repo_name = "finetuning-sentiment-model-3000-samples"
 
training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=True,
)
 
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()



#sentiment_pipeline = pipeline("sentiment-analysis")
#data = ["I love you", "I hate you"]
#print(sentiment_pipeline(data))

#specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
#print(specific_model(data))