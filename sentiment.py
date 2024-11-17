from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
data = ["I love you", "I hate you"]
print(sentiment_pipeline(data))

specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
print(specific_model(data))