#!/usr/bin/env python
# coding: utf-8

# ### DL LAB 2A

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# In[2]:


# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
columns = ['letter', 'x-box', 'y-box', 'width', 'height', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar',
           'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
data = pd.read_csv(url, names=columns)


# In[3]:


# 2. Separate features and labels
X = data.drop('letter', axis=1).values
y = data['letter'].values


# In[4]:


# 3. Encode labels (A-Z -> 0-25)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)


# In[5]:


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)


# In[6]:


# 5. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


# 6. Build the DNN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(16,)),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax')  # 26 letters A-Z
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[12]:


# 7. Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)


# In[13]:


# 8. Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")


# In[14]:


model.save("DNN.h5")


# In[15]:


# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')


# In[16]:


# 9. Make predictions (optional)
y_pred = model.predict(X_test)
predicted_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))


# In[19]:


import random

def random_sample_predict(model, scaler, label_encoder, X_test, y_test):
    # Pick a random index
    idx = random.randint(0, len(X_test) - 1)

    # Select random sample
    sample = X_test[idx].reshape(1, -1)
    true_label = np.argmax(y_test[idx])
    true_letter = label_encoder.inverse_transform([true_label])[0]

    # Predict
    prediction = model.predict(sample)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_letter = label_encoder.inverse_transform(predicted_class)[0]

    print(f"\n--- Random Sample Test ---")
    print(f"True Letter: {true_letter}")
    print(f"Predicted Letter: {predicted_letter}")

# Call this function after model training
random_sample_predict(model, scaler, label_encoder, X_test, y_test)


# ### DL LAB 2B

# In[ ]:


# 1. Import Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 2. Create a small custom dataset (manually for simplicity)
texts = [
    "The movie was fantastic and thrilling",
    "I hated the movie, it was boring and bad",
    "An excellent movie with brilliant performances",
    "The film was dull and too long",
    "Loved the story and the acting was amazing",
    "Terrible movie, complete waste of time",
    "What a masterpiece, loved every moment",
    "Worst movie ever, so disappointed",
    "Absolutely stunning, a wonderful experience",
    "I regret watching this movie, very bad"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# 3. Tokenize the texts
max_words = 1000
max_len = 20

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# 4. Build the Model
model = keras.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 5. Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train the Model
model.fit(padded_sequences, np.array(labels), epochs=20, batch_size=2, verbose=2)

# 7. Real-time Prediction Function
def predict_sentiment(review):
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positive" if pred >= 0.5 else "Negative"
    print(f"\nReview Sentiment: {sentiment} (Score: {pred:.4f})")

# 8. Real-time Testing
sample_review1 = "The movie was fantastic! I really loved the performances."
predict_sentiment(sample_review1)

sample_review2 = "The film was boring and too long. Not good at all."
predict_sentiment(sample_review2)

sample_review3 = "I absolutely hated this movie. Worst experience ever."
predict_sentiment(sample_review3)

sample_review4 = "An excellent masterpiece. Great story and acting."
predict_sentiment(sample_review4)


# In[ ]:


# 1. Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 2. Load the IMDB dataset (with raw text)
imdb = keras.datasets.imdb

# Set vocabulary size
vocab_size = 10000

# Load dataset (already preprocessed as integers)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 3. Decode function to get back text
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(text_ints):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_ints])

# 4. Prepare data (pad sequences)
maxlen = 200
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# 5. Build model
model = keras.Sequential([
    layers.Embedding(vocab_size, 64, input_length=maxlen),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 6. Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Train model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 8. Real-time testing function
def predict_sentiment_text(model, review_text):
    # 8.1 Preprocessing: convert review to integers
    words = review_text.lower().split()
    review_seq = []
    for word in words:
        idx = word_index.get(word, 2)  # 2 is for unknown words
        review_seq.append(idx)

    review_seq = pad_sequences([review_seq], maxlen=maxlen)

    pred = model.predict(review_seq, verbose=0)[0][0]
    sentiment = "Positive" if pred >= 0.5 else "Negative"
    print(f"\nReview Sentiment: {sentiment} (Score: {pred:.4f})")

# 9. Real examples
sample_review1 = "The movie was fantastic! I really loved the performances."
predict_sentiment_text(model, sample_review1)

sample_review2 = "The film was boring and too long. Not good at all."
predict_sentiment_text(model, sample_review2)

sample_review3 = "it is so disappointing."
predict_sentiment_text(model, sample_review3)

sample_review4 = "An excellent movie. Great direction and amazing acting!"
predict_sentiment_text(model, sample_review4)


# In[ ]:




