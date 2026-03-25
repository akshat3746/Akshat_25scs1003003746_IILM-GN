import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 1. Load data
df = pd.read_csv('data.csv')
texts = df['sequence'].astype(str).tolist()
intents = df['intent'].astype(str).tolist()

# 2. Tokenize / sequences
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_len = max(len(seq) for seq in sequences)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# 3. Encode labels
le = LabelEncoder()
labels_int = le.fit_transform(intents)
labels = to_categorical(labels_int)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

# 5. Build model
vocab_size = len(tokenizer.word_index) + 1
num_classes = labels.shape[1]
embedding_dim = 16

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=X_train.shape[1]))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# 6. Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=4,
    validation_split=0.2
)

# 7. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)

# 8. Prediction function
def predict_intent(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=X_train.shape[1], padding='post')
    pred = model.predict(pad)
    idx = np.argmax(pred, axis=1)[0]
    intent_label = le.inverse_transform([idx])[0]
    return intent_label

# 9. Interactive loop
while True:
    user = input("Enter your sequence (or 'quit'): ")
    if user.lower() == 'quit':
        break
    print("Predicted intent:", predict_intent(user))
