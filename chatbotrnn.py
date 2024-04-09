import tensorflow as tf
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Tokenize function
def tokenize(data):
    return word_tokenize(data)

# Encode sentiment labels
label_map = {"positive": 0, "negative": 1, "neutral": 2}

data = [("This is a good game", "positive"),
    ("I don't like it", "negative"),
    ("I want a more better approach", "neutral"),
    ("nice", "positive"),
    ("i don't", "negative"),
    ("what the hell you made", "negative"),
    ("make it more good", "positive"),
    ("what fuck is this", "negative"),
    ("way better than previous", "positive"),
    ("for me its quite good", "positive"),
    ("well it's nice", "positive")]

# Tokenize and encode data
tokenized_data = [(tokenize(sentence), label_map[label]) for sentence, label in data]

# Build vocabulary
vocab = set()
for sentence, _ in tokenized_data:
    vocab.update(sentence)

# Create word-to-index mapping
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Convert data to index sequences
indexed_data = [([word_to_idx[word] for word in sentence], label) for sentence, label in tokenized_data]

# Padding sequences
max_seq_length = max(len(sentence) for sentence, _ in indexed_data)
padded_data = [(pad_sequences([sentence], maxlen=max_seq_length, padding='post')[0], label) for sentence, label in indexed_data]

# Split data into train and test sets
train_data, test_data = train_test_split(padded_data, test_size=0.2, random_state=42)

# Prepare data for TensorFlow Dataset
train_sentences = [sentence for sentence, label in train_data]
train_labels = [label for sentence, label in train_data]

test_sentences = [sentence for sentence, label in test_data]
test_labels = [label for sentence, label in test_data]

train_sentences = np.array(train_sentences)
train_labels = np.array(train_labels)

test_sentences = np.array(test_sentences)
test_labels = np.array(test_labels)

# Create TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels)).shuffle(len(train_data)).batch(2)
test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels)).batch(2)

# Define model architecture
class SimpleRNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.SimpleRNN(hidden_dim)
        self.fc = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        embedded = self.embedding(inputs)
        rnn_output = self.rnn(embedded)
        output = self.fc(rnn_output)
        return output

# Model parameters
vocab_size = len(vocab) + 1  # Add 1 for padding token
embedding_dim = 64
hidden_dim = 64
output_dim = 3  # Number of sentiment classes

# Instantiate the model
model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim)

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    total_batches = len(train_dataset)
    for batch, (inputs, labels) in enumerate(train_dataset, 1):
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        running_loss += loss.numpy()
        
        # Print training progress
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Batch [{batch}/{total_batches}], Loss: {loss.numpy():.4f}", end="")
        
    # Print epoch summary
    print(f"\rEpoch [{epoch+1}/{num_epochs}], Loss: {running_loss/total_batches:.4f}")

# Define a function to preprocess new sentences
def preprocess_sentence(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Convert tokens to indices using the word_to_idx mapping
    indices = [word_to_idx.get(token, 0) for token in tokens]
    # Pad the sequence to match the maximum sequence length
    padded_sequence = pad_sequences([indices], maxlen=max_seq_length, padding='post')
    return padded_sequence

# Define a function to predict sentiment for a given sentence
def predict_sentiment(sentence):
    # Preprocess the sentence
    input_sequence = preprocess_sentence(sentence)
    # Make predictions using the model
    predictions = model.predict(input_sequence)
    # Get the predicted class (index with highest probability)
    predicted_class = np.argmax(predictions)
    # Map the predicted class index to the sentiment label
    sentiment_label = {0: "positive", 1: "negative", 2: "neutral"}[predicted_class]
    return sentiment_label

# Test the model on a new sentence
new_sentence = "This movie is great!"
predicted_sentiment = predict_sentiment(new_sentence)
print(f"The sentiment of the sentence '{new_sentence}' is: {predicted_sentiment}")
