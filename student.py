import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv, find_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import argparse
import pinecone
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import matplotlib.pyplot as plt
import json
from pprint import pprint
# from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input


print("Start.")

# Modell laden
loaded_model = tf.keras.models.load_model("student_performance_model.keras")

# Modellarchitektur anzeigen und in Datei schreiben
json_architecture = loaded_model.to_json()  # Modellarchitektur als JSON
with open("model.json", "w") as json_file:
    json.dump(json.loads(json_architecture), json_file, indent=4)  # Speichere das JSON in der Datei

# Die Konfigurationen jeder Schicht ebenfalls in die Datei schreiben
with open("model.json", "a") as json_file:  # Anhängen an die bestehende Datei
    json_file.write("\n--- Layer Configurations ---\n")
    for layer in loaded_model.layers:
        layer_config = layer.get_config()  # Hol dir die Konfiguration der Schicht
        json.dump(layer_config, json_file, indent=4)  # Schreibe sie in die Datei
        json_file.write("\n")

print("Die Modellarchitektur und -konfiguration wurden in 'model.json' gespeichert.")

# Load environment variables (e.g., Pinecone API key)
load_dotenv(find_dotenv())

# Load the exams.csv file
data = pd.read_csv("exams.csv")

# Preview the first few rows of the data
# print(data.head())

# Define features and target variable
# Convert categorical features to numeric using One-Hot-Encoding
categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
# print("One-Hot-Encoded Data:")
# print(data.head())
X = data.drop(columns=['math score', 'reading score', 'writing score']).apply(pd.to_numeric, errors='coerce').fillna(data.mean())
# y = data[['math score', 'reading score', 'writing score']].mean(axis=1).values
y = data['math score'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better performance during training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, "scaler.pkl")

# Neural Network Model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='linear')
])

# Compile the model
# sgd = SGD(learning_rate=0.1)  # Setze eine sinnvolle Lernrate
# sgd = SGD(learning_rate=0.1, momentum=0.9)
# adam = Adam(learning_rate=0.0005)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

# Train the model and store history
# lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
# history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, callbacks=[lr_schedule])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)

# Save the trained model
model.save("student_performance_model.keras")

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Define the number of epochs, training loss, and validation loss
epochs = range(1, len(history.history['loss']) + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create the plot
plt.figure(figsize=(10, 6))

# Option 1: Logarithmic y-axis scale (uncomment the next line if you want it)
# plt.yscale('log')

# Option 2: Limit y-axis for better visibility of low-loss values (optional)
plt.ylim(0, 500)  # Adjust the range to focus on low-loss values after the sharp drop

plt.plot(epochs, train_loss, 'bo-', label='Training MSE Loss')  # Training loss
plt.plot(epochs, val_loss, 'r^-', label='Validation MSE Loss')  # Validation loss

# Add title and labels
plt.title('Training and Validation Loss over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('MSE Loss', fontsize=14)

# Customize the legend
plt.legend(loc='upper right', fontsize=12)

# Add grid for better readability
plt.grid(True)

# Save the plot to a file
plt.savefig("training_loss_plot_zoomed.png")

# Show the plot
plt.show()

print("Der Plot wurde als loss_plot.png gespeichert.")

# Predict for new data using command line arguments
parser = argparse.ArgumentParser(description="Student Performance Predictor")
# Add arguments corresponding to each feature in exams.csv, except target columns
for column in X.columns:
    arg_name = column.replace(' ', '_').replace('/', '_').replace("'", '').lower()
    parser.add_argument(f"--{arg_name}", type=float, required=True, help=f"Value for {column}")
# Add arguments for scores
# parser.add_argument("--math_score", type=float, required=True, help="Value for math score")
# parser.add_argument("--reading_score", type=float, required=True, help="Value for reading score")
# parser.add_argument("--writing_score", type=float, required=True, help="Value for writing score")

args = parser.parse_args()

# Collect provided arguments
provided_args = [getattr(args, column.replace(' ', '_').replace('/', '_').replace("'", '').lower()) for column in X.columns]
input_data = pd.DataFrame([provided_args], columns=X.columns)  # Create a DataFrame with feature names
# print("One-Hot-Encoded Werte für den Testfall:")
# print(input_data)
input_data_scaled = scaler.transform(input_data)  # Now transform the input

# Calculate average score from provided scores
# average_score = np.mean([args.math_score, args.reading_score, args.writing_score])

# Make prediction
prediction = model.predict(input_data_scaled).flatten()
print(f"Predicted student score: {prediction[0]:.4f}")
# print(f"Provided average score: {average_score:.4f}")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create or connect to the Pinecone index
index_name = "student-performance"
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=X_train.shape[1] + 1, metric="euclidean", spec=ServerlessSpec(
        cloud='aws', region='us-east-1'
    ))

# Connect to the index
index_description = pc.describe_index(index_name)
host = index_description.host
index = pc.Index(index_name, host=host)

# Store the prediction in Pinecone
metadata = {column: provided_args[i] for i, column in enumerate(X.columns)}
metadata.update({
    "predicted_math_score": float(prediction[0])  # Store the predicted math score
    #"reading_score": args.reading_score,
    #"writing_score": args.writing_score
})

# Create a vector for upserting
vector = {
    "id": "student_" + str(np.random.randint(1, 10000)),  # Eine zufällige ID für den Schüler
    "values": [float(val) for val in input_data_scaled.flatten().tolist()] + [float(prediction[0])],
    "metadata": metadata
}

# Upsert the vector to Pinecone
index.upsert(vectors=[vector])

print("Die Vorhersage wurde erfolgreich in Pinecone gespeichert.")

# Ergebnisse in Log-Datei speichern
with open("log.txt", "a") as log_file:
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write("----------")

    log_file.write(f"\nDatum und Uhrzeit: {current_date}\n")
    log_file.write("One-Hot-Encoded Werte für den Testfall:\n")

    # Zeilenweise Anzeige und Schreiben in die Logdatei
    for column in input_data.columns:
        value = input_data[column].values[0]
        print(f"{column}: {value}")  # Ausgabe in der Konsole
        log_file.write(f"{column}: {value}\n")  # Schreiben in die Logdatei

    log_file.write("Vorhersageergebnisse:\n")
    log_file.write(f"Predicted student score: {prediction[0]:.4f}\n")
   # log_file.write(f"Provided average score: {average_score:.4f}\n")
    log_file.write("Die Vorhersage wurde erfolgreich in Pinecone gespeichert.\n")
    log_file.write("----------")

print("Die Vorhersage wurde erfolgreich in log.txt gespeichert.")
print("End.")
