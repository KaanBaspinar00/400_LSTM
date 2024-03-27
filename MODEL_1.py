import matplotlib.pyplot as plt
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape
import READ_FILES
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import os
from sklearn.metrics import precision_recall_fscore_support
from keras.models import load_model
from sklearn.metrics import confusion_matrix
# from keras.layers import GRU

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
np.set_printoptions(threshold=sys.maxsize)

# GET DATA
Reflectance_Data = READ_FILES.result_ref
Thickness_Data = READ_FILES.result_thick
MaterialData = READ_FILES.result_material
number_of_layers = 12
Thickness_Data_padded = []

for i in Thickness_Data:
    padded = i + [0] * (number_of_layers - len(i))
    Thickness_Data_padded.append(np.array(padded))

Thickness_Data_padded = np.array(Thickness_Data_padded)

MaterialData_padded = []
for sublist in MaterialData:
    # Calculate the number of padding rows required
    padding_rows = number_of_layers - len(sublist)

    # Pad the sublist with zeros
    padded_sublist = np.pad(sublist, ((0, padding_rows), (0, 0)), mode='constant')

    # Append the padded sublist to the new list
    MaterialData_padded.append(padded_sublist)

# Now our data : MaterialData_padded ; Thickness_Data_padded ; Reflectance_Data

thicknesses = np.array(Thickness_Data_padded)
materials_data = np.array(MaterialData_padded)
reflectance = np.array(Reflectance_Data)

COMBINED = []
for i in range(len(materials_data)):

    multiplied = np.multiply(np.array(materials_data[i]).transpose(), np.array(thicknesses[i]))
    COMBINED.append(np.array(multiplied).transpose())

# Reshape materials_data to align with reflectance data
materials_data_reshaped = materials_data

# C = COMBINED

C = materials_data_reshaped
R = reflectance

X_train, X_test, y_train, y_test = train_test_split(R, C, test_size=0.20, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=123)

X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Add a dimension for time steps
X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# I will change the names later
y_train_reshaped = np.asarray(y_train)
y_val_reshaped = np.asarray(y_val)
y_test_reshaped = np.asarray(y_test)

# In order to train the model, uncomment the part below.
"""
units = 100
epochs = 5
batch_size = 100

learning_rate = 0.01

# Create the Adam optimizer with the desired learning rate
optimizer = Adam(learning_rate=learning_rate)

# Define the model
model = Sequential()

# Add an LSTM layer
model.add(LSTM(units, input_shape=(X_train.shape[1], 1), activation="tanh", recurrent_activation='sigmoid'))

# Add a dense layer to map LSTM output to the desired shape (n*m)
# n: number of layers   m: number of materials in the catalog
model.add(Dense(np.prod(np.asarray(y_train).shape[1:]), activation='tanh'))
model.add(Dense(np.prod(np.asarray(y_train).shape[1:]), activation='linear'))


# Reshape the output to match the shape of data
model.add(Reshape(np.asarray(y_train).shape[1:]))

# Compile the model
model.compile(loss="huber", optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train_reshaped, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test_reshaped, y_test_reshaped))

# Save the trained model
model.save("trained_model9.h5")

# Plot accuracy graph
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
loss = model.evaluate(X_test_reshaped, y_test_reshaped)
print("Test Loss:", loss)
"""
model = load_model('trained_model9.h5')


def turn_to_one_hot_encoding(predicted_value):
    one_hot = []
    for k in predicted_value:
        max_value = max(k)
        if max_value >= 0.5:
            result = [1 if x == max_value else 0 for x in k]
            one_hot.append(result)
        else:
            result = list(np.zeros(len(k)))
            one_hot.append(result)
    return one_hot


# Predict using the trained model
predicted_y = model.predict(X_val_reshaped)

true_y = y_val_reshaped
print(len(predicted_y))
print(len(true_y))


# Turn the predicted values to one-hot encoding
def pred_to_one_hot(pred):
    one = []
    for z in pred:
        j = turn_to_one_hot_encoding(z)
        one.append(j)
    return one


precision, recall, f1_score, _ = precision_recall_fscore_support(
    np.ravel(true_y), np.ravel(pred_to_one_hot(predicted_y)), average='binary')

print("\n\n\n\n\n")
print("2-12 film deposition")
print("Trained using 20000 data with train-test split: test_size = 0.2")
print(f"precision: {precision}, recall: {recall}, f1_score: {f1_score}")

# Calculate confusion matrix
cm = confusion_matrix(np.ravel(true_y), np.ravel(pred_to_one_hot(predicted_y)))

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap("Blues"))
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(cm))
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Test the data performance manually
"""
X = 0
print("Inside the for loop")
for i in range(len(predicted_y)):
    one_hotted_material = pred_to_one_hot(predicted_y)
    diff = np.subtract(np.asarray(one_hotted_material[i]),true_y[i])
    z = 0
    for sublist in diff:
        # Check if all elements in the sublist are zeros
        if all(element == 0 for element in sublist):
            pass
        else:
            z += 1
    X +=z
    #print(z,i,X)

print(f"There is error for {X} film")
print(len(y_val))
print(f"Accuracy for {len(y_val)} data: ",(len(y_val)*5 - X)/(len(y_val)*5))

print("Finished")
"""

# This part is GRU, it can be tested if ı want to test it.

"""
# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
"""
"""
# Define your model
model = Sequential()

# Add a GRU layer with appropriate input shape
model.add(GRU(units, input_shape=(X_train.shape[1], 1), activation="tanh", reset_after=False))

# Add a dense layer to map GRU output to the desired shape (5*21)
model.add(Dense(np.prod(np.asarray(y_train).shape[1:]), activation='softmax'))

# Reshape the output to match the shape of y_train
model.add(Reshape(np.asarray(y_train).shape[1:]))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train_reshaped, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test_reshaped, y_test_reshaped))

# Evaluate the model
loss = model.evaluate(X_test_reshaped, y_test_reshaped)
print("Test Loss:", loss)"""