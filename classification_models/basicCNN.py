import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, kernel_size=(3, 1), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 1)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid')  # Use sigmoid activation for multi-label classification
    ])

    return model

def train_and_evaluate_cnn(x_train, y_train, x_test, y_test):
    input_shape = x_train.shape[1:]  # Assuming x_train is a NumPy array
    num_classes = y_train.shape[1]    # Extract the number of classes from y_train

    # Build the CNN model
    model = build_cnn(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=2, batch_size=32, validation_split=0.25)

    # Predict on test set
    y_pred = model.predict(x_test)
    print(y_pred[:3])

    # Calculate evaluation metrics for each class
    auc_scores = []
    f1_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    for i in range(num_classes):
        auc_i = roc_auc_score(y_test[:, i], y_pred[:, i])
        f1_i = f1_score(y_test[:, i], (y_pred[:, i] > 0.5).astype(int))
        accuracy_i = accuracy_score(y_test[:, i], (y_pred[:, i] > 0.5).astype(int))
        precision_i = precision_score(y_test[:, i], (y_pred[:, i] > 0.5).astype(int))
        recall_i = recall_score(y_test[:, i], (y_pred[:, i] > 0.5).astype(int))

        auc_scores.append(auc_i)
        f1_scores.append(f1_i)
        accuracy_scores.append(accuracy_i)
        precision_scores.append(precision_i)
        recall_scores.append(recall_i)

    # Return all evaluation metrics for each class
    return {
        'AUC Scores': auc_scores,
        'F1 Scores': f1_scores,
        'Accuracy Scores': accuracy_scores,
        'Precision Scores': precision_scores,
        'Recall Scores': recall_scores
    }