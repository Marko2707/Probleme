"""
Author: Marko Stankovic

Aspects of the code were created by other authors and will be credited in the function

"""
#Imports of Dependencies necessary for execution
import numpy as np
import pandas as pd
import wfdb
import ast
import pandas as pd
import time as tm
import matplotlib.pyplot as plt


#Machine Learning Stuff
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer


#imports of modules in this project
from helper_functions import is_folder_empty
from classification_models.basicResNet import build_resnet, train_and_evaluate_resnet
from classification_models.basicCNN import build_cnn, train_and_evaluate_cnn
from classification_models.wavelet import WaveletModel


#fürs RNN1d
from sklearn.preprocessing import LabelEncoder
from classification_models.rnn1d import AdaptiveConcatPoolRNN, RNN1d

from peak_detection_algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus

#imports for ResNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

from classification_models.x_resnet import xresnet1d101



#add the folder where your ptb-xl data is
pathname = "C:/Users/marko/Desktop/bachelor_arbeit/code/"

#setting the samplerate to 100Hz (choosing)
sampling_rate=100

#Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(pathname+"scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def main(): 
    #Laden der Daten in Trainings und Test Sets
    print("Code initialized")
    #Defining the Folder where the NumpArr and PandSeries are saved
    numpy_path = "BA/NumpyArrays/"
    series_path = "BA/PandaSeries/"
    x_train_unprocessed = numpy_path + "x_train_unprocessed.npy"
    x_test_unprocessed = numpy_path + "x_test_unprocessed.npy"
    y_train_path = series_path + "y_train.pkl"
    y_test_path = series_path + "y_test.pkl"

    if (is_folder_empty(numpy_path) or is_folder_empty(series_path)):
        print("Data must be loaded from the PTB-XL")
        #Timetest
        data_load_time_start = tm.time()
        #Taking in the Sets
        x_train, y_train, x_test, y_test, Y = dataCreation(pathname)
        data_load_time_end = tm.time()
        print("Data Load time from PTB-XL Database: ", data_load_time_end - data_load_time_start)
        
        #Saving the Numpy Arrays for x_test and x_train
        np.save(x_train_unprocessed, x_train)
        np.save(x_test_unprocessed, x_test)

        #Saving PandaSeries
        series_path = "BA/PandaSeries/"
        y_train.to_pickle(y_train_path)
        y_test.to_pickle(y_test_path)

        #Test of Equal Datasets:
        x_train_unprocessed = np.load(x_train_unprocessed)
        x_test_unprocessed = np.load(x_test_unprocessed)
        print(f"Test if the x_data is the same: {x_train_unprocessed.shape == x_train.shape and x_test_unprocessed.shape == x_test.shape}")

        # Load Series from pickle
        y_train_pickle = pd.read_pickle(y_train_path)
        y_test_pickle = pd.read_pickle(y_test_path)
        print(f"Test if the y_data is the same: {y_train_pickle.shape == y_train.shape and y_test_pickle.shape == y_test.shape}")


    #---Loading the Data from the NumpyArrays and PandaSeries------------------------
    #Loading the NumpyArrays
    npy_load_time_start = tm.time()
    x_train = np.load(x_train_unprocessed)
    x_test = np.load(x_test_unprocessed)
    # Load Series from pickle

    y_train= pd.read_pickle(y_train_path)
    y_test = pd.read_pickle(y_test_path)
    npy_load_time_end = tm.time()
    print(f"Data load time from the saved Folders: {npy_load_time_end - npy_load_time_start}")


    #Zählt Menge an Multilabel Klassen
    count = 0
    for ecg_id, ecg_classes in y_train.items():
        # Check if the ecg_id has already been processed
        if len(ecg_classes) > 1:
            count += 1
            break
    print(f"count: {count}")



    #Get for each superclass one ECG Sample
    # Definiere die Klassen, die du extrahieren möchtest
    target_classes = ["NORM", "MI", "STTC", "CD", "HYP"]

    #Get these classes extracted
    extracted_classes = []
    #Get the according ecg_id for the class
    extracted_ecg_ids = []

    #Number of records with no superclass
    number_no_superclass = 0


    # Iterate over y_train until each target class is extracted at least once
    for ecg_id, ecg_classes in y_train.items():
        # Check if the ecg_id has already been processed
        #if ecg_id in extracted_ecg_ids:
        #    continue

        # Check if the list of classes is not empty
        if ecg_classes:
            # Extract the first class
            ecg_class = ecg_classes[0]

            # Check if the class is in the target classes, has not been extracted yet,
            # and there are no other classes present with len(ecg_classes) == 1
            if ecg_class in target_classes and ecg_class not in extracted_classes and len(ecg_classes) == 1:
                # Perform your desired actions with ecg_id and ecg_class
                print(f"ecg_id: {ecg_id}, Klasse: {ecg_class}")

                # Add the extracted class and the according ecg_id to the lists
                extracted_classes.append(ecg_class)
                extracted_ecg_ids.append(ecg_id)

                # Remove the extracted class from target_classes to avoid repeated extraction
                target_classes.remove(ecg_class)

            # Check if all target classes have been extracted
            if not target_classes:
                break
        else:
            #Counting the amount of records with no superclass
            number_no_superclass += 1


    #Get all information of y_train of one inext
    #print(Y.loc[y_train.index[0]])

    #!To plot the different Classes remove the block comment
    #Plotting all 5 ECG Samples of each Super-Class:
    """
    #Defining the 5 Superclasses I want to plot
    classes_to_plot = ["NORM", "MI", "STTC", "CD", "HYP"]

    # List of the ECG-Lead-Names
    lead_names = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


   
    # Iterating over all 5 Classes i want to plot
    for ecg_class in classes_to_plot:
        #Finding the index with ecg class == class i want to plot
        indices = [i for i, label in enumerate(extracted_classes[:5]) if label == ecg_class]

        #Iterating over the 5 indices
        for i in indices:
            # extracting the ecg data from the x_train with the ids from before
            ecg_data = x_train[extracted_ecg_ids[i]]
            # creating the x axis as seconds with a sampling reate of 100Hz
            time_axis = np.arange(0, 10, 1/100)

            # Ploting the ecg data for each lead
            for lead in range(ecg_data.shape[1]):
                plt.plot(time_axis, ecg_data[:, lead], label=f"{lead_names[lead]}")

            # Titel of the plot with each class
            plt.title(f"ECG Signal - Class: {ecg_class}")
            plt.xlabel("Time (s)") # Add the unit to the x-axis label
            plt.ylabel("Amplitude (mV)")  # Add the unit to the y-axis label
            plt.legend()
            plt.show()
    """
    #End of Plotting
    

    #Ausführung des ML auf den Training und Testdaten
    
    # Erstellen Sie zufällige Trainings- und Testdaten
    # Assuming y_train and y_test are DataFrame columns with lists of classes

    #MLB Labels und Mein Modell
    
    # Convert multi-label classes to binary format

    
    mlb = MultiLabelBinarizer()
    y_train_binary = mlb.fit_transform(y_train)
    y_test_binary = mlb.transform(y_test)

    print(f"Y_train: {y_train_binary[:10]}")
    # Add an extra dimension for channels
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    print("Check how multilabel classes are transformed")
    print(y_train[36:42])
    print(y_train_binary[36:42])

    # Call the function with the modified parameters
    evaluation_metrics = train_and_evaluate_cnn(x_train, y_train_binary, x_test, y_test_binary)
    print("Evaluation Metrics for each class:", evaluation_metrics)
    

    """
    print("Trying the ResNet")
    # Beispielaufruf
    hyperparameters = {
        "model": xresnet1d101,  # Modellfunktion
        "model_params": {"num_classes": 5},  # Modellparameter
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
    }
    results = x_ResNet(x_train, y_train, x_test, y_test, hyperparameters)

    print(results)
    """
    #Wavelet(x_train, y_train, x_test, y_test)

    #Erster Umgang mit PanTompkinsPlusPlus
    """
    # Extrahieren Sie das erste EKG-Signal (1. Leitung) "NORM"
    #first_ekg_signal = x_train[0, :, 0]
    # CD Klasse
    first_ekg_signal = x_train[31,:,0]

    #Setting the frequency to 100Hz for the PanTompkinsPlusPlus
    freq = 100

    # Initialisieren Sie eine Instanz des Pan_Tompkins_Plus_Plus-Algorithmus
    pan_tompkins = Pan_Tompkins_Plus_Plus()

    # Wenden Sie den Algorithmus auf das erste EKG-Signal an
    r_peaks_indices = pan_tompkins.rpeak_detection(first_ekg_signal, freq)

    # Zeitachse für das Plot
    time_axis = np.arange(0, 10, 1/ freq )

    r_peaks = r_peaks_indices.astype(int)

    # Ausgabe der detektierten R-Peaks-Indizes
    print("Detected R-peaks indices:", r_peaks_indices)

    # Plot des EKG-Signals
    plt.figure(figsize=(12, 6))
    plt.plot( first_ekg_signal, label='ECG Signal')
    plt.plot(r_peaks, first_ekg_signal[r_peaks], "x", color="red")
    plt.title('ECG Signal with Detected R-peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    """
    


    




def train_and_evaluate(model, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=32, learning_rate=0.001):
    # Konvertiere die Daten und Labels in PyTorch-Tensoren
    x_train_tensor = torch.Tensor(train_data)
    y_train_tensor = torch.Tensor(train_labels).long()  # Annahme: Klassifizierungsaufgabe mit Ganzzahlen als Labels

    x_test_tensor = torch.Tensor(test_data)
    y_test_tensor = torch.Tensor(test_labels).long()

    # Erstelle DataLoader für das Training und Testen
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialisiere das Modell, den Verlust und den Optimierer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Trainingsschleife
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Berechne den durchschnittlichen Verlust pro Epoche
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

    # Evaluierung auf Testdaten
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())

    # Berechne die Genauigkeit auf den Testdaten
    accuracy = accuracy_score(test_labels, all_predictions)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

    return model



def x_ResNet(x_train, y_train, x_test, y_test, hyperparameters):
    # 1. Vorbereitung der Daten

    # MultiLabelBinarizer verwenden, um die Labels in ein geeignetes Format zu bringen
    mlb = MultiLabelBinarizer()
    y_train_binary = torch.tensor(mlb.fit_transform(y_train.values), dtype=torch.float32)
    y_test_binary = torch.tensor(mlb.transform(y_test.values), dtype=torch.float32)

    x_train, y_train, x_test, y_test = (
        torch.tensor(x_train).float(),
        torch.tensor(y_train_binary).long(),
        torch.tensor(x_test).float(),
        torch.tensor(y_test_binary).long(),
    )

    print("Shape of x_train:")
    print(x_train.shape)
    print("Shape of x_test:")
    print(x_test.shape)


      # 2. Definiere das Modell
    model = hyperparameters["model"](**hyperparameters["model_params"])

    # 3. Definiere die Verlustfunktion und den Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])


    # 4. Schreibe Trainings- und Evaluierungsschleifen
    def train_model(model, dataloader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            #Problem Area
            outputs = sh
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(dataloader)

    def evaluate_model(model, dataloader):
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_predictions, all_labels

     # 5. Kombiniere alles
    batch_size = hyperparameters.get("batch_size", 32)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    epochs = hyperparameters.get("epochs", 10)
    for epoch in range(epochs):
        train_loss = train_model(model, train_dataloader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    # Evaluierung auf den Testdaten
    predictions, true_labels = evaluate_model(model, test_dataloader)

    # Berechne und drucke Metriken
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    return {
        "model": model,
        "predictions": predictions,
        "true_labels": true_labels,
        "accuracy": accuracy,
    }

#Problem ist die Anzahl an Samples Index Probleme plus jede Klasse muss einzeln getestet werden, Binär statt Multi Label
def Wavelet(x_train, y_train, x_test, y_test):
    # Definiere die Anzahl der Folds für die Kreuzvalidierung
    # Preprocessing target variables to be binary
    mlb = MultiLabelBinarizer()
    y_train_binary = mlb.fit_transform(y_train)
    y_test_binary = mlb.transform(y_test)

    # Definiere die Klassen
    ecg_classes = ["NORM", "MI", "STTC", "CD", "HYP"]

    # Definiere die Eingabeformate (shape) entsprechend deiner Daten
    input_shape = (1000, 12)

    # Definiere das Modell
    wavelet_model = WaveletModel(name='WaveletModel', n_classes=len(ecg_classes), freq=100, outputfolder='./', input_shape=input_shape, classifier='RF')

    # Stratified K-Fold Validierung
    skf = StratifiedKFold(n_splits=len(ecg_classes), shuffle=True, random_state=42)

    # Listen zum Speichern der Ergebnisse
    accuracy_scores = []
    precision_scores = []
    f1_scores = []
    roc_auc_scores = []

    for class_index, ecg_class in enumerate(ecg_classes):
        print(f"Training and evaluating model for class: {ecg_class}")

        # Trainings- und Testdaten für diese Klasse vorbereiten
        y_train_class = y_train_binary[:, class_index]
        y_test_class = y_test_binary[:, class_index]

        for fold, (train_index, test_index) in enumerate(skf.split(x_train, y_train_class)):  # Anpassung hier
            X_train, X_test = x_train[train_index], x_train[test_index]  # Anpassung hier
            y_train_class_fold, y_test_class_fold = y_train_class[train_index], y_test_class[test_index]

            # Überprüfen der Größe der Daten
            print(f"Fold {fold + 1}: Train Size: {len(y_train_class_fold)}, Test Size: {len(y_test_class_fold)}")

            # Modell trainieren
            wavelet_model.fit(X_train, y_train_class_fold, X_test, y_test_class_fold)

            # Vorhersagen auf den Testdaten
            y_pred_class_fold = wavelet_model.predict(X_test)

            # Konvertiere die Vorhersagen in binäre Klassen (1 oder 0)
            y_pred_binary_class_fold = (y_pred_class_fold > 0.5).astype(int)

            # Berechne Metriken
            accuracy_scores.append(accuracy_score(y_test_class_fold, y_pred_binary_class_fold))
            precision_scores.append(precision_score(y_test_class_fold, y_pred_binary_class_fold))
            f1_scores.append(f1_score(y_test_class_fold, y_pred_binary_class_fold))
            roc_auc_scores.append(roc_auc_score(y_test_class_fold, y_pred_class_fold))

    # Durchschnittliche Metriken berechnen
    average_accuracy = np.mean(accuracy_scores)
    average_precision = np.mean(precision_scores)
    average_f1 = np.mean(f1_scores)
    average_roc_auc = np.mean(roc_auc_scores)

    # Ausgabe der Ergebnisse
    print(f'Average Accuracy: {average_accuracy}')
    print(f'Average Precision: {average_precision}')
    print(f'Average F1 Score: {average_f1}')
    print(f'Average ROC AUC: {average_roc_auc}')

def dataCreation(pathname):

    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    path =  "C:/Users/marko/Desktop/bachelor_arbeit/code/"
    sampling_rate=100
    # load and convert annotation data
    Y = pd.read_csv(path+"ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+"scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # Split data into train and test
    test_fold = 10

    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    return (X_train, y_train, X_test, y_test, Y)




if __name__ == "__main__":
    main()
