import os
import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
import pickle

def carregar_dados(caminho):
    X, y_class, y_reg = [], [], []
    with open(caminho, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            qp = float(row[3])
            pf = float(row[4])
            rf = float(row[5])
            gr = float(row[6])
            cl = int(row[7])
            X.append([qp, pf, rf])
            y_class.append(cl)
            y_reg.append(gr)
    return np.array(X), np.array(y_class), np.array(y_reg)

treino_path = "datasets/data_4000v/env_vital_signals.txt"
teste_path = "datasets/data_800v/env_vital_signals.txt"

X_train, y_class_train, y_reg_train = carregar_dados(treino_path)
X_test, y_class_test, y_reg_test = carregar_dados(teste_path)

# Normalização 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurações de Classificador/Regressor
redes_neurais_configs = [
    {
        "name": "config1",
        "params": {
            "hidden_layer_sizes": (10,),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "momentum": 0.9,
            "max_iter": 2000,
            "random_state": 42
        }
    },
    {
        "name": "config2",
        "params": {
            "hidden_layer_sizes": (20, 10),
            "activation": "tanh",
            "solver": "sgd",
            "learning_rate_init": 0.01,
            "momentum": 0.9,
            "max_iter": 2000,
            "random_state": 42
        }
    },
    {
        "name": "config3",
        "params": {
            "hidden_layer_sizes": (30, 20, 10),
            "activation": "logistic",
            "solver": "lbfgs",
            "learning_rate_init": 0.001,
            "momentum": 0.9,
            "max_iter": 2000,
            "random_state": 42
        }
    }
]

print("=== CLASSIFICADOR (Redes Neurais) ===")
best_clf = None
best_score = -1
best_clf_name = ""
for config in redes_neurais_configs:
    clf = MLPClassifier(**config["params"])
    scores = cross_val_score(clf, X_train_scaled, y_class_train, cv=3, scoring='f1_weighted')
    mean_score = scores.mean()
    p = config["params"]
    print(f"{config['name']} - f1_weighted média (cross-val): {mean_score:.4f}")
    print(f"  Camadas: {p['hidden_layer_sizes']}, ativação: {p['activation']}, solver: {p['solver']}, lr: {p['learning_rate_init']}, momentum: {p['momentum']}")
    if mean_score > best_score:
        best_score = mean_score
        best_clf = clf
        best_clf_name = config["name"]

best_clf.fit(X_train_scaled, y_class_train)
y_pred = best_clf.predict(X_test_scaled)
print("Acurácia (teste cego):", accuracy_score(y_class_test, y_pred))
print("Matriz de confusão:\n", confusion_matrix(y_class_test, y_pred))
print(f"Melhor classificador: {best_clf_name} - Camadas: {best_clf.hidden_layer_sizes}")
print("\nRelatório de classificação (teste cego):")
print(classification_report(y_class_test, y_pred))

with open("NN_Classificador.pkl", "wb") as f:
    pickle.dump(best_clf, f)


print("\n=== REGRESSOR (Redes Neurais) ===")
best_reg = None
best_rmse = float("inf")
best_reg_name = ""
for config in redes_neurais_configs:
    reg = MLPRegressor(**config["params"])
    scores = cross_val_score(reg, X_train_scaled, y_reg_train, cv=3, scoring='neg_mean_squared_error')
    mean_rmse = np.sqrt(-scores.mean())
    p = config["params"]
    print(f"{config['name']} - RMSE médio (cross-val): {mean_rmse:.4f}")
    print(f"  Camadas: {p['hidden_layer_sizes']}, ativação: {p['activation']}, solver: {p['solver']}, lr: {p['learning_rate_init']}, momentum: {p['momentum']}")
    if mean_rmse < best_rmse:
        best_rmse = mean_rmse
        best_reg = reg
        best_reg_name = config["name"]

best_reg.fit(X_train_scaled, y_reg_train)
y_pred_reg = best_reg.predict(X_test_scaled)
mse = mean_squared_error(y_reg_test, y_pred_reg)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
print(f"Melhor regressor: {best_reg_name} - Camadas: {best_reg.hidden_layer_sizes}")

with open("NN_Regressor.pkl", "wb") as f:
    pickle.dump(best_reg, f)

with open("NN_Scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)