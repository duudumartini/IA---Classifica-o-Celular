import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

historico = None  # Definindo historico como variável global inicialmente
fig, ax, canvas = None, None, None  # Definindo fig, ax e canvas como variáveis globais inicialmente

features_num1 = [
    "Capacidade da Bateria (mAh)", "Velocidade do Clock (GHz)", "Pixels da Câmera Frontal", 
    "Memória Interna (GB)", "Espessura do Aparelho (cm)", "Peso do Aparelho (g)", "Largura em Pixels"
]

features_num2 = [
    "Memória RAM (MB)", "Altura da Tela (cm)", "Largura da Tela (cm)", "Tempo de Duração da Bateria (h)",
    "Núcleos", "Pixels da Câmera Principal", "Altura em Pixels"
]

features_bool = [
    "Bluetooth", "Dual SIM", "Possui 4G", "Possui 3G", "Touch Screen", "Wi-Fi"
]

def train_model(epochs, feature_test):
    global historico, fig, ax, canvas
    
    # Carregar os dados de treinamento e teste
    train_data = pd.read_csv('D:/DowloadsChrome/archive/train.csv', sep=',')  
    test_data = pd.read_csv('D:/DowloadsChrome/archive/test.csv', sep=',') #caso não tenha cabeçalho usar header=none

    # Separar features e labels dos dados de treinamento
    X_train = train_data.iloc[:, :-1]  # Todas as colunas, exceto a última
    y_train = train_data.iloc[:, -1]  # A última coluna é a label

    # Normalizar os dados de treinamento e teste
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / std

    # Construir o modelo
    modelo = Sequential([
        Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(512, activation='relu'),
        Dropout(0.1),
        Dense(4, activation='softmax')
    ])

    # Compilar o modelo
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    # Treinar o modelo
    historico = modelo.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1, validation_split=0.2, callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: update_progress(epoch, logs, epochs))])

    # Criar e plotar o gráfico
    fig, ax = plt.subplots()
    ax.set_title('Acurácia por épocas')
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Acurácia')

    ax.plot(historico.history['accuracy'], label='Treino')
    ax.plot(historico.history['val_accuracy'], label='Validação')
    ax.legend()

    # Integrar o gráfico na janela Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=5, sticky="nsew", rowspan=20)

    feature_test1 = (np.array(feature_test) - mean) / std
    prediction = modelo.predict(np.array([feature_test1]))
    predicted_class = np.argmax(prediction)
    descrever_faixa_preco(predicted_class)

    # Atualizar o label com o resultado do predicted_class
    update_predicted_class_label(predicted_class)

def update_progress(epoch, logs, total_epochs):
    global fig, ax, canvas, historico
    
    progress = (epoch + 1) / total_epochs
    progress_bar["value"] = progress * 100
    root.update_idletasks()

    # Verificar se os eixos foram inicializados
    if ax is not None:
        # Limpar os eixos
        ax.clear()

        # Plotar os novos dados de treinamento e validação
        ax.plot(historico.history['accuracy'][:epoch+1], label='Treino')
        ax.plot(historico.history['val_accuracy'][:epoch+1], label='Validação')
        ax.set_title('Acurácia por épocas')
        ax.set_xlabel('Épocas')
        ax.set_ylabel('Acurácia')
        ax.legend()

        # Atualizar o canvas
        canvas.draw()

def start_training():
    global ax  # Definindo ax como uma variável global antes de chamar a função train_model
    epochs = int(epochs_entry.get())  # Corrigindo para pegar o valor do campo de entrada
    feature_values = []

    # Capturar os valores das entradas das características numéricas
    for entry in entries:
        feature_values.append(float(entry.get()))

    # Capturar os valores das caixas de seleção das características booleanas
    for checkbox_var in checkboxes:
        feature_values.append(int(checkbox_var.get()))
 
    feature_test = [feature_values[0],feature_values[14],feature_values[1],feature_values[15],feature_values[2],feature_values[16],feature_values[3],feature_values[4],feature_values[5],feature_values[11],feature_values[12],feature_values[6],feature_values[13],feature_values[7],
                    feature_values[8],feature_values[9],feature_values[10],feature_values[17],feature_values[18],feature_values[19]]
    
    train_model(epochs, feature_test)

def descrever_faixa_preco(numero):
    if numero == 0:
        return "Muito acessível, preço bastante baixo."
    elif numero == 1:
        return "Acessível, preço razoável."
    elif numero == 2:
        return "Moderado, preço um pouco acima da média."
    elif numero == 3:
        return "Caro, preço elevado em comparação com outras opções."
    else:
        return "Faixa de preço não reconhecida."

def update_predicted_class_label(predicted_class):
    predicted_class_text.set(descrever_faixa_preco(predicted_class))

root = tk.Tk()
root.title("Treinamento do Modelo")

# Label e Entry para o número de épocas
epochs_label = tk.Label(root, text="Número de Épocas:")
epochs_label.grid(row=1, column=0, padx=10, sticky='ns', columnspan=5)
epochs_entry = tk.Entry(root)
epochs_entry.grid(row=2, column=0, padx=10, pady=5, columnspan=5)

#Crie a barra de progresso
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(row=3, column=0, padx=10, pady=5, columnspan=5)

# Criar botão de treinamento
train_button = tk.Button(root, text="Treinar Modelo", command=start_training)
train_button.grid(row=4, column=0, padx=10, pady=5, columnspan=5)

# Entradas para cada característica
entries = []

# Entradas para features_num1
for i, feature in enumerate(features_num1):
    label = tk.Label(root, text=feature)
    label.grid(row=i+5, column=1, pady=5, sticky=tk.W)
    entry = tk.Entry(root)
    entry.grid(row=i+5, column=0, padx=(10,0), pady=5)
    entries.append(entry)

# Entradas para features_num2
for i, feature in enumerate(features_num2):
    label = tk.Label(root, text=feature)
    label.grid(row=i+5, column=3, pady=5, sticky=tk.W)
    entry = tk.Entry(root)
    entry.grid(row=i+5, column=2, padx=(80,0), pady=5)
    entries.append(entry)

# Caixas de seleção para características booleanas
checkboxes = []

for i, feature in enumerate(features_bool):
    checkbox_var = tk.BooleanVar()
    checkbox = tk.Checkbutton(root, text=feature, variable=checkbox_var)
    checkbox.grid(row=i+5, column=4, padx=(80,0), pady=5, sticky=tk.W)
    checkboxes.append(checkbox_var)

# Crie um label para mostrar o retorno do predicted_class
predicted_class_label = tk.Label(root, text="")
predicted_class_label.grid(row=0, column=0, padx=10, pady=5, columnspan=5, sticky="ns")

# Variável para armazenar o texto da faixa de preço predita
predicted_class_text = tk.StringVar()
predicted_class_text.set("")  # Inicialmente vazio

# Label para exibir o texto da faixa de preço predita
predicted_class_result_label = tk.Label(root, textvariable=predicted_class_text)
predicted_class_result_label.grid(row=0, column=0, padx=10, pady=5, columnspan=5, sticky="ns")

# Iniciar o loop da janela Tkinter
root.mainloop()
