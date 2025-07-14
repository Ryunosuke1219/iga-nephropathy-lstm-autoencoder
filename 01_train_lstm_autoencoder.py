
import random
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Masking, Embedding, Concatenate, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
import optuna

# Set random seed for reproducibility
SEED_VALUE = 42
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# --- Data Loading ---
SEQ_FILE = 'data/seq_tensor_873.npz'
BASE_FILE = 'data/baseline.csv' # Used in the next script

data = np.load(SEQ_FILE)
X = data['X'].astype(np.float32)
patient_ids = data['patient_ids']

print("X shape:", X.shape)

# --- Scaler for Cross-Validation ---
class CVScaler:
    """Standardize quantitative variables (indices 1, 2) for each fold."""
    def __init__(self):
        self.scalers_ = []

    def fit_transform(self, X_train):
        X_scaled = np.copy(X_train)
        for i in range(X_train.shape[0]):
            scaler = StandardScaler()
            X_scaled[i, :, 1:] = scaler.fit_transform(X_train[i, :, 1:])
            self.scalers_.append(scaler)
        return X_scaled

    def transform(self, X_test):
        X_scaled = np.copy(X_test)
        for i in range(X_test.shape[0]):
            X_scaled[i, :, 1:] = self.scalers_[i].transform(X_test[i, :, 1:])
        return X_scaled

# --- LSTM Autoencoder Model ---
def create_lstm_autoencoder(input_shape, embedding_dim, lstm_units, latent_dim, dropout_rate):
    # Encoder
    input_layer = Input(shape=input_shape)
    masking_layer = Masking(mask_value=0.)(input_layer)

    # Separate ordinal and continuous features
    ordinal_input = masking_layer[:, :, 0]
    continuous_input = masking_layer[:, :, 1:]

    embedding_layer = Embedding(input_dim=6, output_dim=embedding_dim)(ordinal_input)
    concatenated = Concatenate()([embedding_layer, continuous_input])

    encoder_lstm = LSTM(lstm_units, dropout=dropout_rate, return_sequences=False)(concatenated)
    latent_vector = Dense(latent_dim, activation='linear', name='latent_vector')(encoder_lstm)

    # Decoder
    repeat_vector = RepeatVector(input_shape[0])(latent_vector)
    decoder_lstm = LSTM(lstm_units, return_sequences=True)(repeat_vector)
    reconstructed = TimeDistributed(Dense(input_shape[1]))(decoder_lstm)

    autoencoder = Model(inputs=input_layer, outputs=reconstructed)
    return autoencoder

# --- Optuna Objective Function ---
def objective(trial, X, n_splits=5):
    # Hyperparameters to tune
    embedding_dim = trial.suggest_int('embedding_dim', 2, 4)
    lstm_units = trial.suggest_categorical('lstm_units', [16, 32, 64])
    latent_dim = trial.suggest_int('latent_dim', 3, 12)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.05, 0.25)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED_VALUE)
    val_losses = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]

        # Scale data
        scaler = CVScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = create_lstm_autoencoder((X.shape[1], X.shape[2]), embedding_dim, lstm_units, latent_dim, dropout_rate)
        model.compile(optimizer='adam', loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        history = model.fit(X_train_scaled, X_train_scaled,
                          epochs=200,
                          batch_size=32,
                          validation_data=(X_val_scaled, X_val_scaled),
                          callbacks=[early_stopping],
                          verbose=0)

        val_losses.append(min(history.history['val_loss']))

    return np.mean(val_losses)

# --- Main Execution ---
if __name__ == '__main__':
    # Run Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X), n_trials=100)

    print("Best trial:", study.best_trial.params)

    # Train final model with best hyperparameters
    best_params = study.best_trial.params
    scaler = CVScaler()
    X_scaled = scaler.fit_transform(X)

    final_model = create_lstm_autoencoder((X.shape[1], X.shape[2]), **best_params)
    final_model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    final_model.fit(X_scaled, X_scaled,
                    epochs=200,
                    batch_size=32,
                    validation_split=0.2, # Using a validation split on the full dataset
                    callbacks=[early_stopping],
                    verbose=1)

    # Save the final model
    if not os.path.exists('output/models'):
        os.makedirs('output/models')
    final_model.save('output/models/lstm_autoencoder.h5')

    # Extract latent representations
    encoder = Model(inputs=final_model.input, outputs=final_model.get_layer('latent_vector').output)
    latent_representations = encoder.predict(X_scaled)

    # Save latent representations
    np.savez('output/latent_representations.npz', latent_reps=latent_representations, patient_ids=patient_ids)

    print("LSTM Autoencoder training and latent representation extraction complete.")
