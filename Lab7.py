import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------
# Streamlit basic config
# -------------------------------------------------------------------
st.set_page_config(page_title="CNN & LSTM Autoencoders",
                   layout="wide")

st.title("Feature Extraction & Dimensionality Reduction using Autoencoders")
st.write("CNN Autoencoder on CIFAR-10 images and LSTM Autoencoder on synthetic time-series.")


# -------------------------------------------------------------------
# Utility: CIFAR-10 data
# -------------------------------------------------------------------
@st.cache_resource
def load_cifar10():
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, x_test


# -------------------------------------------------------------------
# CNN Autoencoder
# -------------------------------------------------------------------
def build_cnn_autoencoder(input_shape=(32, 32, 3)):
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(encoder_input, decoded)
    encoder = models.Model(encoder_input, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


@st.cache_resource
def train_cnn():
    x_train, x_test = load_cifar10()

    cnn_autoencoder, cnn_encoder = build_cnn_autoencoder((32, 32, 3))

    history = cnn_autoencoder.fit(
        x_train, x_train,
        epochs=10,             # you can lower/raise for speed/quality
        batch_size=128,
        validation_data=(x_test, x_test),
        verbose=0
    )

    # Reconstructions
    decoded_imgs = cnn_autoencoder.predict(x_test[:10], verbose=0)

    # Full MSE
    decoded_full = cnn_autoencoder.predict(x_test, verbose=0)
    mse = mean_squared_error(x_test.flatten(), decoded_full.flatten())

    # Latent space + t-SNE
    latent_representations = cnn_encoder.predict(x_test[:1000], verbose=0)
    latent_flat = latent_representations.reshape(latent_representations.shape[0], -1)

    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_flat)

    return x_test, decoded_imgs, mse, latent_2d


# -------------------------------------------------------------------
# Synthetic Time-Series data & LSTM Autoencoder
# -------------------------------------------------------------------
def generate_sequences(num_samples=1000, seq_length=100):
    sequences = []
    for _ in range(num_samples):
        freq = np.random.uniform(0.1, 1.0)
        t = np.linspace(0, 4 * np.pi, seq_length)
        seq = np.sin(freq * t) + 0.1 * np.random.randn(seq_length)
        sequences.append(seq)
    sequences = np.array(sequences)
    sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
    return sequences


def build_lstm_autoencoder(input_shape):
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(encoder_input)
    x = layers.LSTM(32)(x)
    encoded = layers.RepeatVector(input_shape[0])(x)

    # Decoder
    x = layers.LSTM(32, return_sequences=True)(encoded)
    x = layers.LSTM(64, return_sequences=True)(x)
    decoded = layers.TimeDistributed(layers.Dense(1))(x)

    autoencoder = models.Model(encoder_input, decoded)
    # latent representation is the output of the last LSTM in decoder
    encoder = models.Model(encoder_input, x)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


@st.cache_resource
def train_lstm():
    sequences = generate_sequences()
    x_train_seq, x_test_seq = train_test_split(sequences, test_size=0.2, random_state=42)

    input_shape_seq = (x_train_seq.shape[1], x_train_seq.shape[2])
    lstm_autoencoder, lstm_encoder = build_lstm_autoencoder(input_shape_seq)

    history_lstm = lstm_autoencoder.fit(
        x_train_seq, x_train_seq,
        epochs=20,
        batch_size=64,
        validation_data=(x_test_seq, x_test_seq),
        verbose=0
    )

    # Reconstructions for a few sequences
    decoded_seqs = lstm_autoencoder.predict(x_test_seq[:5], verbose=0)

    # MSE on all test sequences
    decoded_all = lstm_autoencoder.predict(x_test_seq, verbose=0)
    mse_lstm = mean_squared_error(x_test_seq.flatten(), decoded_all.flatten())

    # Latent space for classification
    latent_seq = lstm_encoder.predict(x_test_seq, verbose=0)
    latent_seq_flat = latent_seq.reshape(latent_seq.shape[0], -1)

    # Dummy binary labels (since synthetic data)
    labels = np.random.randint(0, 2, size=latent_seq_flat.shape[0])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(latent_seq_flat, labels)
    acc = clf.score(latent_seq_flat, labels)

    return x_train_seq, x_test_seq, decoded_seqs, mse_lstm, acc


# -------------------------------------------------------------------
# UI Layout
# -------------------------------------------------------------------
mode = st.sidebar.radio(
    "Choose view",
    ["CNN Autoencoder (Images)", "LSTM Autoencoder (Time-Series)", "Comparison"]
)

# ================= CNN SECTION =================
if mode == "CNN Autoencoder (Images)":
    st.header("Part 1: CNN Autoencoder on CIFAR-10")

    with st.spinner("Training CNN Autoencoder (first run may take some time)..."):
        x_test, decoded_imgs, mse_cnn, latent_2d = train_cnn()

    st.subheader("Reconstructed Images")
    n = 10
    fig, ax = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        ax[0, i].imshow(x_test[i])
        ax[0, i].set_title("Original")
        ax[0, i].axis("off")

        ax[1, i].imshow(decoded_imgs[i])
        ax[1, i].set_title("Reconstructed")
        ax[1, i].axis("off")
    st.pyplot(fig)

    st.subheader("Reconstruction Error (MSE)")
    st.write(f"**CNN Autoencoder MSE:** `{mse_cnn:.6f}`")

    st.subheader("Latent Space Visualization (t-SNE)")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_title("Latent Space (CNN Encoder Output)")
    st.pyplot(fig2)

    st.markdown("""
**Observations:**
- CNN autoencoder reconstructs main object shapes and colors, but with some blur.
- Latent space compresses the original 32×32×3 image into a smaller feature map.
""")

# ================= LSTM SECTION =================
elif mode == "LSTM Autoencoder (Time-Series)":
    st.header("Part 2: LSTM Autoencoder on Synthetic Time-Series")

    with st.spinner("Training LSTM Autoencoder (first run may take some time)..."):
        x_train_seq, x_test_seq, decoded_seqs, mse_lstm, acc_lstm = train_lstm()

    st.subheader("Original vs Reconstructed Sequences")

    fig3, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    for i in range(5):
        axes[i].plot(x_test_seq[i].flatten(), label="Original")
        axes[i].plot(decoded_seqs[i].flatten(), label="Reconstructed")
        axes[i].legend(loc="upper right")
    st.pyplot(fig3)

    st.subheader("Reconstruction Error (MSE)")
    st.write(f"**LSTM Autoencoder MSE:** `{mse_lstm:.6f}`")

    st.subheader("Latent Space for Classification")
    st.write(f"Dummy logistic regression accuracy on latent features: **{acc_lstm:.3f}**")
    st.caption("Labels here are randomly generated, just to demonstrate using the latent space for a downstream task.")

    st.markdown("""
**Observations:**
- LSTM autoencoder follows the sinusoidal trend very closely.
- Reconstruction is smooth and denoised compared to the noisy input.
- Latent vector captures temporal pattern of the whole sequence.
""")

# ================= COMPARISON SECTION =================
else:
    st.header("Part 3: Comparison Between CNN and LSTM Autoencoders")

    with st.spinner("Loading results for comparison..."):
        _, _, mse_cnn, _ = train_cnn()
        _, _, _, mse_lstm, acc_lstm = train_lstm()

    st.subheader("Reconstruction Error")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CNN Autoencoder MSE", f"{mse_cnn:.6f}")
    with col2:
        st.metric("LSTM Autoencoder MSE", f"{mse_lstm:.6f}")

    st.subheader("Discussion")
    st.markdown(f"""
- **Data type:**
  - CNN autoencoder is designed for **spatial data (images)**.
  - LSTM autoencoder is designed for **sequential / time-series data**.

- **Reconstruction quality:**
  - On CIFAR-10, CNN MSE ≈ `{mse_cnn:.4f}`: captures objects but with blur.
  - On synthetic sequences, LSTM MSE ≈ `{mse_lstm:.4f}`: closely tracks the sine wave.

- **Dimensionality reduction:**
  - CNN compresses a 32×32×3 image into a much smaller **feature map** in the encoder.
  - LSTM compresses a length-100 sequence into a **latent vector / sequence representation**.

- **Typical applications:**
  - CNN autoencoders → image compression, denoising, pre-training for image classification.
  - LSTM autoencoders → anomaly detection in time-series, sequence representation, pre-training for sequence tasks.
""")
