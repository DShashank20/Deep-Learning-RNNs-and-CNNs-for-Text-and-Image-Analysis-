# Deep-Learning-RNNs-and-CNNs-for-Text-and-Image-Analysis-


Name: Shashank Reddy Dasari      
ID: 700781569       

---

---

## 🛠 Technologies Used

* **Language:** Python 3
* **Deep Learning Frameworks:** TensorFlow 2.x / Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Manipulation & Computation:** NumPy
* **Data Visualization:** Matplotlib
* **Model Evaluation:** Scikit-Learn (`confusion_matrix`, `classification_report`)
* **Environment:** Google Colab (with GPU Acceleration)

---

## 📝 Project Structure & Questions Addressed

### Q1: Implementing an RNN for Text Generation

*Objective: Build a character-level Language Model using stacked LSTM layers to generate text in the style of Shakespeare.*

* **Task 1: Load and Preprocess Dataset**
* Downloaded and unzipped `shakespeare.txt` dataset (~1.11 million characters).
* Cleaned text by lowercasing and extracted $39$ unique vocabulary characters.


* **Task 2: Convert Text into Sequences**
* Mapped characters to unique integer IDs.
* Formatted input/target sequences of length 100 shifted by 1 character, batched into batches of 64 using `tf.data.Dataset`.


* **Task 3: Define Model Architecture**
* Constructed a model consisting of an **Embedding layer (256-dim)**, **2 stacked LSTM layers (1024 units each)**, and a **Dense output layer (39-dim output)**.


* **Task 4: Train Model & Generate Text**
* Trained the model for **30 epochs** using `adam` optimizer and `SparseCategoricalCrossentropy` loss.
* Constructed an inference model (`stateful=True`, `batch_size=1`) restored from the latest checkpoint (`ckpt_30.weights.h5`).
* Generated text using character sampling across varying temperatures (**0.2**, **0.8**, and **1.5**).



---

### Q2: Sentiment Classification Using RNN

*Objective: Train a binary sentiment classifier on the IMDB Movie Reviews dataset.*

* **Task 1: Load Dataset**
* Loaded `tensorflow.keras.datasets.imdb` capped at the top 10,000 most frequent words.


* **Task 2: Tokenization & Sequence Padding**
* Padded and truncated sequences to a fixed sequence length of 256 using `pad_sequences`.


* **Task 3: Train LSTM Classifier**
* Built a Sequential architecture: `Embedding(10000, 128)` $\rightarrow$ `LSTM(128)` $\rightarrow$ `Dense(1, activation='sigmoid')`.
* Trained for 10 epochs using `binary_crossentropy` and achieved **~84.2% test accuracy**.


* **Task 4: Evaluation Metrics**
* Evaluated performance on 25,000 test reviews.
* Generated confusion matrix and classification report (`Precision`: ~0.84, `Recall`: ~0.83, `F1-Score`: ~0.83).


* **Task 5: Precision-Recall Tradeoff Interpretation**
* Outlined why balancing Precision and Recall is crucial depending on downstream requirements (e.g., minimizing false positives vs. false negatives in sentiment evaluation).



---

### Q3: Convolution Operations with Different Parameters

*Objective: Perform manual 2D convolutions using `tf.nn.conv2d` across different strides and padding strategies.*

* **Inputs:**
* $5 \times 5$ Input matrix (values from 1 to 25).
* $3 \times 3$ Laplacian/Edge Filter Kernel:



* **Evaluated Configurations:**
1. `Stride = 1`, `Padding = 'VALID'` $\rightarrow$ Output Shape: $3 \times 3$
2. `Stride = 1`, `Padding = 'SAME'` $\rightarrow$ Output Shape: $5 \times 5$
3. `Stride = 2`, `Padding = 'VALID'` $\rightarrow$ Output Shape: $2 \times 2$
4. `Stride = 2`, `Padding = 'SAME'` $\rightarrow$ Output Shape: $3 \times 3$



---

### Q4: CNN Feature Extraction with Filters and Pooling

*Objective: Apply edge-detection spatial filters to real-world images using Computer Vision techniques.*

* **Task 1: Implement Edge Detection Using Convolution**
* Loaded an image and converted it to grayscale via OpenCV (`cv2.imread`).
* Applied **Sobel-X** (vertical edge detection) and **Sobel-Y** (horizontal edge detection) filters using `cv2.filter2D`.
* Visualized original vs. filtered gradient outputs using `matplotlib`.



---

## ⚙️ Installation & Requirements

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn

```
