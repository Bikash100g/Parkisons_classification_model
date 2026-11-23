# Parkinson's Classification Model

A notebook-based machine-learning project for analyzing movement-sensor data and building models related to Parkinson's disease. The project processes time-series observations, cleans and prepares the data, and experiments with CNN, LSTM, and hybrid CNN-LSTM neural networks.

## Dataset Information
Source: PhysioNet - Parkinson’s Disease Smartwatch (PADS) Dataset (https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)
Observations: 5,159 movement task recordings from 469 participants
Sensors: Accelerometer (X, Y, Z), Gyroscope (X, Y, Z) sampled at 100Hz
Classes: Parkinson’s Patients (PD), Differential Diagnoses (DD), Healthy Controls (HC)

## Project workflow

1. Parse raw observation metadata and movement time-series files.
2. Combine the metadata and sensor readings into a structured dataset.
3. Explore the data and handle missing values and outliers.
4. Scale numeric features and encode the target labels.
5. Segment the time-series data into sequences of 10 time steps.
6. Train and evaluate deep-learning models using training curves, classification reports, confusion matrices, or prediction comparisons.

## Data

The processing notebooks expect a raw dataset containing JSON observation files and movement `.txt` files. The movement files are interpreted with these columns:

- `Time`
- `Accel_X`, `Accel_Y`, `Accel_Z`
- `Gyro_X`, `Gyro_Y`, `Gyro_Z`

Metadata such as participant ID, session name, device location, and condition is extracted from the JSON files. The processed data is combined into a CSV file and later cleaned into `Clean_Dataset.csv`.

The classification workflow keeps the `Healthy` and `Parkinson's` classes and maps them to binary labels:

- `Healthy` -> `0`
- `Parkinson's` -> `1`

The cleaned dataset is not included in this repository. Update the Google Drive paths in the notebooks or place the required CSV files in an accessible local path before running them.

## Notebooks

| Notebook | Description |
| --- | --- |
| `Faster_Data_processing.ipynb` | Optimized processing of JSON and time-series files using parallel file processing, producing a structured CSV dataset. |
| `Data_cleaning.ipynb` | Exploratory data analysis, class filtering, visualization, outlier inspection, feature scaling, label conversion, and saving the cleaned dataset. |
| `CNN.ipynb` | Binary classification with a 1D CNN using movement and metadata sequences. Includes training curves, classification metrics, and a confusion matrix. |
| `LSTM.ipynb` | LSTM-based time-series regression for predicting future accelerometer and gyroscope readings. Includes loss visualization and prediction export for analysis. |
| `CNN-LSTM.ipynb` | Hybrid CNN-LSTM binary classifier that extracts local patterns with convolutional layers and temporal dependencies with LSTM layers. |

## Models

### CNN classifier

The CNN notebook uses stacked `Conv1D`, batch-normalization, max-pooling, dropout, and dense layers. It uses a sigmoid output for binary classification and trains for up to 50 epochs with a batch size of 32.

### LSTM model

The LSTM notebook uses a bidirectional LSTM followed by additional LSTM and dense layers. Unlike the classification notebooks, its target is the next movement vector: the three accelerometer and three gyroscope values. It trains for up to 30 epochs with a batch size of 16 and evaluates regression loss/MAE.

### CNN-LSTM classifier

The hybrid model combines a `Conv1D` feature-extraction stage with stacked LSTM layers, batch normalization, dropout, and dense layers. It predicts the binary `Healthy` or `Parkinson's` condition from sequences of movement data.

## Running the notebooks

These notebooks were developed for Google Colab and use TensorFlow/Keras, pandas, NumPy, scikit-learn, SciPy, Matplotlib, and Seaborn.

1. Open a notebook in Google Colab or a local Jupyter environment.
2. Install or enable the required Python packages.
3. Provide the raw dataset or cleaned CSV files.
4. Update dataset and Google Drive paths where necessary.
5. Run the notebooks in this order for the full pipeline:

   `Faster_Data_processing.ipynb` -> `Data_cleaning.ipynb` -> `CNN.ipynb` / `LSTM.ipynb` / `CNN-LSTM.ipynb`

## Notes

- The notebooks use random seeds in the modeling workflows to improve reproducibility.
- Model performance depends on the dataset split, preprocessing choices, and available hardware.
- No trained model files, raw data, or fixed benchmark results are currently stored in this repository.

## License

No license has been specified for this repository.
