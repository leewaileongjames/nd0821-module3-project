# Deploying a ML model with FastAPI

## A) Environment Setup

#### Install Miniconda

1. Download and Install Miniconda
```bash!
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

2. Initialize Miniconda in bash
```bash!
~/miniconda3/bin/conda init bash
```

3. In case Miniconda does not appear, run the following command.
```bash!
exec bash
```

More Information [here](https://docs.conda.io/projects/miniconda/en/latest/)

#### Create Conda Environment

1. Run the following command to create a new conda environment. The installation process might take a few minutes.

```bash!
conda env create -f environment.yml
```

<br/>

## B) Usage

1. Activate conda environment

```bash!
conda activate module-3-project
```

2. Launch the FastAPI app by starting the server.
```bash!
uvicorn --host 0.0.0.0 starter.app.main:app
```

## File Description

- `start/src/` - a directory that contains the functions used in scripts
- `starter/train_model.py` - main script that performs data cleaning & preprocessing and trains the KNN model
- `starter/training_artifacts/` - a directory that contains the trained model, encoder and label binarizer
- `starter/calculate_slice_metrics.py` - main script that performs calculates the metrics of various data slices
- `starter/slice_output.txt` - File that contains the performance metrics of various data slices
- `starter/screenshots/` - a directory that contains screenshots
- `starter/app/live_testing_low.py` - script that sends a request to the model API and expects a result of "<=50K"
- `starter/app/live_testing_high.py` - script that sends a request to the model API and expects a result of ">50K"
