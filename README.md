# AA-Project-2
Repository for the second project of AA.

# Report
[Report](https://github.com/filipecolladavid/MultipleMyelomaSurvival/blob/main/MultipleMyelomaSurvivalReport.pdf)

## Dataset
To store the dataset you must create a directory in the root of this repository called ```data```<br>
The dataset will not be present in this repository, you must download it into the ```./data``` directory.<br>
The dataset is available at Kaggle.com in the [competition page](https://www.kaggle.com/competitions/machine-learning-nova-multiple-myeloma-survival/data)

### Kaggle token
A Kaggle token will allow you to easily download kaggle datasets.<br>

In the kaggle website, go to Profile -> Settings -> API -> Create New Token<br>
This will download a file called ```kaggle.json```.<br>
To see the expected file location run:
```bash
kaggle
```
Output
```txt
...
OSError: Could not find kaggle.json. Make sure it's located in [LOCATION]/.kaggle/
```
Place the file there, you might need to create a directory named ```.kaggle```.
You may need to change it's permissions.
```
chmod 600 [LOCATION]/.kaggle/kaggle.json
```
<b>Note:</b> This is your own private authentication key, don't share it with anyone.

Download the dataset
```
cd data
kaggle competitions download -c machine-learning-nova-multiple-myeloma-survival
unzip [dataset.zip]
```

### Alternative methods
Just download and place the data into the ```./data``` directory, using the Download All option in the Data section of the kaggle competion.<br>
You might need to create one.

## Usage

First create a python environment<br>
Unix
```bash
# Unix
python3 -m venv venv
# Windows 
python -m venv venv
```

Activate the python environment
```bash
# Unix
source venv/bin/activate
# Windows
venv\Scripts\activate 
```

Install the requirements
```bash
pip install -r requirements.txt
```

Save new requirements to the requirements.txt
```bash
# Unix
pip freeze > requirements.txt

# Windows
python -m pip freeze > requirements.txt
```

## Project Structure
```
.
├── data                    # Holds the data, such as the initial dataset, the splits, the submissions for kaggle
├── models                  # Holds the jolib format saved models
├── notebooks               # Holds the notebooks experimentation/analysis
├── README.md               
├── requirements.txt        
├── results                 # Holds some plots, and other submissions
└── scripts                 # Holds some useful scripts
```

 
