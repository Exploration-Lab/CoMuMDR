# Dialogue Discourse Parsing

Code for paper: Zhouxing Shi and Minlie Huang. A Deep Sequential Model for Discourse Parsing on Multi-Party Dialogues. In AAAI, 2019.

## Requirements
- Python 2.7
- Tensorflow 1.3

## Data Format

We use `JSON` format for data. The data file should contain an array consisting of examples represented as objects. The format for each example object looks like:
```json
{
    // a list of EDUs in the dialogue
    "edus": [ 
        {
            "text": "text of edu 1",
            "speaker": "speaker of edu 1"
        },    
        // ...
    ],
    // a list of relations
    "relations": [
        {
            "x": 0,
            "y": 1,
            "type": "type 1"
        },
        // ...
    ]
}
```

## STAC Corpus

We used the linguistic-only [STAC corpus](https://www.irit.fr/STAC/corpus.html). The latest available verison on the website is [stac-linguistic-2018-05-04.zip](https://www.irit.fr/STAC/stac-linguistic-2018-05-04.zip). It appears that test data is missing in this version. We share the [test data we used](https://drive.google.com/file/d/1rdUUyVxRZEgg8fKf2ILI2TDw8v6kAdYF/view?usp=sharing) from the 2018-03-21 version.


To process a raw dataset into the required `JSON` format, run:

```bash
python data_pre.py <input_dir> <output_json_file>
```

There are 1086 dialogues in the training data and 111 dialogues in the test data. 

## Word Vectors
For pre-trained word verctors, we used [GloVe](https://nlp.stanford.edu/projects/glove/) (100d).

## How to Run
```bash
python main.py {--[option1]=[value1] --[option2]=[value2] ... }
```

Available options can be found at the top of `main.py`.

For example, to train the model with default settings:

```bash
python main.py --train
```

Change the data directories in line 118 and 119 of `main.py` to point to the training and test data files, respectively.

To train and test the model on CoMuMDR data, you can use the following code snippet to load the data and map relations:
```python
data_train = load_data('../comumdr_data/train.json', map_relations)
data_test = load_data('../comumdr_data/test.json', map_relations)
```
