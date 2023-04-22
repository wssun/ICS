# ICS
Integrating Extractive and Abstractive Models for Code Comment Generation



## Requirements

The dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```



## Original Dataset

The CodeSearchNet original dataset can be downloaded from the github repo: [https://github.com/github/CodeSearchNet](https://github.com/github/CodeSearchNet), and the cleaned dataset (CodeXGLUE) can be downloaded from the [https://drive.google.com/open?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h](https://drive.google.com/open?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h)



## Quick Start

### Extractor

Run the `Extractor_LSA/train.py` to build the LSA model.

For example:

```bash
cd Extractor_LSA/
```

```bash
python train.py --data_dir {data_path} --saved_model_path {saved_model_path} --result_path {result_path}
```

{data_path} specifies the path to the training dataset.

{saved_model_path} specifies the path to the folder where the parameter files for the model are stored.

{saved_data_path} specifies the path to the folder where the output file is located.

And run the `Extractor_LSA/test.py` to test the LSA model:

```
python test.py --data_dir {data_path} --saved_model_path {saved_model_path} --result_path {result_path}
```

The output samples are as follows:

```
- output samples:
0	int matrix i height convolution width ...
1	event state listener synchronized this ...
2	t try finally write unlock ...
...
```

Then run the `Extractor_LSA/gen.py` to extract important statements based on the key tokens:

```
python gen.py --file_path {file_path} --ex_file_path {ex_file_path}
```

{file_path} specifies the path to the original dataset.

{ex_file_path} specifies the path to the key tokens file.

The output samples are as follows:

```
{
    "idx": 0, 
    "code_tokens": ["public", "ImageSource", "apply", ...], 
    "docstring_tokens": ["Pops", "the", "top", ...],
    ...,
    "extractive sum": ["event", "state", "listener", ...], 
    "my_pred_cleaned_seqs": [1, 0, 1, 0, 1]
}
```

### ICS + CodeT5

```bash
cd ICS_codeT5/
```

```bash
python run_gen {language}
```

The {language} can be selected in `java, python, go, php, ruby, javascript`

### Evaluation

After training the ICS + CodeT5 model, run the evaluation code to output Bleu, Meteor and Rouge-L:

(*Switch into python 2.7*)

```bash
cd Evaluation/
```

```bash
python evaluate.py
```

