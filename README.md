

# Testing the Generalizability of MBR-exec and Further Improvements to Code Generation Accuracy
[Jeremy Dapaah](mailto:jdapaah@princeton.edu), [Rebecca Zhu](mailto:rebeccazhu@princeton.edu), [Aidan Walsh](mailto:abwalsh@princeton.edu)

## Setup
1. Navigate to our shared google drive that contains generated data from Codex and CodeGen: https://drive.google.com/drive/u/0/folders/0AN99DFfGjOUYUk9PVA 
2. If working in Colab, take note of the location or if working locally, be sure to download the data
3. If working locally, install the `conda` environment by 
```bash
conda env create -f env.yml
```
--- 

## Recommendations
If collecting data from CodeGen, we highly recommend using GPU if the model of CodeGen is at least 2 billion parameters. It can take up to a day just to generate 500 examples of CodeGen on CPU if parameters >= 2B. Otherwise, analyzing the decoding schemes is very doable on CPU. 

## Running Locally 
1. Download all files except the python notebook files. 
2. If collecting data from CodeGen, run the following command: 
```bash
python3 collect.py collect --output-path <> --split <> --seed <> --n-prompts <> --mode <> --n-samples <> --temperature <> --slurm-ntasks <>
``` 
where <> indicates your user input. Ensure that your argument for slurm ntasks and n samples are the same. All are optional and have default values except output-path. 
3. Our code submits this to a slurm cluster. If you have do not have access to a GPU on the slurm cluster (or any cluster), then we do not recommend doing this. Instead, do this in Colab. 
4. To test the decoding schemes, run run.py with 
```bash
python3 run.py
```
5. Assistance for running it is given in the file and all available decoding schemes are given. Everything except "mbr_meteor" can be ran. Note that one file is used for running everything except linear interpolation. 

## Running in Colab (Recommended if you do not have GPU locally or on cluster)
1. Download the ipynb files
2. Ensure you have access to our Shared Google Drive. If you do, the paths within the ipynb files should be correct.
3. Baselines.ipynb lets you generate the code from CodeGen. We have comments indicating where this can be done. Be sure the paths to mbpp.jsonl are correct.
4. Baselines.ipynb also lets you test the decoding schemes (except linear interpolation) on both Codex and CodeGen. We have comments and code that contain more info. 
5. Linear Interpolation only lets you test linear interpolation, where our hyperparameters are given. We also have comments and code that contain more info. 





--- 
## Acknowledgements
This code and data is used and adapted from https://github.com/facebookresearch/mbr-exec

## License
MIT
