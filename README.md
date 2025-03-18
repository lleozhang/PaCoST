# PaCoST: Paired Confidence Significance Testing for Benchmark Contamination Detection in Large Language Models
This is the repo for PaCoST (EMNLP 2024 findings)

## Code Usage
We release a simple demo script in this repo. 

### Environment Setup
We use some popular packages in our code, including ```pytorch, transformers, numpy, scipy```

### Code Organization
There are two main files in this repo. The ```test_model_pipeline.py``` serves as the main role for inference, paraphrase and confidence estimation. The ```test.py``` is only used for significance testing.

### How to Use Our Code 
We implemented loading mmlu as an example in ```test_model_pipeline.py```. To test a certain model, run

```python3 test_model_pipeline.py --model $NAME_OF_THE_MODEL --yes_id $ID_of_yes_Token_OF_THE_MODEL```

Then run the test script:

```python3 test.py --model $NAME_OF_THE_MODEL```

You can also change the random seed in the command line arguments. 

### NOTES
We would like to note an important point as follows:

This is only a demo script of how PaCoST works. There are many parts that can (or should) be modified to test other models, including the model used to paraphrase, the prompt format, the special tokens, the "yes" token id (specified in command line arguments) and the dataset loading. One should always check these parts and make sure they are correct before running our algorithm. 

## Citation
If you find our work helpful, please cite:

```
@inproceedings{zhang-etal-2024-pacost,
    title = "{P}a{C}o{ST}: Paired Confidence Significance Testing for Benchmark Contamination Detection in Large Language Models",
    author = "Zhang, Huixuan  and
      Lin, Yun  and
      Wan, Xiaojun",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.97/",
    doi = "10.18653/v1/2024.findings-emnlp.97",
    pages = "1794--1809",
}