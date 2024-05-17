# ProSST
Code for ProSST: A Pre-trained Protein Sequence and Structure Transformer with Disentangled Attention.

## 1 Install

```shell
git clone https://github.com/ginnm/ProSST.git
cd ProSST
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 2 Structure quantizer

![Structure quantizer](images/structure_quantizer.png)

```python
from prosst.structure.quantizer import PdbQuantizer
processor = PdbQuantizer(structure_vocab_size=2048) # can be 20, 128, 512, 1024, 2048, 4096
result = processor("example_data/p1.pdb", return_residue_seq=False)
```

Output:
```
[407, 998, 1841, 1421, 653, 450, 117, 822, ...]
```


## 3 ProSST models have been uploaded to huggingface 🤗 Transformers
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
model = AutoModelForMaskedLM.from_pretrianed("AI4Protein/ProSST-2048", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
```

See [AI4Protein/ProSST-2048](https://huggingface.co/AI4Protein?search_models=ProSST) for more models.

## 4 Zero-shot mutant effect prediction

### 4.1 Example notebook
[Zero-shot mutant effect prediction](zero_shot/score_mutant.ipynb)

### 4.2 Run ProteinGYM Benchmark
Download dataset from [Google Driver](https://drive.google.com/file/d/1lSckfPlx7FhzK1FX7EtmmXUOrdiMRerY/view?usp=sharing).
(This file contains quantized structures within ProteinGYM).

```shell
cd example_data
unzip proteingym_benchmark.zip
```

```shell
python zero_shot/proteingym_benchmark.py --model_path AI4Protein/ProSST-2048 \
--structure_dir example_data/structure_sequence/2048
```
<!-- 
## 5 Representation
```

```

## 6 Transfer-Learning
```

``` -->

## Citation

If you use ProSST in your research, please cite the following paper:

```

```