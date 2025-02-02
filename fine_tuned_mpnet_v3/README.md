---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:891
- loss:ContrastiveLoss
widget:
- source_sentence: 'note 14 4g 8/256gb (green) | color: green, ram: 8, rom: 256gb'
  sentences:
  - 'poco x7 12/512gb серебро (silver) | color: silver, ram: 12, rom: 512gb'
  - 'apple iphone 16 pro 128gb пустынный титан (desert titanium) esim | color: desert
    titanium, ram: 8, rom: 128gb'
  - 'xiaomi redmi note 14 8/256gb зеленый (green) | color: green, ram: 8, rom: 256gb'
- source_sentence: 'infinix hot 50 6/256gb (gray) | color: gray, ram: 6, rom: 256gb'
  sentences:
  - 'infinix hot 50 6/256gb зеленый (sage green) eac | color: sage green, ram: 6,
    rom: 256gb'
  - 'xiaomi redmi 14c 8/256gb фиолетовый (purple) | color: purple, ram: 8, rom: 256gb'
  - 'apple iphone 15 256gb черный (black) nano sim + esim | color: black, ram: nan,
    rom: 256gb'
- source_sentence: 'note 14 pro 8/256gb (ocean blue) | color: ocean blue, ram: 8,
    rom: 256gb'
  sentences:
  - 'apple ipad air 13 (2024) 1tb wi-fi серый космос (space gray) | color: space gray,
    ram: nan, rom: 1024gb'
  - 'xiaomi redmi note 14 pro 8/256gb белый (white) | color: white, ram: 8, rom: 256gb'
  - 'xiaomi redmi pad se 4g 8.7 4/128gb синий (blue) | color: blue, ram: 4, rom: 128gb'
- source_sentence: 'poco m6 pro 12/512gb (purple) | color: purple, ram: 12, rom: 512gb'
  sentences:
  - 'realme 12+ 8/256gb зеленый (green) eac | color: green, ram: 8, rom: 256gb'
  - 'apple iphone 14 256gb темная ночь (midnight) nano sim + esim | color: midnight,
    ram: nan, rom: 256gb'
  - 'poco m6 pro 12/512gb синий (blue) | color: blue, ram: 12, rom: 512gb'
- source_sentence: 'mi 14t 5g 12/256gb (green) | color: green, ram: 12, rom: 256gb'
  sentences:
  - 'infinix note 30i 8/128gb черный (black) eac | color: black, ram: 8, rom: 128gb'
  - 'xiaomi 14t 12/256gb зелёный (lemon green) | color: lemon green, ram: 12, rom:
    256gb'
  - 'poco c61 4/128gb белый (white) | color: white, ram: 4, rom: 128gb'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: pearson_cosine
      value: 0.8797406547235329
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.845792518336755
      name: Spearman Cosine
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'mi 14t 5g 12/256gb (green) | color: green, ram: 12, rom: 256gb',
    'xiaomi 14t 12/256gb зелёный (lemon green) | color: lemon green, ram: 12, rom: 256gb',
    'poco c61 4/128gb белый (white) | color: white, ram: 4, rom: 128gb',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.8797     |
| **spearman_cosine** | **0.8458** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 891 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 891 samples:
  |         | sentence_0                                                                         | sentence_1                                                                        | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            | float                                                          |
  | details | <ul><li>min: 23 tokens</li><li>mean: 27.35 tokens</li><li>max: 38 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 34.3 tokens</li><li>max: 50 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.44</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                    | sentence_1                                                                                      | label            |
  |:------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:-----------------|
  | <code>15 256gb (green) | color: green, ram: <na>, rom: 256gb</code>           | <code>apple iphone 15 256gb зеленый (green) dualsim | color: green, ram: nan, rom: 256gb</code> | <code>0.0</code> |
  | <code>realme c61 6/128gb (gold) | color: gold, ram: 6, rom: 128gb</code>      | <code>realme c61 6/256gb золотой (gold) | color: gold, ram: 6, rom: 256gb</code>                | <code>0.0</code> |
  | <code>samsung f15 4/128gb (purple) | color: purple, ram: 4, rom: 128gb</code> | <code>samsung galaxy f15 4/128gb фиолетовый | color: purple, ram: 4, rom: 128gb</code>          | <code>1.0</code> |
* Loss: [<code>ContrastiveLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#contrastiveloss) with these parameters:
  ```json
  {
      "distance_metric": "SiameseDistanceMetric.COSINE_DISTANCE",
      "margin": 0.5,
      "size_average": true
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 10
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | spearman_cosine |
|:------:|:----:|:-------------:|:---------------:|
| 1.0    | 56   | -             | 0.5701          |
| 1.7857 | 100  | -             | 0.7464          |
| 2.0    | 112  | -             | 0.7828          |
| 3.0    | 168  | -             | 0.8074          |
| 3.5714 | 200  | -             | 0.8109          |
| 4.0    | 224  | -             | 0.8191          |
| 5.0    | 280  | -             | 0.8400          |
| 5.3571 | 300  | -             | 0.8401          |
| 6.0    | 336  | -             | 0.8431          |
| 7.0    | 392  | -             | 0.8446          |
| 7.1429 | 400  | -             | 0.8442          |
| 8.0    | 448  | -             | 0.8451          |
| 8.9286 | 500  | 0.0107        | 0.8456          |
| 9.0    | 504  | -             | 0.8456          |
| 10.0   | 560  | -             | 0.8458          |


### Framework Versions
- Python: 3.13.1
- Sentence Transformers: 3.4.1
- Transformers: 4.48.2
- PyTorch: 2.6.0+cu124
- Accelerate: 1.3.0
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### ContrastiveLoss
```bibtex
@inproceedings{hadsell2006dimensionality,
    author={Hadsell, R. and Chopra, S. and LeCun, Y.},
    booktitle={2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06)},
    title={Dimensionality Reduction by Learning an Invariant Mapping},
    year={2006},
    volume={2},
    number={},
    pages={1735-1742},
    doi={10.1109/CVPR.2006.100}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->