---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:24523
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) <!-- at revision 1427fd652930e4ba29e8149678df786c240d8825 -->
- **Maximum Sequence Length:** 256 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'XLMRobertaForSequenceClassification'})
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
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Crop calendar: how much urea per acre 3 weeks after broadcasting (2 weeks after planting) for a 3 1/2 month variety?', 'Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: Fertilizer recommendation (kg/ac)'],
    ['Crop calendar: how much urea per acre 3 weeks after broadcasting (2 weeks after planting) for a 3 1/2 month variety?', 'Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: Time duration | Urea | T.S.P | M.O.P'],
    ['Crop calendar: how much urea per acre 3 weeks after broadcasting (2 weeks after planting) for a 3 1/2 month variety?', 'Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: For 3 1/2 month variety'],
    ['Crop calendar: how much urea per acre 3 weeks after broadcasting (2 weeks after planting) for a 3 1/2 month variety?', 'Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: Broadcasting/planting (basal fertilizer) | | 22 |'],
    ['Crop calendar: how much urea per acre 3 weeks after broadcasting (2 weeks after planting) for a 3 1/2 month variety?', 'Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: Broadcasting/planting (basal fertilizer) | | 22 | ; 3 weeks after broadcasting / 2 weeks after planting | 20 | |'],
]
scores = model.predict(pairs)
print(scores)
# [6.4330e-04 8.7240e-04 3.3347e-01 2.5024e-03 9.6513e-01]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Crop calendar: how much urea per acre 3 weeks after broadcasting (2 weeks after planting) for a 3 1/2 month variety?',
    [
        'Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: Fertilizer recommendation (kg/ac)',
        'Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: Time duration | Urea | T.S.P | M.O.P',
        'Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: For 3 1/2 month variety',
        'Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: Broadcasting/planting (basal fertilizer) | | 22 |',
        'Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: Broadcasting/planting (basal fertilizer) | | 22 | ; 3 weeks after broadcasting / 2 weeks after planting | 20 | |',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
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

* Size: 24,523 training samples
* Columns: <code>question</code>, <code>sentence</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | question                                                                           | sentence                                                                            | label                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                             | string                                                                              | float                                                          |
  | modality | text                                                                               | text                                                                                |                                                                |
  | details  | <ul><li>min: 31 tokens</li><li>mean: 34.56 tokens</li><li>max: 36 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 50.38 tokens</li><li>max: 210 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.16</li><li>max: 1.0</li></ul> |
* Samples:
  | question                                                                                                                          | sentence                                                                                                            | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Crop calendar: how much urea per acre 3 weeks after broadcasting (2 weeks after planting) for a 3 1/2 month variety?</code> | <code>Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: Fertilizer recommendation (kg/ac)</code>       | <code>0.0</code> |
  | <code>Crop calendar: how much urea per acre 3 weeks after broadcasting (2 weeks after planting) for a 3 1/2 month variety?</code> | <code>Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: Time duration \| Urea \| T.S.P \| M.O.P</code> | <code>0.0</code> |
  | <code>Crop calendar: how much urea per acre 3 weeks after broadcasting (2 weeks after planting) for a 3 1/2 month variety?</code> | <code>Paddy Crop Calendar - Dry and Intermediate Zone (Irrigated) :: For 3 1/2 month variety</code>                 | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": 7.358214042263121
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `learning_rate`: 2e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `seed`: 20261018
- `data_seed`: 20261018
- `disable_tqdm`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.1
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
- `seed`: 20261018
- `data_seed`: 20261018
- `jit_mode_eval`: False
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
- `disable_tqdm`: True
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
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
- `hub_revision`: None
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
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0652 | 50   | 1.3133        |
| 0.1304 | 100  | 0.8976        |
| 0.1956 | 150  | 0.7367        |
| 0.2608 | 200  | 0.7499        |
| 0.3259 | 250  | 0.6246        |
| 0.3911 | 300  | 0.5973        |
| 0.4563 | 350  | 0.4909        |
| 0.5215 | 400  | 0.5545        |
| 0.5867 | 450  | 0.6083        |
| 0.6519 | 500  | 0.5304        |
| 0.7171 | 550  | 0.4581        |
| 0.7823 | 600  | 0.5088        |
| 0.8475 | 650  | 0.4743        |
| 0.9126 | 700  | 0.4672        |
| 0.9778 | 750  | 0.4579        |
| 1.0430 | 800  | 0.4333        |
| 1.1082 | 850  | 0.3891        |
| 1.1734 | 900  | 0.3943        |
| 1.2386 | 950  | 0.4059        |
| 1.3038 | 1000 | 0.3769        |
| 1.3690 | 1050 | 0.3498        |
| 1.4342 | 1100 | 0.4163        |
| 1.4993 | 1150 | 0.2941        |
| 1.5645 | 1200 | 0.3175        |
| 1.6297 | 1250 | 0.3234        |
| 1.6949 | 1300 | 0.345         |
| 1.7601 | 1350 | 0.344         |
| 1.8253 | 1400 | 0.3012        |
| 1.8905 | 1450 | 0.3367        |
| 1.9557 | 1500 | 0.294         |


### Training Time
- **Training**: 2.5 hours

### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 5.7.0
- Transformers: 4.57.6
- PyTorch: 2.14.1+cpu
- Accelerate: 1.15.0
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

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