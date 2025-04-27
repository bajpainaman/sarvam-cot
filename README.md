# sarvam-cot
# Proposal: Building an Indic Chain-of-Thought (CoT) Dataset & CoT-Finetuning Sarvam-1

**Objective:**  
Create a high-quality, instruction-tuned Chain-of-Thought (CoT) dataset in Hindi (and other Indic languages) by distilling reasoning traces from GPT-4 / R1 onto samples from open-source Indic corpora (e.g., Sangraha). Then fine-tune Sarvam-1 to produce “Sarvam-1-CoT,” capable of step-by-step explanations in Hindi.

---

## 1. Data Sources

| Source                      | Content                                                 | Link                                                                                       |
|-----------------------------|---------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **Sangraha (Hindi split)**  | ~34 GB of curated Hindi text in Parquet shards          | https://huggingface.co/datasets/ai4bharat/sangraha/tree/main/verified/hin/                  |
| **IndicGLUE / MMLU (Indic)**| Reasoning & knowledge benchmarks in 10 Indic languages  | https://github.com/AI4Bharat/indic-glue; https://github.com/huggingface/transformers/tree/main/examples/research_projects/mmlu |
| **IndicGenBench**           | Summarization & QA tasks in Indic                       | https://github.com/google-research/indic-GenBench                                                |

---

## 2. Dataset Construction

1. **Shard Selection**  
   - Randomly sample **12 Parquet files** from Sangraha’s `verified/hin/` directory.  
   - Within each shard, select **100 examples** → **1,200 total**.

2. **Prompt Template**  
   ```text
   You are an expert in Hindi explanations. Think step-by-step.
   Q: "{input_text}"
   A (Chain of Thought):
   1.
   2.
   …
   Answer:
   ```

3. **Distillation with GPT-4 / R1**  
   - For each sample, send the above prompt to GPT-4 (or internal R1).  
   - Extract the numbered reasoning steps + final answer.  
   - Store as JSONL:
     ```json
     {
       "instruction": "Explain step-by-step in Hindi",
       "input":    "...",
       "output":   "1. …\n2. …\nAnswer: \"…\""
     }
     ```

4. **Quality Control**  
   - Automatic filters: ensure ≥3 steps, answer length ≤256 tokens.  
   - Human spot-check 5% for coherence & faithfulness.

---

## 3. Model Finetuning

- **Base model:** `sarvamai/sarvam-1` (2 B parameters)  
- **Toolkit:** Hugging Face Transformers & `bitsandbytes` 4-bit quantization  
- **Training recipe:**  
  ```python
  from datasets import load_dataset
  from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

  ds = load_dataset("json", data_files="indic_cot.jsonl", split="train")
  tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")
  model     = AutoModelForCausalLM.from_pretrained(
                "sarvamai/sarvam-1",
                load_in_4bit=True, device_map="auto",
                torch_dtype=torch.float16
              )

  def preprocess(ex):
      enc = tokenizer(f"{ex['instruction']}\nQ: {ex['input']}\nA:", truncation=True)
      lbl = tokenizer(ex["output"], truncation=True)
      return {"input_ids": enc["input_ids"], "labels": lbl["input_ids"]}

  ds = ds.map(preprocess, remove_columns=ds.column_names)
  args = TrainingArguments(
      output_dir="sarvam1-cot", epochs=3,
      per_device_train_batch_size=8, fp16=True,
      save_total_limit=2, logging_steps=50,
  )
  Trainer(model, args, train_dataset=ds).train()
  ```

---

## 4. Compute & Timeline

| Stage                    | Resources                    | Duration  |
|--------------------------|------------------------------|-----------|
| Data sampling & distill  | 1 × A100 (GPT-4 distill)      | 1 week    |
| QC & cleanup             | 0.5 FTE annotator            | 1 week    |
| Finetuning Sarvam-1-CoT  | 4 × A100 (4-bit, 16 GB each) | 2 weeks   |
| Evaluation & analysis    | 2 × A100                     | 1 week    |

_Total: ~5 weeks_

---

## 5. Expected Impact

- **Novelty:** First open-source Indic CoT dataset; “Sarvam-1-CoT” will be the first Indian LLM exhibiting explicit chain-of-thought reasoning.  
- **Feasibility:** Leverages existing high-quality corpora (Sangraha), mature distillation (GPT-4/R1), and efficient fine-tuning (4-bit quant).  
- **Follow-up:** Expand to other Indic languages, integrate into conversational UI, downstream tasks (QA, tutoring, reasoning).  

---

*For questions or collaboration, reach out to [naman@encryptech.ai](mailto:naman@encryptech.ai).*
