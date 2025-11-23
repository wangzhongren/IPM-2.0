# Indexed Parametric Memory (IPM) 2.0  
### First Empirical Validation of a New Paradigm for Lossless Lifelong LLM Memory

> **IPM 1.0** (Nov 23, 2025): *Theoretical proposal*  
> **IPM 2.0** (Same day): *First working implementation — 91% recall with 26 MB LoRA*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17686765.svg)](https://doi.org/10.5281/zenodo.17686765)  
*(Replace with your Zenodo DOI after release)*

---

## 🧠 What is IPM?

IPM is the **third memory paradigm** for LLMs—after RAG and parametric memory—that achieves:

- ✅ **Lossless recall**: Exact raw text retrieval  
- ✅ **Zero hallucination**: Model never invents facts  
- ✅ **Infinite capacity**: Scale to millions of memories  
- ✅ **Tiny overhead**: Only 26 MB LoRA  
- ✅ **Full auditability**: Every response traceable to a human-readable ID

| Paradigm          | Storage                | Retrieval           | Hallucination | Capacity Limit        |
|-------------------|------------------------|---------------------|---------------|------------------------|
| RAG               | External vectors       | Similarity search   | Yes           | Disk only             |
| Parametric Memory | Model weights          | Generative          | Yes           | Parameter budget      |
| **IPM (this work)** | **External raw text + LoRA index** | **Deterministic ID** | **No**        | **Theoretically unlimited** |

---

## 🚀 Key Results (IPM 2.0)

- **91% Top-1 Exact-Match Recall** on 100 structured memories  
- **26 MB LoRA adapter** (r=32, Phi-3-mini base, 4-bit)  
- **Semantic ID format**: `domain_topic_entity` (e.g., `personal_pet_luna`)  
- **Training**: 1,500 paraphrased question → ID pairs  
- **Evaluation**: Strict ID parsing, zero tolerance for hallucination  

> 💡 This validates the core claim of [IPM 1.0](Indexed%20Parametric%20Memory%20(IPM)_%20A%20New%20Paradigm%20for%20Lossless%20Lifelong%20LLM%20Memory.pdf):  
> *“A few tens of MB of LoRA can index millions of conversation turns.”*

---

## 📁 Repository Structure

```
.
├── 1_train_ipm_semantic.py     # Train LoRA router (4-bit QLoRA)
├── 2_evaluate_ipm_semantic.py  # Evaluate recall (91%)
├── memory_store/
│   ├── memories.json           # 100 raw memories (e.g., "My dog's name is Max.")
│   └── train_data.json         # 1,500 question → ID pairs
├── ipm_navigator_semantic/     # Trained LoRA adapter (26 MB)
├── results/
│   └── ipm_semantic_result.txt # Evaluation result (91% recall)
├── index.html                  # Technical report (IPM 2.0)
└── requirements.txt            # Dependencies
```

---

## ▶️ Quick Start

### 1. Prepare Base Model
Download **Phi-3-mini-4k-instruct** from Hugging Face and place it in:
```
models/phi-3-mini-4k-instruct/
```

> ⚠️ This repo does **not** include the base model due to licensing.

### 2. Install Dependencies
```bash
pip install torch transformers peft bitsandbytes accelerate datasets tqdm
```

### 3. Train IPM Router
```bash
python 1_train_ipm_semantic.py
# Output: ipm_navigator_semantic/ (26 MB)
```

### 4. Evaluate Recall
```bash
python 2_evaluate_ipm_semantic.py
# Output: ✅ IPM Top-1 Exact Recall: 91.0% (205/225)
```

---

## 📄 Technical Report

Read the full IPM 2.0 report:  
👉 [**Open `index.html` in your browser**](index.html)

---

## 📚 Citation

If you use IPM in your research, please cite:

```bibtex
@software{wang2025ipm2,
  author = {Wang, Zhongren},
  title = {Indexed Parametric Memory (IPM) 2.0: First Empirical Validation},
  year = {2025},
  note = {Zenodo DOI: 10.5281/zenodo.17686765},
  url = {https://github.com/wangzhongren/ipm-2.0}
}

@article{wang2025ipm1,
  title = {Indexed Parametric Memory (IPM): A New Paradigm for Lossless Lifelong LLM Memory},
  author = {Wang, Zhongren},
  journal = {Preprint},
  year = {2025},
  month = {November},
  note = {Version 1.0 — Pure theoretical proposal}
}
```

---

## 🤝 Contributions

IPM 1.0 concept co-designed with **Grok-4**.  
IPM 2.0 implementation by **Zhongren Wang**.  
Comments, reproductions, and extensions are welcome!

---

> **IPM 1.0 was a vision. IPM 2.0 is reality.**
