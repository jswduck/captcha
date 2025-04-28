# Build a captcha solver

## Introduction

A website uses Captchas on a form to keep the web-bots away. However, the captchas it generates, are quite similar each time:
- the number of characters remains the same each time  
- the font and spacing is the same each time  
- the background and foreground colors and texture, remain largely the same
- there is no skew in the structure of the characters.  
- the captcha generator, creates strictly 5-character captchas, and each of the characters is either an upper-case character (A-Z) or a numeral (0-9).


We are provided with 26 sample captchas. The task is to build a captcha solver to identify the unsees captchas. 

There are a few well-known OCR libraries that have been leveraged to solve captchas. Both Tesseact and TrOCR are tried, and eventually TrOCR is used.  

### Tesseract 
[Tesseract](https://github.com/tesseract-ocr/tesseract) exists before the GenAI era. It primarily uses a **hybrid approach combining traditional image processing and a neural network-based character classifier**.



#### Strengths:
- Fast and lightweight
- Works well for clean, document-style text
- Minimal training required for deployment

#### Limitations:
- Sensitive to image noise, unusual fonts, or tightly spaced characters
- Less adaptable to domain-specific text (e.g., captchas or stylized layouts)
- Errors common in visually similar characters like `0/O`, `1/I`, `5/S`, etc.

---

### TrOCR.

[TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr) (Transformer-based OCR) is an end-to-end neural OCR system introduced by Microsoft. It applies **transformer-based encoder-decoder modeling** directly to the OCR task.

#### Strengths:
- Robust to font variation, background noise, and layout irregularities
- Learns semantic context and can disambiguate similar characters using visual + linguistic cues
- Extensible to both printed and handwritten OCR tasks

#### Limitations:
- Computationally heavy (transformer encoder and decoder)
- Requires GPU for training and efficient inference
- May overfit or hallucinate if not fine-tuned properly on domain-specific data

### Experiements conducted
A jupyter notebook has been attached here, documenting the experiments conducted. 

- First, we applied Tesseract, which achieved a CER (character level) 20%, and a WER (word level) of 53.85%. 

- Given its relatively poor performance, we then applied TrOCR. To balance inference speed and accuracy. we used **trocr-base-printed**, which is fine-tuned on the SROIE dataset. This dataset contains scanned receipts in English.

   - We tried two passes. First is without preprocessing, i.e., the captchas are fed as-is to the model; and then captchas went through simple pre-processings before being fed into the model. TrOCR achieves better performance in the second pass, with a CER of ~3.85% and WER ~15.38%. 
   - We then augmented the set of 26 samples with some image manipulations. Given the known patterns of the unseen captchas, we only apply simple manipulations, e.g, adding noising, adjust brightness, etc. We generate additional 78 captchas. 
   - TrOCR was applied on the new captchas. It showed a dropped performance though, with a CER of ~7% and WER of ~21%. 
   - A detailed analysis shows that TrOCR still face challenges in diffentiating paris like "0" and "O", "S" and "5", "7" and "Z".
- Finally a fine tuning was attempted with the augmented data set, but given the limited time and compute resources available, the fine tuning is not done completely. 


## How to use

Based on the experiments outcome, we decided to use **trocr-base-printed** model. A class **CaptchaTrOCR** is defined in captcha_trocr.py. 


To run the inference, first create an instance of **CaptchaTrOCR**. 

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
captcha_solver = CaptchaTrOCR(device, "../config.yaml") #second parameter is the config.yaml location.
```

Then either call it with an **individial image file**:

```python
preds = captcha_solver(, im_path = "../data/input/input02.jpg") 
```

**im_path** is the location of the input file. If the file name does not contain folder name, default input folder would be used (this is set in the config.ymal **raw_dir**) 

```yaml
data:
  raw_dir: "../data/input"
  output_dir: "../data/output"
  aug_dir: "../data/augmented"
  finetune_eval_dir: "../data/finetune_eval"
```

or call with a folder containing batch of image files: 
```python
preds = captcha_solver(, im_path = "../data/input/") 
```

If no value is passed to **im_path**, all the jpg files under the default **raw_dir** will be loaded for inference. 

If there is no **save_path** provided, predicted captcha text will be written as text file under the default folder set by **"output_dir"** in config.yaml. 

In config.yaml, there is anothe switch determining whether preprocessing will be applied. 

```yaml
model:
  preprocessing: false
```

Setting this switch to **true** is expected to improve the performance (both CER and WER)

**NOTE** This project use python 3.10.13