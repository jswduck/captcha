# Build a captcha solver

## Introduction

A website uses Captchas on a form to keep the web-bots away. However, the captchas it generates, are quite similar each time:
- the number of characters remains the same each time  
- the font and spacing is the same each time  
- the background and foreground colors and texture, remain largely the same
- there is no skew in the structure of the characters.  
- the captcha generator, creates strictly 5-character captchas, and each of the characters is either an upper-case character (A-Z) or a numeral (0-9).


We are provided with 26 sample captchas. The task is to build a captcha solver to identify the unsees captchas. 

There are a few well-known OCR libraries that have been leveraged to solve captchas. In this project, Tesseact and TrOCR are tried. 

### Tesseract 
[Tesseract](https://github.com/tesseract-ocr/tesseract) is one of them, which exists before the GenAI era. It primarily uses a **hybrid approach combining traditional image processing and a neural network-based character classifier**.



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

### Flow of the experiements
A jupyter notebook has been attached here, documenting the experiments conducted. 

- First, we applied Tesseract, which achieved a 20%, and a WER of 53.85%. 

- Given its relatively poor performance, we then applied TrOCR. To balance inference speed and accuracy. we used **trocr-base-printed**, which is fine-tuned with 

   - We tried two passes. First is without preprocessing, i.e., the captchas are fed as-is to the model; and then captchas went through simple pre-processings before being fed into the model. TrOCR achieves better performance in the second pass, with a CER of 3.85% and WER 15.38%. 
   - We then augmented the set of 26 samples with some image manipulations. Given the known patterns of the unseen captchas, we only apply simple manipulations, e.g, adding noising, adjust brightness, etc. We generate additional 78 captchas. 
   - TrOCR was applied on the new captchas. It showed a dropped performance, with a CER of 7.18% and WER of 21.79%. 
   - A detailed analysis shows that TrOCR still face challenges in diffentiating paris like "0" and "O", "S" and "5", "7" and "Z".
- Finally a fine tuning was attempted with the augmented data set, but given the limited time and compute resources available, the fine tuning is not done completely. 


## How to use

The experiments are captured in the attached notebook. It calls the functions defined in captcha.py (follwing the provided template with additional functions). Some other utilty functions are defined in util.py

To run the inference, first create an instance of **Captch**. 

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
captcha_solver = Captcha(device, "../config.yaml")
```

Then either call it with an individial image file:

```python
preds = captcha_solver(mode = "tesseract", im_path = "../data/input/input02.jpg") 
```
or a folder containing a batch of images:
```python
preds = captcha_solver(mode = "tesseract", im_path = "") 
```

**mode** decides whether to use **tesseract** or **TrOCR**. If no **im_path** is provided, it will load all the .jpg files in **"data/input/"** folder. If there is no **save_path** provided, predicted captcha text will be written as text file to a file under **"data/output"** folder. 





You can modify the configuration settings in `config.yaml` to adjust settings related to data and model.


**NOTE** This project use python 3.10.13