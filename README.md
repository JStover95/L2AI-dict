# L2AI-dict

This is a tool for demonstrating how to use my [L2AI-dictionary model](https://huggingface.co/JesseStover/L2AI-dictionary-klue-bert-base) that is a fine-tuned checkpoint of [klue/bert-base](https://huggingface.co/klue/bert-base).

**Note:** This script must be run in a Python 3.12 environment with PyTorch. Refer to [this link](https://pytorch.org/get-started/locally/) for installing PyTorch locally.

## Quickstart

On Linux, run `./deploy.sh` to initialize a Docker container with an empty MongoDB container and store the contents of `dict.json` in MongoDB.

On Windows, manually create a new MongoDB container. Then, open Python in a command line and run the following commands:

```
>>> from dictionary import create_db
>>> create_db()
```

Then, run the `infer.py` script from the command line and provide a query:

```
python infer.py "강아지는 뽀송뽀송하다."
```

English definitions of each word in the supplied query, and their respective scores from 0 to 1 (1 being the highest confidence that the definition is correct), will be printed to the terminal. An example output will be:

```
강아지
0.9999988079071045 puppy: The pup of a dog.
1.2131376934121363e-06 baby; sweetheart: A word of endearment used to refer to or address one's child or grandchild.

뽀송뽀송
0.9997172951698303 fuzzily; downily: In the state of having very short, soft hair or something similar.
0.0002767054829746485 in a dry and tender state: In the state of being dried yet soft. 
5.952259471087018e-06 smoothly; softly: In a state in which one's skin or face is soft and smooth.
```

Alternatively, you can import `get_inference` from `infer.py` for use in other modules.

## L2AI-dictionary

The L2AI-dictionary model is fine-tuned for multiple choice, specifically for selecting the best dictionary definition of a given word in a sentence. Below is an example usage:

```python
import numpy as np
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

model_name = "JesseStover/L2AI-dictionary-klue-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

prompts = "\"강아지는 뽀송뽀송하다.\"에 있는 \"강아지\"의 정의는 "
candidates = [
    "\"(명사) 개의 새끼\"예요.",
    "\"(명사) 부모나 할아버지, 할머니가 자식이나 손주를 귀여워하면서 부르는 말\"이예요."
]

inputs = tokenizer(
    [[prompt, candidate] for candidate in candidates],
    return_tensors="pt",
    padding=True
)

labels = torch.tensor(0).unsqueeze(0)

with torch.no_grad():
    outputs = model(
        **{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels
    )

print({i: float(x) for i, x in enumerate(outputs.logits.softmax(1)[0])})
```

Training data was procured under Creative Commons [CC BY-SA 2.0 KR DEED](https://creativecommons.org/licenses/by-sa/2.0/kr/) from the National Institute of Korean Language's [Basic Korean Dictionary](https://krdict.korean.go.kr) and [Standard Korean Dictionary](https://stdict.korean.go.kr/).
