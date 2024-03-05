from functools import reduce
import re
import sys
import jamotools
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from dictionary import query_dictionary, get_query_str


def ends_in_vowel(str: str) -> bool:
    char = str[-1]

    if not jamotools.is_syllable(char):
        return False

    jamos = jamotools.split_syllable_char(char)
    return ord(jamos[-1]) >= 0x1161 and ord(jamos[-1]) <= 0x1175


model_name = "JesseStover/L2AI-dictionary-klue-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

try:
    query = sys.argv[1]
except IndexError:
    print("Usage: %s \"<query>\"" % sys.argv[0])
    quit()

groups = query_dictionary(query)
qstr = get_query_str(query)
words = []

for group in groups:
    query_strs = group[0]["queryStrs"]
    variations = group[0]["variations"]

    i = 0
    while i < len(query_strs) and query_strs[i] not in qstr:
        i = i + 1
    if i == len(query_strs):
        msg ="%s not found in query %s" % (group[0]["variations"][0], qstr)
        raise ValueError(msg)

    word = variations[i]
    prompt = "\"%s\"에 있는 \"%s\"의 정의는 " % (query, word)

    senses = reduce(
        lambda l, r: (
            l + ["(%s) %s" % (r["partOfSpeechKo"], x) for x in r["sensesKo"]]
        ),
        group,
        []
    )

    if len(senses) == 1:
        result = [1.0]

    else:
        candidates = []

        for sense in senses:
            if sense.endswith("."):
                sense = sense[:-1]

            sense_stripped = re.sub(r"[^\u3131-\uD79DA-Za-z\d]", "", sense)
            end = "예요." if ends_in_vowel(sense_stripped) else "이에요."
            candidates.append("\"%s\"%s" % (sense, end))

        inputs = tokenizer(
            [[prompt, candidate] for candidate in candidates],
            return_tensors="pt",
            padding=True
        )

        labels = torch.tensor(0).unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                **{k: v.unsqueeze(0) for k, v in inputs.items()},
                labels=labels
            )

        result = [float(x) for x in outputs.logits.softmax(1)[0]]

    start = 0
    print(group[0]["variations"])
    for word in group:
        end = start + len(word["sensesKo"])
        word["senseScores"] = result[start:end]
        start = end
        ranks = list(zip(word["sensesEn"], word["senseScores"]))
        ranks = sorted(ranks, key=lambda x: x[1])

        for sense, rank in reversed(ranks):
            print(rank, sense)

    print()
