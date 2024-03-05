from mecab import Morpheme

"""
Sorted by occrence in content.json:
common noun..................4162
nominal postposition.........1350
verb.........................1070
ending.......................1015
symbol.......................881
verbial postposition.........717
sentence-final punctuation...620
suffix.......................589
auxiliary....................570
adverb.......................431
number.......................303
dependent noun...............297
sentence-final ending........287
adjective....................231
noun suffix..................220
auxiliary verb...............211
determiner...................199
proper noun..................198
prefix.......................192
conjunction..................181
copula.......................136
pronoun......................113
counting noun................83
numeral......................81
interjection.................67
root.........................47
"""

morph_types = {
    "common noun": lambda pos: pos.startswith("NNG"),
    "nominal postposition": lambda pos: pos[0:3] in ["JKS", "JKC", "JKG", "JKO"],
    "verb": lambda pos: pos.startswith("VV"),
    "ending": lambda pos: pos[0:2] in ["EC", "ET"],
    "symbol": lambda pos: pos.startswith("S") and pos[1] not in ["F"],
    "verbial postposition": lambda pos: pos[0:3] in ["JKB", "JKV", "JKQ"],
    "sentence-final punctuation": lambda pos: pos.startswith("SF"),
    "verb suffix": lambda pos: pos.startswith("XSV"),
    "adjective suffix": lambda pos: pos.startswith("XSA"),
    "auxiliary": lambda pos: pos.startswith("JX"),
    "adverb": lambda pos: pos.startswith("MA"),
    "dependent noun": lambda pos: pos.startswith("NNB") and not pos.startswith("NNBC"),
    "number": lambda pos: pos.startswith("SN"),
    "sentence-final ending": lambda pos: pos[0:2] in ["EP", "EF"],
    "adjective": lambda pos: pos.startswith("VA"),
    "noun suffix": lambda pos: pos.startswith("XSN"),
    "auxiliary verb": lambda pos: pos.startswith("VX"),
    "determiner": lambda pos: pos.startswith("MM"),
    "proper noun": lambda pos: pos.startswith("NNP"),
    "prefix": lambda pos: pos.startswith("XP"),
    "conjunction": lambda pos: pos.startswith("JC"),
    "copula": lambda pos: pos.startswith("VC"),
    "pronoun": lambda pos: pos.startswith("NP"),
    "counting noun": lambda pos: pos.startswith("NNBC"),
    "numeral": lambda pos: pos.startswith("NR"),
    "interjection": lambda pos: pos.startswith("IC"),
    "root": lambda pos: pos.startswith("XR"),
    "unknown": lambda pos: pos[0:2] in ["UN", "NA"]
}

exclude_general = [
    "symbol",
    "sentence-final punctuation",
    "number",
    "copula",
    "unknown"
]

exclude_dictionary = [
    "nominal postposition",
    "ending",
    "symbol",
    "verbial postposition",
    "sentence-final punctuation",
    "auxiliary",
    "number",
    "sentence-final ending",
    "noun suffix",
    "conjunction",
    "copula",
    "unknown"
]

exclude_dictionary_results = [
    "Particle",
    "Determiner",
    "Suffix",
    "Dependent noun",
    "Auxiliary verb",
    "Auxiliary adjective"
]

exclude_idioms = [
    "nominal postposition",
    "symbol",
    "sentence-final punctuation",
    "number",
    "unknown"
]

dependent_types = [
    "nominal postposition",
    "ending",
    "verbial postposition",
    "verb suffix",
    "adjective suffix",
    "auxiliary",
    "dependent noun",
    "sentence-final ending",
    "auxiliary verb",
    "conjunction"
]


def is_morph_type(morph: Morpheme | str, types: list[str] | str) -> bool:
    if type(morph) is Morpheme:
        pos = morph.pos
    else:
        pos = morph

    if type(types) is list:
        return any(map(lambda x: morph_types[x](pos), types))
    else:
        return morph_types[types](pos)


def get_morph_type(morph: Morpheme) -> str:
    if type(morph) is Morpheme:
        pos = morph.pos
    else:
        pos = morph

    for key, function in morph_types.items():
        if function(pos):
            return key

    print(morph)
    raise ValueError("Morph pos not found: %s" % str(pos))
