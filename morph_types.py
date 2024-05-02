from mecab import Morpheme

# a mapping of all morph types to functions that determine whether a part-of-speech tag is that morph type
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

# morphs to exclude when searching the dictionary
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

# parts of speech to exclude from results returned from the dictionary
exclude_dictionary_results = [
    "Particle",
    "Determiner",
    "Suffix",
    "Dependent noun",
    "Auxiliary verb",
    "Auxiliary adjective"
]

def is_morph_type(morph: Morpheme | str, types: list[str] | str) -> bool:
    """
    Determine whether a Morpheme or part-of-speech tag is one of a given set of
    morph types.

    Args:
        morph (Morpheme | str): Either the Morpheme or the part-of-speech tag
            from a Morpheme.
        types (list[str] | str): Either one morph type or a list of morph types.

    Returns:
        bool
    """
    if type(morph) is Morpheme:
        pos = morph.pos
    else:
        pos = morph

    if type(types) is list:
        return any(map(lambda x: morph_types[x](pos), types))
    else:
        return morph_types[types](pos)
