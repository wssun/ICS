from functools import lru_cache
from typing import List
import sys
from transformers import (RobertaTokenizer)
import nltk

REGEX_TEXT = ("(?<=[a-z0-9])(?=[A-Z])|"
              "(?<=[A-Z0-9])(?=[A-Z][a-z])|"
              "(?<=[0-9])(?=[a-zA-Z])|"
              "(?<=[A-Za-z])(?=[0-9])|"
              "(?<=[@$.'\"])(?=[a-zA-Z0-9])|"
              "(?<=[a-zA-Z0-9])(?=[@$.'\"])|"
              "_|\\s+")

if sys.version_info >= (3, 7):
    import re
    SPLIT_REGEX = re.compile(REGEX_TEXT)
else:
    import regex
    SPLIT_REGEX = regex.compile("(?V1)"+REGEX_TEXT)


@lru_cache(maxsize=5000)
def split_identifier_into_parts(identifier: str) -> List[str]:
    """
    Split a single identifier into parts on snake_case and camelCase
    """
    identifier_parts = list(s for s in SPLIT_REGEX.split(identifier) if len(s)>0)
    if len(identifier_parts) == 0:
        return [identifier]
    return identifier_parts

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    # tokenizer.do_lower_case = True
    review = " comprehensive"
    # print(review)
    # id, code = review.split('\t')
    # review = " ".join(split_identifier_into_parts(code))
    # print(review)
    s = tokenizer.tokenize(review)
    print(s)
    s = [word.lower() for word in s]
    s = [word for word in s if not word in set(
        nltk.corpus.stopwords.words('english'))]
    s = [word for word in s if word.isalpha()]
    m = []
    # m = [stemmer.stem(word) for word in s]
    for word in s:
        if word[0] == 'ġ':
            word = word[1:]
        if word != 'ĉ':
            m.append(word)
    print(m)