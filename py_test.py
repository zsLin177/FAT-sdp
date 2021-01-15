import string
import re


def preprocess(text):
    rex = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    res = re.sub(rex, '', text)
    return res


test = 'Happy I am happy are you happy'


def func1(s):
    lst = s.split(" ")
    res = {}
    for word in lst:
        res[word] = res.get(word, 0) + 1
    print(res)


# func1(test)
text = 'I scream, you scream, we all scream for ice-cream!'
print(preprocess(text))
