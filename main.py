import sys

from src.crf_ner import *

import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm
from spacy.training import Example
from spacy.tokens import Doc

class CharacterLevelTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = list(text)
        #text.split(" ")
        spaces = [True] * len(words)
        # # Avoid zero-length tokens
        # for i, word in enumerate(words):
        #     if word == "":
        #         words[i] = " "
        #         spaces[i] = False
        # # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
           spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)


def main():
    print()
    TRAIN_DATA = [
    ('Who is Nishanth?', {
        'entities': [(7, 15, 'GIRL')]
    }),
     ('Who is Kamal Khumar?', {
        'entities': [(7, 19, 'GIRL')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'GIRL'), (18, 24, 'GIRL')]
    })
    ]

    TRAIN_DATA = [
    ('Who is Nishanth?', {
        'entities': [(0, 1, 'CNST'),
                    (1, 2, 'CNST'),
                    (2, 3, 'CNST'),
                    (3, 4, 'CNST'),
                    (4, 5, 'CNST'),
                    (5, 6, 'CNST'),
                    (6, 7, 'CNST'),
                    (7, 8, 'Person'),
                    (8, 9, 'Person'),
                    (9, 10, 'Person'),
                    (10, 11, 'Person'),
                    (11, 12, 'Person'),
                    (12, 13, 'Person'),
                    (13, 14, 'Person'),
                    (14, 15, 'Person'),
                    (15, 16, 'CNST'),
                    ]
    }),
    ('Who is Abdulman?', {
        'entities': [(0, 1, 'CNST'),
                    (1, 2, 'CNST'),
                    (2, 3, 'CNST'),
                    (3, 4, 'CNST'),
                    (4, 5, 'CNST'),
                    (5, 6, 'CNST'),
                    (6, 7, 'CNST'),
                    (7, 8, 'Person'),
                    (8, 9, 'Person'),
                    (9, 10, 'Person'),
                    (10, 11, 'Person'),
                    (11, 12, 'Person'),
                    (12, 13, 'Person'),
                    (13, 14, 'Person'),
                    (14, 15, 'Person'),
                    (15, 16, 'CNST'),
                    ]
    })
    ]

    TRAIN_DATA = [
    ('AAAAABBBBBBBB?', {
        'entities': [(0, 1, 'ACLS'),
                    (1, 2, 'ACLS'),
                    (2, 3, 'ACLS'),
                    (3, 4, 'ACLS'),
                    (4, 5, 'ACLS'),
                    (5, 6, 'BCLS'),
                    (6, 7, 'BCLS'),
                    (7, 8, 'BCLS'),
                    (8, 9, 'BCLS'),
                    (9, 10, 'BCLS'),
                    (10, 11, 'BCLS'),
                    (11, 12, 'BCLS'),
                    (12, 13, 'BCLS'),
                    (13, 14, 'BCLS'),
                    ]
    }),
    ('AAAAAAABBBBB', {
        'entities': [(0, 1, 'ACLS'),
                    (1, 2, 'ACLS'),
                    (2, 3, 'ACLS'),
                    (3, 4, 'ACLS'),
                    (4, 5, 'ACLS'),
                    (5, 6, 'ACLS'),
                    (6, 7, 'ACLS'),
                    (7, 8, 'BCLS'),
                    (8, 9, 'BCLS'),
                    (9, 10, 'BCLS'),
                    (10, 11, 'BCLS'),
                    (11, 12, 'BCLS'),
                    ]
    }),
    ]

    model = None
    output_dir=Path("")
    n_iter=100

    if model is not None:
        nlp = spacy.load(model)  
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  
        print("Created blank 'en' model")
        nlp.tokenizer = CharacterLevelTokenizer(nlp.vocab)
        doc = nlp("What's happened to me?")
        print([token.text for token in doc])

    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in tqdm(TRAIN_DATA):
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example],
                    drop=0.5,  
                    sgd=optimizer,
                    losses=losses)
            print(losses)

    TEST_DATA = [
    ('ABC EFG', {
        'entities': [(0, 1, 'ACLS'),(1, 2, 'BCLS')]
    })
    ]
    doc = None

    for text, _ in TEST_DATA:
        doc = nlp(text)
        print([(token.text,token.idx) for token in doc])
        s = []
        for token in doc:
            s.append(token.text)
        s = ''.join(s)
        print(f"Test:#{s}#")
        print('Entities', [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])
        
if __name__ == "__main__":
    main()
