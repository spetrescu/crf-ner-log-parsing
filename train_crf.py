
import spacy
import random
from spacy.training import Example
from spacy.tokens import Doc
from tqdm import tqdm
import pandas as pd
import ast

train = pd.read_csv("data/entity_dataset/train.csv")
raw = []
label = []
a = 0
for index, row in train.iterrows():
    x = ast.literal_eval(row['input_data'])
    spec_raw = []
    for el in x:
        spec_raw.append(el)
    raw.append(spec_raw)

    spec_label = []
    x = ast.literal_eval(row['ner_labels'])
    for el in x:
        spec_label.append(el)
    label.append(spec_label)
    a += 1
    # if a == 1000:
    #     break

print("Raw")
for el in raw:
    print(el)

print("Label")
for el in label:
    print(el)

# raw = [
#     ["Need", "345", "to", "stop"],
#     ['Failed', 'to', 'renew', 'lease', 'for', '[DFSClient_NONMAPREDUCE_-274751412_1]', 'for', '1295', 'seconds.', '', 'Will', 'retry', 'shortly', '...'],
#     ['Failed', 'to', 'renew', 'lease', 'for', '[DFSClient_NONMAPREDUCE_-274751412_1]', 'for', '1295', 'seconds.', '', 'Will', 'retry', 'shortly', '...']
# ]
# label = [
#     ["CONSTANT", "GENERIC_VAR", "CONSTANT", "CONSTANT"],
#     ['CONSTANT', 'CONSTANT', 'CONSTANT', 'CONSTANT', 'CONSTANT', 'GENERIC_VAR', 'CONSTANT', 'GENERIC_VAR', 'CONSTANT', 'CONSTANT', 'CONSTANT', 'CONSTANT', 'CONSTANT', 'CONSTANT'],
#     ['CONSTANT', 'CONSTANT', 'CONSTANT', 'CONSTANT', 'CONSTANT', 'GENERIC_VAR', 'CONSTANT', 'GENERIC_VAR', 'CONSTANT', 'CONSTANT', 'CONSTANT', 'CONSTANT', 'CONSTANT', 'CONSTANT']
#     ]

start = 0
end = 0

all_entities = []

for r, l in zip(raw, label):
    
    start = 0
    end = 0

    input = " ".join(r)
    print(input)

    entities = []
    # print("Now analyzing labels")

    initial_token = 0
    for rnt, lbl in zip(r, l):
        if initial_token == 0:
            initial_token += 1
            pass
        else:
            end += 1
        print(rnt, lbl)
        if lbl == "CONSTANT":
            end += len(rnt)
            # print("(start, end)", start, end)
            continue
        else:
            start = end
            end += len(rnt)
            # print("(start, end)", start, end)
            entities.append((start, end, lbl))
    all_entities.append([input, entities])

for ent in all_entities:
    print(ent)

# sdkfsdf = "Successfully started service 'sparkExecutorActorSystem' on port 36792."
# print(f"-{sdkfsdf[64:70]}-")
# bmmmsdf = "Need abc the"
# asdfsd = "Failed to renew lease for [DFSClient_NONMAPREDUCE_-274751412_1] for 1295 seconds.  Will retry shortly ..."
# print(f"-{bmmmsdf[5:8]}-")
# print(asdfsd[26:63])

TRAIN_DATA = []

for entry in all_entities:
    TRAIN_DATA.append((entry[0], {'entities': entry[1]}))

for el in TRAIN_DATA:
    print(el)


def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        # ner = nlp.create_pipe('ner')
        ner = nlp.add_pipe('ner')
    else: 
        ner = nlp.get_pipe('ner')
        # nlp.add_pipe(ner, last=True)
       
    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Iteration" + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in tqdm(TRAIN_DATA):
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example],
                    drop=0.2,  
                    sgd=optimizer,
                    losses=losses)
            print(losses)
    return nlp

prdnlp = train_spacy(TRAIN_DATA, 3)

# Test: Failed to renew lease for 83994 today

# Save our trained Model
modelfile = "crf_ner"
# input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)

#Test your text
# test_text = input("Enter your testing text: ")
test_text = "Failed to renew lease for 83994 today"
test_text_2 = "Failed to change 5767 times yesterday"
doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

print("here")
doc = prdnlp(test_text_2)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


# TEST
print("\nNow testing...")
test = pd.read_csv("data/entity_dataset/test.csv")
raw = []
label = []
a = 0
for index, row in test.iterrows():
    x = ast.literal_eval(row['input_data'])
    spec_raw = []
    for el in x:
        spec_raw.append(el)
    raw.append(spec_raw)

    spec_label = []
    x = ast.literal_eval(row['ner_labels'])
    for el in x:
        spec_label.append(el)
    label.append(spec_label)
    a += 1
    # if a == 100:
    #     break

print("Raw")
for el in raw:
    print(el)

print("Label")
for el in label:
    print(el)

start = 0
end = 0

all_entities = []

for r, l in zip(raw, label):
    
    start = 0
    end = 0

    input = " ".join(r)
    # print(input)

    entities = []
    # print("Now analyzing labels")

    initial_token = 0
    for rnt, lbl in zip(r, l):
        if initial_token == 0:
            initial_token += 1
            pass
        else:
            end += 1
        # print(rnt, lbl)
        if lbl == "CONSTANT":
            end += len(rnt)
            # print("(start, end)", start, end)
            continue
        else:
            start = end
            end += len(rnt)
            # print("(start, end)", start, end)
            entities.append((start, end, lbl))
    all_entities.append([input, entities])

for ent in all_entities:
    print(ent)

TEST_DATA = []

for entry in all_entities:
    TEST_DATA.append((entry[0], {'entities': entry[1]}))

print("TEST DATA")
for el in TEST_DATA:
    print(el)

gdth_test = []
for _, annotations in TEST_DATA:
    gdth_test_indiv = []
    for ent in annotations.get('entities'):
        gdth_test_indiv.append((ent[0], ent[1], ent[2]))
    gdth_test.append(gdth_test_indiv)

print("gdth_test")
for el in gdth_test:
    print(el)

a = 1
total = 0
match = 0
mislabeled_entites = 0
total_number_of_entities = 0
import datetime

a = datetime.datetime.now()
for run_test, label_test, gdth in zip(raw, label, gdth_test):
    total += 1
    run_test_str = " ".join(run_test)
    a += 1
    print(f"\n{a}: Testing with: -{run_test_str}-")
    doc = prdnlp(run_test_str)
    #print(len(doc.ents))
    for ent, lbl in zip(doc.ents, gdth):
        total_number_of_entities += 1
        print(lbl)
        print("gdth:", lbl[0], "pred:", ent.start_char)
        print("gdth:", lbl[1], "pred:", ent.end_char)
        print("gdth:", lbl[2], "pred:", ent.label_)
        if lbl[0] == ent.start_char and lbl[1] == ent.end_char:
            if lbl[2] == ent.label_:
                match += 1
            else:
                mislabeled_entites += 1
        print(f"-{ent.text}-", ent.start_char, ent.end_char, ent.label_)
b = datetime.datetime.now()
delta = b - a
print(f"CRF time elapsed: ", int(delta.total_seconds() * 1000))

print("Matches: ", match)
print("total_number_of_entities: ", total_number_of_entities)
print("Accuracy: ", match/total_number_of_entities)
print("Mislabeled entites: ", mislabeled_entites)
        