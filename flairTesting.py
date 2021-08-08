from flair.data import Sentence
from flair.models import SequenceTagger

# make a sentence
sentence = Sentence('''Do you have Danny listed in personal address book We experienced some past
issues with his email address but resolved it with new email address,
dmccarty@enron.com''')

# load the NER tagger
tagger = SequenceTagger.load('ner')

# run NER over sentence
tagger.predict(sentence)

print(sentence)
print('The following NER tags are found:')

# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)