# -*- coding: utf-8 -*-

!git clone https://www.github.com/GEM-benchmark/NL-Augmenter
!cd /path/to/extracted/NL-Augmenter
!python setup.py sdist
!pip install -e .
!pip install -r requirements.txt --quiet
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
!pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_sm-3.0.0/en_coref_sm-3.0.0.tar.gz
!pip install torch==1.7.1
!pip install checklist
!pip install sacrebleu
!pip install torchtext==0.8.0
!pip install benepar
!pip install sacremoses

#some other interestings augmenters are commented, because they require too much computational power

from nlaugmenter.transformations.factive_verb_transformation import FactiveVerbTransformation
from nlaugmenter.transformations.formality_change import Formal2Casual
#from nlaugmenter.transformations.lost_in_translation import LostInTranslation
from nlaugmenter.transformations.replace_with_hyponyms_hypernyms import ReplaceHypernyms
from nlaugmenter.transformations.style_paraphraser import StyleTransferParaphraser
#from nlaugmenter.transformations.suspecting_paraphraser import SuspectingParaphraser
from nlaugmenter.transformations.synonym_substitution import SynonymSubstitution
#from nlaugmenter.transformations.yes_no_question import YesNoQuestionPerturbation
#from nlaugmenter.transformations.adjectives_antonyms_switch import SentenceAdjectivesAntonymsSwitch
#from nlaugmenter.transformations.antonyms_substitute import AntonymsSubstitute
#from nlaugmenter.transformations.auxiliary_negation_removal import SentenceAuxiliaryNegationRemoval
from nlaugmenter.transformations.back_translation import BackTranslation
#from nlaugmenter.transformations.contextual_meaning_perturbation import ContextualMeaningPerturbation
from nlaugmenter.transformations.filler_word_augmentation import FillerWordAugmentation
#from nlaugmenter.transformations.negate_strengthen import NegateStrengthen
from nlaugmenter.transformations.protaugment_diverse_paraphrase import ProtaugmentDiverseParaphrase
#from nlaugmenter.transformations.quora_trained_t5_for_qa import QuoraT5QaPairGenerator
#from nlaugmenter.transformations.sentence_additions import SentenceAdditions
from nlaugmenter.transformations.slangificator import Slangificator
#from nlaugmenter.transformations.syntactically_diverse_paraphrase import ParaphraseSowReap
from nlaugmenter.transformations.factive_verb_transformation import *

import itertools, re, time
import pprint as pp
import pandas as pd
import ast
import copy

"""<h3>Wrapping augmentation techniques that require utterance splitting</h3>"""

#replaces common nouns with other related words that are either hyponyms or hypernyms
class HyperNymAugmentation():

    def generate(self, utterance):
        tr = ReplaceHypernyms()
        result = []
        if ". " in utterance:
            for sentence in utterance.split(". "):
                aug_pool = tr.generate(sentence)
                for aug_sentence in aug_pool:
                    result.append(aug_sentence)
        else:
            aug_pool = tr.generate(utterance)
            for aug_sentence in aug_pool:
                result.append(aug_sentence)

        result = list(itertools.chain.from_iterable(result))
        result = "".join(result).replace('\\','')
        result = re.sub('\n+', ' ', result)
        return result

class SentenceAdd():
    def generate(self, utterance):
        tr = SentenceAdditions()
        result = []
        if ". " in utterance:
            for sentence in utterance.split(". "):
                    result.append(tr.generate(sentence))
        else:
            result.append(tr.generate(utterance))
        result = list(itertools.chain.from_iterable(result))
        result = "".join(result).replace('\\','')
        result = re.sub('\n+', ' ', result)
        return result

"""<h3>Instantiating the augmentation pipeline</h3>
More info about each augmenter can be found in NL-Augmenter repo.

Please note this code is also present in the Anno-MI augmentation notebook. This is meant to be here just as a standalone playground.
"""

'''This is necessary on some devices to avoid conflicts in NL-augmenter'''
cd /path/to/NL-Augmenter/
cd /path/to/NL-Augmenter/nlaugmenter/transformations/factive_verb_transformation

#Check below for information on each element. Computationally intensive ones are omitted.

start_time = time.time()
print("Instantiating augmenters, this may take a while...")
augmenters = [(FactiveVerbTransformation(),"FactiveVerb"),
              (Formal2Casual(),"Formal2Casual"),
              #(LostInTranslation(),"LostInTranslation"),
              (HyperNymAugmentation(),"Hypernym substitution"),
              (StyleTransferParaphraser(style="Basic"),"Basic style"),
              (StyleTransferParaphraser(style="Tweets"),"Tweet style"),
              (SynonymSubstitution(),"Synonym substitution"),
              (BackTranslation(),"Backtranslation"),
              (FillerWordAugmentation(),"Filler Word"),
              (ProtaugmentDiverseParaphrase(),"ProtAugment"),
              #(SentenceAdd(),"Sentence Add"),
              (Slangificator(),"Slangificator"),
              #(ParaphraseSowReap(max_outputs=4),"Sow Reap")
             ]
print(f"Augmenters instantiated ({round(time.time()-start_time,2)} seconds)")

"""<h3>Augmentation playground (testing individual elements)</h3"""

#adds noise to all types if text source (sentence, paragraph, etc.) by adding factive verbs based paraphrases
tr = FactiveVerbTransformation()
tr.generate(input_sentence)

#Transfers text style from formal to informal and informal to formal
tr = Formal2Casual()
tr.generate(input_sentence)

#longer BackTranslation (any languages supported by Helsinki-NLP OpusMT models)
tr = LostInTranslation(max_outputs=3)
tr.generate(input_sentence)

hyper_aug = HyperNymAugmentation()
tr.generate(input_sentence)

#provides a range of possible styles of writing, enabling an easy use of style transfer paraphrase models originally introduced in the paper Reformulating Unsupervised Style Transfer as Paraphrase Generation (2020)
tr = StyleTransferParaphraser(style="Tweets")
tr.generate(input_sentence)

#This paraphraser transforms a yes/no question into a tag question, which helps to add more question specific informality to the dataset.
# must check more about this
tr = SuspectingParaphraser()
tr.generate("Well, I don't think that I'm ready to cut down to seven drinks a week. That seems like a lot but I would consider cutting back to two drinks a night. I think that would be my goal.",
            "can you improve on alcohol consumption?",
            "Yes")

#adds noise to all types of text sources (sentence, paragraph, etc.) by randomly substituting words with their synonyms.
tr = SynonymSubstitution()
tr.generate(input_sentence)

tr = YesNoQuestionPerturbation()
tr.generate(input_sentence)

#Switches English adjectives in a sentence with their antonyms to generate new sentences with opposite meanings
tr = SentenceAdjectivesAntonymsSwitch()
tr.generate(input_sentence)

#Aims to substitute an even number of words with their antonyms which would increase the diversity of the given content. Its double negation mechanism does not revert original sentence semantics.
tr = AntonymsSubstitute()
tr.generate(input_sentence)

#Removes the negation of English auxiliaries to generate new sentences with opposite meanings
tr = SentenceAuxiliaryNegationRemoval()
tr.generate(input_sentence)

#translates a given English sentence into German and back to English.
tr = BackTranslation()
tr.generate(input_sentence)

#changes the meaning of the sentence while avoiding grammar, spelling and logical mistakes.
tr = ContextualMeaningPerturbation(perturbation_rate=0.5)
tr.generate(input_sentence)

# adds noise to all types of text sources (sentence, paragraph, etc.) by inserting filler words and phrases ("ehh", "urr", "perhaps", "you know") in the text.
tr = FillerWordAugmentation()
tr.generate(input_sentence)

#transfers text style from formal to informal and informal to formal.
tr = Formal2Casual()
tr.generate(input_sentence)

#This transformation method is from the paper PROTAUGMENT: Unsupervised diverse short-texts paraphrasing for intent detection meta-learning
tr = ProtaugmentDiverseParaphrase()
tr.generate(input_sentence)

#This transformation creates new QA pairs in English by generating question paraphrases from a T5 model fine-tuned on Quora Question pairs (English question-question pairs).
tr = QuoraT5QaPairGenerator()
tr.generate(context="That seems like a lot but I would consider cutting back to two drinks a night.",
            question="Well, I don't think that I'm ready to cut down to seven drinks a week.",
            answers="I think that would be my goal.")

#adds generated sentence to all types of text sources (sentence, paragraph, etc.) by passing the input text to a GPT-2 Text Generation model
tr = SentenceAdditions()
tr.generate(input_sentence)
for sentence in utterance.split(". "):
    print(tr.generate(sentence).strip("\\"))

test = SentenceAdd()
tr.generate(input_sentence

#replaces terms with slang ones
tr = Slangificator()
tr.generate(input_sentence)

#produces syntactically diverse paraphrases for a given input sentence in English
tr = ParaphraseSowReap(max_outputs=4)
tr.generate(input_sentence)