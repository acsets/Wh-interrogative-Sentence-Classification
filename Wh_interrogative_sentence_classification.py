# Author: Alex Chen <alexchensets@gmail.com>
# This is a rule-based model which classifies whether an input Wh-interrogative sentence is grammatical
# Currently in development mode, working on single clause sentence

from nltk.tokenize import word_tokenize
import nltk

class InterrogativeWords:
    determiners = {'whose', 'what', 'which'}
    pronouns = {'who', 'whom', 'what', 'which'}
    pro_adverbs = {'why', 'where', 'when', 'how'}
    
    @classmethod
    def get_all_words(cls):
        return cls.determiners.union(cls.pronouns, cls.pro_adverbs)

class SentenceFactory:
    def get_sentence_obj(self, user_input_string):
        return Sentence(user_input_string)

class Sentence:
    def __init__(self, user_input_string):
        self.content = user_input_string
        self.tokens = self.to_token()
        self.pos_tags = self.get_pos_tags()

    def to_token(self):
        tokens = word_tokenize(self.content)
        return tokens

    def get_pos_tags(self):
        pos_tags = nltk.pos_tag(self.tokens)
        return pos_tags
    
    def get_pos_tags_of_simplified_tagset(self):
        pos_tags = nltk.pos_tag(self.tokens, tagset = 'universal')
        return pos_tags

    def is_grammatical(self):
        def has_verb_within():
            pos_tags_universal = self.get_pos_tags_of_simplified_tagset()
            verbs = [token for (token, tag) in pos_tags_universal if tag == 'VERB']
            return len(verbs) >= 1

        def has_exactly_one_verb_within():
            pos_tags_universal = self.get_pos_tags_of_simplified_tagset()
            verbs = [token for (token, tag) in pos_tags_universal if tag == 'VERB']
            return len(verbs) == 1
        return has_exactly_one_verb_within()

class WhInterrogativeSentence(Sentence):
    def __init__(self, user_input_string):
        super().__init__(user_input_string)

    def is_grammatical(self):
        def begin_with_interrogative_word():
            interrogative_words = InterrogativeWords.get_all_words()
            return self.tokens[0].lower() in interrogative_words
        return super().is_grammatical() and begin_with_interrogative_word()

def main():
    # user_input = input("Please input a single-clause Wh-interrogative sentence: ")
    user_input = "How are you?"
    sentence_factory = SentenceFactory()
    s = sentence_factory.get_sentence_obj(user_input)
    print(s.to_token())
    print(s.get_pos_tags_of_simplified_tagset())
    print(s.is_grammatical())

    print(InterrogativeWords.get_all_words())
main()
