import copy

from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

from cnt_func import pair_counts, unigram_counts, bigram_counts, starting_counts, ending_counts
from helpers import Dataset
from util import simplify_decoding, accuracy

data = Dataset("data\\tags-universal.txt", "data\\brown-universal.txt", train_test_split=0.8)
emission_counts = pair_counts(*list(zip(*list(data.training_set.stream())))[::-1])
tag_unigrams = unigram_counts(data.training_set.Y)
tag_bigrams = bigram_counts(data.training_set.Y)
tag_starts = starting_counts(data.training_set.Y)
tag_ends = ending_counts(data.training_set.Y)

basic_model = HiddenMarkovModel(name="base-hmm-tagger")

# TODO: create states with emission probability distributions P(word | tag) and add to the model
# (Hint: you may need to loop & create/add new states)
# basic_model.add_states()
states = []

dict_tag = copy.deepcopy(emission_counts)
dict_states = {}  # key:val = tag:State(tag)
for tag, d_tag in dict_tag.items():
    Ct = tag_unigrams[tag]
    for word in d_tag:
        d_tag[word] = d_tag[word] / Ct
    word_emissions = DiscreteDistribution(d_tag)
    state = State(word_emissions, name=tag)
    dict_states[tag] = state
    states.append(state)
basic_model.add_states(*states)

# TODO: add edges between states for the observed transition frequencies P(tag_i | tag_i-1)
# (Hint: you may need to loop & add transitions
# basic_model.add_transition()
N_seq = len(data.training_set.Y)
for tag in tag_starts:
    basic_model.add_transition(basic_model.start, dict_states[tag], tag_starts[tag] / N_seq)

for tag in tag_ends:
    basic_model.add_transition(dict_states[tag], basic_model.end, tag_ends[tag] / tag_unigrams[tag])

for tag1 in tag_unigrams:
    for tag2 in tag_unigrams:
        if (tag1, tag2) not in tag_bigrams:
            continue
        basic_model.add_transition(dict_states[tag1], dict_states[tag2], tag_bigrams[(tag1, tag2)] / tag_unigrams[tag1])

# NOTE: YOU SHOULD NOT NEED TO MODIFY ANYTHING BELOW THIS LINE
# finalize the model
basic_model.bake()

hmm_training_acc = accuracy(data.training_set.X, data.training_set.Y, basic_model, data)
print("training accuracy basic hmm model: {:.2f}%".format(100 * hmm_training_acc))

hmm_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, basic_model, data)
print("testing accuracy basic hmm model: {:.2f}%".format(100 * hmm_testing_acc))

for key in data.testing_set.keys[:3]:
    print("Sentence Key: {}\n".format(key))
    print("Predicted labels:\n-----------------")
    print(simplify_decoding(data.sentences[key].words, basic_model, data))
    print()
    print("Actual labels:\n--------------")
    print(data.sentences[key].tags)
    print("\n")
