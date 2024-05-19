# 返回table:
# table[词性][单词] = 出现次数
def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.

    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """
    # TODO: Finish this function!
    # raise NotImplementedError
    from collections import defaultdict
    d = defaultdict(dict)
    for i, tag in enumerate(sequences_A):
        d[tag][sequences_B[i]] = d[tag].get(sequences_B[i], 0) + 1
    return d


# 返回table
# table[词性] = 出现次数
def unigram_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequence list that
    counts the number of occurrences of the value in the sequences list. The sequences
    collection should be a 2-dimensional array.

    For example, if the tag NOUN appears 275558 times over all the input sequences,
    then you should return a dictionary such that your_unigram_counts[NOUN] == 275558.
    """
    # TODO: Finish this function!
    # raise NotImplementedError
    d = {}
    for seq in sequences:
        for tag in seq:
            d[tag] = d.get(tag, 0) + 1
    return d


# 返回table
# table[(某单层词性，下一个单词词性)] = 出现次数
def bigram_counts(sequences):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.

    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """

    # TODO: Finish this function!
    # raise NotImplementedError
    d = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            d[(seq[i], seq[i + 1])] = d.get((seq[i], seq[i + 1]), 0) + 1
    return d


# 返回table
# table[词性] = 在句子开头的出现次数
def starting_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the beginning of
    a sequence.

    For example, if 8093 sequences start with NOUN, then you should return a
    dictionary such that your_starting_counts[NOUN] == 8093
    """
    # TODO: Finish this function!
    # raise NotImplementedError
    d = {}
    for seq in sequences:
        d[seq[0]] = d.get(seq[0], 0) + 1
    return d


# 返回table
# table[词性] = 在句子结尾的出现次数
def ending_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the end of
    a sequence.

    For example, if 18 sequences end with DET, then you should return a
    dictionary such that ending_counts[DET] == 18
    """
    # TODO: Finish this function!
    # raise NotImplementedError
    d = {}
    for seq in sequences:
        d[seq[-1]] = d.get(seq[-1], 0) + 1
    return d
