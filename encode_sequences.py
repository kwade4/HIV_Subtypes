"""
This script encodes HIV nucleotide sequences using One Hot Encoding and k-mer encoding
"""

import numpy as np
import re

from sklearn.preprocessing import LabelEncoder
import csv


def one_hot_encode(seqs):
    """
    Encode a DNA sequence using one-hot encoding by right padding the sequences with 0's
    :param seqs: a list of sequences
    :return: a numpy array of one-hot-encoded sequences
    """
    one_hot = []

    max_len = max([len(x) for x in seqs])

    for seq in seqs:
        mapping = dict(zip("-atgc", range(5)))
        seq2 = np.zeros(max_len, dtype=np.uint8)
        seq2[:len(seq)] = [mapping[i] for i in seq]

        encoded_seq = np.eye(5)[seq2]
        one_hot.append(encoded_seq)

    return np.array(one_hot)


def string_to_array(seq_string):
    seq_string = seq_string.lower()
    seq_string = re.sub('[^acgt]', 'n', seq_string)
    seq_string = np.array(list(seq_string))

    return seq_string


def ordinal_encode(arr_seq):
    """
    Encode a DNA sequence suing
    :param arr_seq: a numpy array of characters
    :return: a numpy array of an ordinal-encoded sequence
    """

    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(['a', 't', 'g', 'c']))

    integer_encoded = label_encoder.transform(arr_seq)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 0.25  # A
    float_encoded[float_encoded == 1] = 0.50  # C
    float_encoded[float_encoded == 2] = 0.75  # G
    float_encoded[float_encoded == 3] = 1.00  # T

    return float_encoded


def generate_dna_kmers(k):
    """
    Return a list of all possible substrings of
    length k using only characters A, C, T, and G
    """
    bases = ["a", "c", "t", "g"]

    last = bases
    current = []
    for i in range(k-1):
        for b in bases:
            for l in last:
                current.append(l+b)
        last = current
        current = []
    return last


def count_mer(mer, seq):
    """
    Counts the number of times a substring mer
    ocurrs in the sequence seq (including overlapping
    occurrences)

    sample use: count_mer("GGG", "AGGGCGGG") => 2
    """

    k = len(mer)
    count = 0
    for i in range(0, len(seq)-k+1):
        if mer == seq[i:i+k]:
            count = count + 1
    return count


def kmer_count(k, seq, d):
    """
    Return a list of the number of times each possible k-mer appears
    in seq, including overlapping occurrences.
    """

    for i in range(0, len(seq)-k+1):
        subseq = seq[i:i+k]
        v = d.get(subseq, 0)
        d[subseq] = v + 1
    return d


if __name__ == '__main__':

    five_mers = generate_dna_kmers(5)
    six_mers = generate_dna_kmers(6)
    seven_mers = generate_dna_kmers(7)

    seqs = []
    with open('./hiv.txt') as seq_handle:
        seqs = [seq.rstrip() for seq in seq_handle]

    # Encode the sequence using 5-mers, counts are normalized
    with open('./pentamer.csv', 'w+', newline='') as outfile:
        writer = csv.writer(outfile)
        for seq in seqs:
            # Create a dictionary to count kmers
            d = {}
            for mer in five_mers:
                d[mer] = 0

            d = kmer_count(5, seq, d)
            kmer_freq = np.array(list(d.values()))
            writer.writerow(kmer_freq / kmer_freq.sum())

    # Encode the sequence using 6-mers, counts are normalized
    with open('./hexamer.csv', 'w+', newline='') as outfile:
        writer = csv.writer(outfile)
        for seq in seqs:
            # Create a dictionary to count kmers
            d = {}
            for mer in six_mers:
                d[mer] = 0

            d = kmer_count(6, seq, d)
            kmer_freq = np.array(list(d.values()))
            writer.writerow(kmer_freq / kmer_freq.sum())

    # Encode the sequence using 6-mers, counts are normalized
    with open('./septamer.csv', 'w+', newline='') as outfile:
        writer = csv.writer(outfile)
        for seq in seqs:
            # Create a dictionary to count kmers
            d = {}
            for mer in seven_mers:
                d[mer] = 0

            d = kmer_count(7, seq, d)
            kmer_freq = np.array(list(d.values()))
            writer.writerow(kmer_freq / kmer_freq.sum())

    # seqs = ['atgcatcg', 'aaaaaaaa', 'ttttttt']

    # Encode each nucleotide as a float between 0 and 1
    with open('./ordinal_encoding.csv', 'w+', newline='') as outfile:
        writer = csv.writer(outfile)
        for seq in seqs:
            arr_seq = string_to_array(seq)
            enc_seq = ordinal_encode(arr_seq)
            writer.writerow(enc_seq)


