"""
This script encodes HIV nucleotide sequences using One Hot Encoding and k-mer encoding
"""

import csv
import itertools
import re

import numpy as np

DNA_ALPHABET = ['a', 'c', 'g', 't']


def ordinal_encode(seq, max_len):
    """
    Encode a DNA sequence using ordinal encoding by right padding the sequences with 0's
    :param seq: the sequence
    :param max_len: the longest sequence in the dataset
    :return: numpy array representing the encoded sequence
    """

    mapping = {'-': 0.0, 'a': 0.25, 't': 0.50, 'c': 0.75, 'g': 1.0}
    encoded_seq = np.zeros(max_len, dtype=float)
    encoded_seq[:len(seq)] = [mapping[i] for i in seq]

    return encoded_seq


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


def get_nucleotide_frequencies(seqs):

    all_freqs = []

    for seq in seqs:
        seq_freq = []
        pos = {'a': [], 'c': [], 'g': [], 't': []}

        for nt in DNA_ALPHABET:
            nt_locs = []
            for match in re.finditer(nt, seq):
                nt_locs.append(match.start())
            pos[nt] = nt_locs

            seq_freq.append(len(nt_locs))

        all_freqs.append((seq_freq, pos))

    return all_freqs


def get_pos_in_seq(nt, seq):
    """
    Finds the position of a given nucleotide in the sequence
    :param nt: the nucleotide of interest
    :param seq: the sequence
    :return: a tuple where the first element is np array of positions (1-based indexing)
            and the second element is the average position of the given nucleotide
    """
    nt_locs = []
    for match in re.finditer(nt, seq):
        nt_locs.append(match.start())

    nt_locs = np.array(nt_locs)
    nt_locs += 1

    nt_mu = (nt_locs / len(nt_locs)).sum()

    return nt_locs, nt_mu


def natural_vector_encode(seq):
    """
    Embed the DNA sequence as a 12-dimensional natural vector, all values are normalized by the sequence length
        - Frequencies for each nucleotide (freq_A, freq_C, freq_G, freq_T)
        - Average position of each nucleotide (mu_A, mu_C, mu_G, mu_T)
        - Central moment position of the nucleotide (d_A, d_C, d_T, d_G)
    :param seq: the sequence to encode
    :return: a 12-dimensional vector that represents the nucleotide distribution
    """
    freq = []   # [a, c, g, t]
    mu = []
    pos = {'a': [], 'c': [], 'g': [], 't': []}
    norm_moments = []
    seq_len = len(seq)

    # Find position from the origin for each nucleotide in the sequence
    for i, nt in enumerate(DNA_ALPHABET):
        nt_locs = []
        for match in re.finditer(nt, seq):
            nt_locs.append(match.start())
        pos[nt] = nt_locs

        # Use 1-based indexing
        nt_locs_np = np.array(nt_locs)
        nt_locs_np += 1

        # Mu represents the average position for each nucleotide
        freq.append(len(nt_locs))
        nt_mu = (nt_locs_np / len(nt_locs_np)).sum()
        mu.append(nt_mu)

        # Calculate the central moment position of the nucleotide (second central moment)
        d_nt = (((nt_locs_np - nt_mu) ** 2) / (freq[i] * seq_len)).sum()
        norm_moments.append(d_nt)

    nat_vec = freq + mu + norm_moments
    nat_vec_np = np.array(nat_vec) / seq_len

    return nat_vec_np


def natural_vector_covariance_encode(seq):
    """
    Embed the DNA sequence as an 18-dimensional vector, all values are normalize dby the sequence length
        - Frequencies for each nucleotide (freq_A, freq_C, freq_G, freq_T)
        - Average position of each nucleotide (mu_A, mu_C, mu_G, mu_T)
        - Central moment position of the nucleotide (d_A, d_C, d_T, d_G)
        - Covariance of each nucleotide pair (AC, AT, AG, CT, CG, TG)
    :param seq: the sequence to encode
    :return: an 18-dimensional vector that represents the nucleotide distribution and covariance of the sequence
    """

    seq_len = len(seq)
    covs = []

    all_nt_pairs = itertools.combinations(DNA_ALPHABET, 2)
    for nt_1, nt_2 in all_nt_pairs:

        # Get position information for each nucleotide pair
        nt_1_pos, nt_1_mu = get_pos_in_seq(nt_1, seq)
        num_nt_1 = len(nt_1_pos)
        nt_2_pos, nt_2_mu = get_pos_in_seq(nt_2, seq)
        num_nt_2 = len(nt_2_pos)
        total_locs = np.concatenate((nt_1_pos, nt_2_pos), axis=None)

        # Compute the covariance
        cov = (((total_locs - nt_1_mu) * (total_locs - nt_2_mu))
               / (seq_len * np.sqrt(num_nt_1) * np.sqrt(num_nt_2))).sum()
        covs.append(cov)

    nat_vec = natural_vector_encode(seq)
    covs_np = np.array(covs) / seq_len

    return np.concatenate((nat_vec, covs_np), axis=None)


def subseq_nat_vec_encode(seq, num_subseq):
    """
    Embed the sequence as a 12 * num_subseq dimensional vector.
    Divides the sequence into segments and computes the natural vector for each segment.
    This gives a localized representation of nucleotide frequencies and positions
    :param seq: the input sequence
    :param num_subseq: the number of subsequences
    :return: a 12 * num_subseq dimensional vector
    """

    # Compute number of subsequences, ensuring the lengths are normalized
    q = int(np.floor(len(seq) / num_subseq))
    r = int(len(seq) - num_subseq * q)

    subseq_nat_vec = []

    # The first r sequences are q+1 in length
    for i in range(0, (r*(q+1)), q+1):
        sub_seq = seq[i: i+(q+1)]
        ss_nat_vec = natural_vector_encode(sub_seq)
        subseq_nat_vec.append(ss_nat_vec)

    # The remaining l-r sequences are q-1 in length
    for i in range(r*(q+1), len(seq), q):
        sub_seq = seq[i: i+(q-1)+1]
        ss_nat_vec = natural_vector_encode(sub_seq)
        subseq_nat_vec.append(ss_nat_vec)

    return np.ravel(np.array(subseq_nat_vec))


if __name__ == '__main__':

    seqs = []
    with open('./data/hiv.txt') as seq_handle:
        seqs = [seq.rstrip() for seq in seq_handle]

    # Encode each sequence as a subsequence natural vector
    num_samples = len(seqs)
    num_subseq = np.floor(num_samples / (12 * np.log(num_samples)))
    with open('subseq_nat_vec.csv', 'w+') as outfile:
        writer = csv.writer(outfile)
        for seq in seqs:
            sub_seq_nat_vec = subseq_nat_vec_encode(seq, num_subseq)
            writer.writerow(sub_seq_nat_vec)

    # Encode each sequence as a natural vector
    with open('natural_vector.csv', 'w+') as outfile:
        writer = csv.writer(outfile)
        for seq in seqs:
            nat_vec = natural_vector_encode(seq)
            writer.writerow(nat_vec)

    # Encode each sequence as a natural vector + covariance
    with open('natural_vector_cov.csv', 'w+') as outfile:
        writer = csv.writer(outfile)
        for seq in seqs:
            nat_vec_cov = natural_vector_covariance_encode(seq)
            writer.writerow(nat_vec_cov)

    # Generate k-mers of size 5, 6, and 7
    five_mers = generate_dna_kmers(5)
    six_mers = generate_dna_kmers(6)
    seven_mers = generate_dna_kmers(7)

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

    # Encode each nucleotide as a float between 0 and 1
    max_len = max([len(x) for x in seqs])
    with open('./ordinal_encoding.csv', 'w+', newline='') as outfile:
        writer = csv.writer(outfile)
        for seq in seqs:
            enc_seq = ordinal_encode(seq, max_len)
            writer.writerow(enc_seq)
