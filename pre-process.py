"""
This script processes HIV sequences downloaded from the HIV Sequences Database at the Los Alamos National Laboratory.
"""

from Bio import SeqIO

DNA_ALPHABET = 'atgc'


def get_subtype_seqs(seqs):
    """
    Organize subtype sequences into a dictionary
    :param seqs: a list of FASTA sequences
    :return: a dictionary of sequences where each subtype is a key and the values is a list of sequences
    """
    count = 0
    raw_subtype_seqs = {}
    for fasta in seqs:
        subtype = fasta.id.split('.')[0]
        if subtype not in raw_subtype_seqs:
            raw_subtype_seqs[subtype] = []
        raw_subtype_seqs[subtype].append(str(fasta.seq))
        count +=1

    print('There are {} sequences before processing.'.format(count))
    print('There are {} subtypes before processing.'.format(len(raw_subtype_seqs.keys())))
    return raw_subtype_seqs


def process_sequences(subtype_seqs, threshold=18):
    """
    Process the HIV sequences by removing subtypes with counts below a given threshold.
    :param subtype_seqs: dictionary of sequences, grouped by subtypes
    :param threshold: minimum number of sequences for a subtype, as per Solis-Reyes et al., 2018
    :return: a dictionary of counts for each subtype, with low frequency subtypes removed
    """

    # Remove subtypes that have too few examples, as defined by the threshold
    for subtype in subtype_seqs.copy():

        # Check that all sequences contain only 'a', 't', 'g', or 'c'
        seq_list = subtype_seqs[subtype]
        clean_seq_list = []
        for seq in seq_list:
            if all(nt in DNA_ALPHABET for nt in seq):
                clean_seq_list.append(seq)
        subtype_seqs[subtype] = clean_seq_list    # Include only sequences that contain 'a', 't', 'g', or 'c'

        # Remove sequences with too few examples
        count = len(subtype_seqs[subtype])
        if count < threshold:
            del subtype_seqs[subtype]

    print('There are {} subtypes after processing.'.format(len(subtype_seqs)))
    return subtype_seqs


def get_subtype_counts(subtype_seqs, outfile='./subtype-counts.txt'):
    """
    Writes the counts for each subtype to a file
    :param subtype_seqs:
    :param outfile: path to the output file
                    By default, it will create a file in the working directory called 'raw-subtype-counts.txt'
    """
    with open(outfile, 'w+') as out_handle:
        for subtype in subtype_seqs:
            out_handle.write('{} {}\n'.format(subtype, len(subtype_seqs[subtype])))
    return


def write_clean_data(subtype_seqs, seq_outfile='./hiv.txt', label_outfile='./label.txt'):
    """
    Write DNA sequences to a file
    :param subtype_seqs: dictionary of sequences
    :param seq_outfile: output file for sequences
    :param label_outfile: output file for labels
    """
    with open(seq_outfile, 'w+') as seq_handle, open(label_outfile, 'w+') as label_handle:
        for subtype in subtype_seqs:
            for seq in subtype_seqs[subtype]:
                seq_handle.write('{}\n'.format(seq))
                label_handle.write('{}\n'.format(subtype))
    return


if __name__ == '__main__':

    # Read in and parse sequences
    infile = './hiv-db.fasta'
    fasta_sequences = list(SeqIO.parse(open(infile), 'fasta'))
    raw_subtype_seqs = get_subtype_seqs(fasta_sequences)

    # Remove subtypes with too few examples
    subtype_seqs = process_sequences(raw_subtype_seqs)
    get_subtype_counts(subtype_seqs)

    # Write sequences and labels to output files
    # The first line in hiv.txt corresponds to the first sequence and the label/ subtype is the first line in label.txt
    write_clean_data(subtype_seqs)
