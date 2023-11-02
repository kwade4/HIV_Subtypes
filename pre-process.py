from Bio import SeqIO

infile = './hiv-db.fasta'
fasta_sequences = SeqIO.parse(open(infile), 'fasta')

subtype_counts = {}

# with open('./subtype-accession.txt', 'w') as outfile:
#     for fasta in fasta_sequences:
#         name, sequence = fasta.id, str(fasta.seq)
#         outfile.write('{}\n'.format(name))

for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    subtype_name = name.split('.')[0]

    if subtype_name not in subtype_counts:
        subtype_counts[subtype_name] = 0
    subtype_counts[subtype_name] += 1

# 289 total subtypes, including one unknown
clean_subtype_counts = subtype_counts.copy()
for st in subtype_counts:
    # Set arbitrary threshold for cut-off
    if subtype_counts[st] < 18:
        del clean_subtype_counts[st]

# 37 subtypes (min count=21)
print(len(clean_subtype_counts))
print('\n'.join('{} {}'.format(k, int(v)) for k, v in clean_subtype_counts.items()))
