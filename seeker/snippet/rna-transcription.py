#date: 2024-11-08T17:02:15Z
#url: https://api.github.com/gists/871eb941905bc61eb3ef25e8178209e6
#owner: https://api.github.com/users/Fuanyi-237

def to_rna(dna_strand):
    """Convert a DNA strand to its RNA complement.

    :param dna_strand: str - a string representing the DNA sequence.
    :return: str - the RNA complement of the DNA sequence.
    """
    complement = {
        'G': 'C',
        'C': 'G',
        'T': 'A',
        'A': 'U'
    }
    
    return ''.join(complement[nucleotide] for nucleotide in dna_strand)
