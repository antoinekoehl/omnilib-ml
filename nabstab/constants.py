
# DESIGN POSITIONS
CDR1_NUMBERS = [28, 29, 30, 35, 36, 37, 38]
CDR1_NUMBERS_STR = [str(i) for i in CDR1_NUMBERS]
CDR2_NUMBERS = [52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66]
CDR2_NUMBERS_STR = [str(i) for i in CDR2_NUMBERS]

CDR3_NUMBERS_STR = ['105', '106', '107', '108', '109', '110', '111', '111A', '111B', '111C', '111D', '111E', '111F', '111G', '112H', '112G', '112F', '112E', '112D', '112C', '112B', '112A', '112', '113', '114', '115', '116', '117']

# Alphabet/Vocab
AAS = 'ACDEFGHIKLMNPQRSTVWY'
ALPHABET = AAS + '-'
AA2INDEX = {v:i for i,v in enumerate(ALPHABET)}
IDX2AA = {i:v for i,v in enumerate(ALPHABET)}