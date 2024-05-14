from prosst.structure.quantizer import PdbQuantizer
processor = PdbQuantizer()
result = processor("example_data/p1.pdb", return_residue_seq=True)
print(result)