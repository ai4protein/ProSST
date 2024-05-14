from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from scipy.stats import spearmanr
import pandas as pd
from pathlib import Path
from Bio import SeqIO
import torch
from argparse import ArgumentParser

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_seq(fasta):
    for record in SeqIO.parse(fasta, "fasta"):
        return str(record.seq)


def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor(
        [
            shift_structure_sequence,
        ],
        dtype=torch.long,
    )


@torch.no_grad()
def score_protein(
    model,
    tokenizer,
    residue_sequence_dir: str,
    structure_sequence_dir: str,
    mutant_dir: str,
    name: str,
    model_name: str,
):
    print(f"Scoring {name}...")
    residue_fasta = Path(residue_sequence_dir) / f"{name}.fasta"
    structure_fasta = Path(structure_sequence_dir) / f"{name}.fasta"
    mutant_file = Path(mutant_dir) / f"{name}.csv"
    sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)

    structure_sequence = [int(i) for i in structure_sequence.split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
    tokenized_results = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized_results["input_ids"].to(device)
    attention_mask = tokenized_results["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
        labels=input_ids,
    )

    logits = outputs.logits
    logits = torch.log_softmax(logits[:, 1:-1, :], dim=-1)

    df = pd.read_csv(mutant_file)
    mutants = df["mutant"].tolist()
    scores = []
    vocab = tokenizer.get_vocab()
    for mutant in mutants:
        pred_score = 0
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            score = logits[0, idx, vocab[mt]] - logits[0, idx, vocab[wt]]
            pred_score += score.item()
        scores.append(pred_score)

    df[model_name] = scores
    df.to_csv(mutant_file, index=False)
    corr = spearmanr(df["DMS_score"], df[model_name]).correlation
    print(f"{name}: {corr}")


def read_names(fasta_dir):
    files = Path(fasta_dir).glob("*.fasta")
    names = [file.stem for file in files]
    return names


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--residue_dir",
        type=str,
        required=False,
        default="example_data/residue_sequence",
        help="Directory containing FASTA files of residue sequences",
    )
    parser.add_argument(
        "--structure_dir",
        type=str,
        required=False,
        default="example_data/structure_sequence/2048",
        help="Directory containing FASTA files of structure sequences",
    )
    parser.add_argument(
        "--mutant_dir",
        type=str,
        required=False,
        default="example_data/substitutions",
        help="Directory containing CSV files with mutants",
    )
    args = parser.parse_args()

    print("Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Scoring proteins...")
    model_name = args.model_path.split("/")[-1]
    protein_names = read_names(args.residue_dir)
    print(protein_names)
    for protein_name in protein_names:
        score_protein(
            model,
            tokenizer=tokenizer,
            residue_sequence_dir=args.residue_dir,
            structure_sequence_dir=args.structure_dir,
            mutant_dir=args.mutant_dir,
            model_name=model_name,
            name=protein_name,
        )


if __name__ == "__main__":
    main()
