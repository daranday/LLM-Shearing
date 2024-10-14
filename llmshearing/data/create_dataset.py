import json
import os
import random

import numpy as np
from joblib import Memory
from llama_tokenizer import Tokenizer
from streaming import MDSWriter
from tqdm import tqdm

job_memory = Memory(
    location="/nvmefs1/daranhe/llm-shearing/out/joblib_cache", verbose=0
)


@job_memory.cache
def write_jsonl_list(root_dir: str) -> str:
    log = open("jsonl_list.txt", "w")
    for root, ds, fs in os.walk(root_dir):
        for f in fs:
            full_path = os.path.join(root, f)
            relative_path = full_path[len(root_dir) :].lstrip(os.sep)
            print(relative_path)
            log.write(relative_path + "\n")
    return "jsonl_list.txt"


@job_memory.cache
def tokenize_single_file(index_id, raw_dir, target_dir, seq_length=4096):
    file_name = open("jsonl_list.txt").readlines()[index_id].strip()

    target_name = os.path.join(target_dir, os.path.splitext(file_name)[0] + ".npy")
    file_name = os.path.join(raw_dir, file_name)

    print("Raw file path:", file_name)
    print("Target path:", target_name)

    target_folder = os.path.dirname(target_name)
    if not os.path.exists(target_folder):
        print("Make target folder:", target_folder)
        os.makedirs(target_folder)

    print("Load tokenizer...")
    tok = Tokenizer("tokenizer.model")  # this is faster than the huggingface tokenizer
    print("Done")

    print("Loading file...")
    lines = open(file_name).readlines()
    print("Done")

    buffer = []
    data = []
    for line in tqdm(lines):
        item = json.loads(line)
        tokens = buffer + tok.encode(item["text"], bos=True, eos=True)
        buffer = []
        for start_id in range(0, len(tokens), seq_length):
            if start_id + seq_length < len(tokens):
                data.append(tokens[start_id : start_id + seq_length])
            else:
                buffer = tokens[start_id:]
                break

    print("Stacking numpy...")
    data = np.array(np.stack(data), dtype=np.uint16)
    print("Done")

    print(f"Saving to {target_name}...")
    np.save(target_name, data)
    print("Done")


def make_dir_if_not_ex(path):
    if not os.path.exists(path):
        print("Make target folder:", path)
        os.makedirs(path)


@job_memory.cache
def sample_domain(
    index_id,
    target_dir,
    tokenized_dir,
    eval_seq=2,
    seq_length=4096,
    for_prune=0.001,
    for_ft=0.001,
):

    target_folder = target_dir

    # RedPajama sampling rate
    folders = {
        "arxiv": 0.025,
        "book": 0.045,
        "c4-rp": 0.15,
        "cc": 0.67,
        "github": 0.045,
        "stackexchange": 0.02,
        "wiki": 0.045,
    }
    files = open("jsonl_list.txt").readlines()
    folder_to_files = {f: [] for f in folders}
    for line in files:
        tname = os.path.join(tokenized_dir, os.path.splitext(line)[0] + ".npy")
        for split in folders:
            if line[: len(split)] == split:
                folder_to_files[split].append(tname)

    target_folders = [list(folders.keys())[index_id]]

    random.seed(42)
    np.random.seed(42)

    # Eval first
    print("Sampling eval data...")
    for folder in []:  # target_folders:  # TODO: Disabling eval since it's not used
        print("Split: %s" % folder)
        random.shuffle(folder_to_files[folder])
        # Use the first half of files as evaluation
        selected = folder_to_files[folder][: len(folder_to_files[folder]) // 2]
        # The left will be used for training later
        folder_to_files[folder] = folder_to_files[folder][
            len(folder_to_files[folder]) // 2 :
        ]

        # Check if the files exist
        selected_verified = []
        for fname in selected:
            if os.path.exists(fname):
                selected_verified.append(fname)
        selected = selected_verified
        if len(selected) == 0:
            import pdb

            pdb.set_trace()

        folder_eval_target = eval_seq
        num_sample_each_file = max(1, folder_eval_target // len(selected) + 1)
        print(
            "  sample from %d files, %d samples each, total %d"
            % (len(selected), num_sample_each_file, folder_eval_target)
        )
        out = MDSWriter(
            columns={"tokens": "bytes", "set": "str"},
            out=os.path.join(target_folder, "eval", folder),
            compression=None,
        )
        total = 0
        for fname in tqdm(selected):
            data = np.load(fname)
            if len(data) % 2 != 0:
                data = data[:-1]
            data = data.reshape(-1, seq_length)

            indices = np.random.choice(len(data), num_sample_each_file, replace=False)
            sampled_data = data[indices]
            for sample in sampled_data:
                out.write({"tokens": sample.tobytes(), "set": folder})
                total += 1
                if total >= folder_eval_target:
                    break
            if total >= folder_eval_target:
                print("Hit eval target")
                break
        out.finish()
        print("Total: %d" % total)

    print("Eval done.")

    # Train then
    seq_1b = 1000000000 // seq_length  # this leads to roughly 1B data

    for_prune = int(seq_1b * for_prune)  # #seq for prune
    for_ft = int(seq_1b * for_ft)  # #seq for ft

    print("Sampling pruning data...")
    for folder in target_folders:
        print("Split: %s" % folder)
        random.shuffle(folder_to_files[folder])
        # This is what was left after sampling eval
        selected = folder_to_files[folder]

        # Check if the files exist
        selected_verified = []
        for fname in selected:
            if os.path.exists(fname):
                selected_verified.append(fname)
        selected = selected_verified
        if len(selected) == 0:
            import pdb

            pdb.set_trace()

        folder_for_prune = int(for_prune * folders[folder])
        file_for_prune = max(1, folder_for_prune // len(selected) + 1)

        folder_for_ft = int(for_ft * folders[folder])
        file_for_ft = max(1, folder_for_ft // len(selected) + 1)

        print(f"In total {len(selected)} files")
        print(
            f"For prune sample {folder_for_prune} in total, {file_for_prune} for each file"
        )
        print(f"For ft sample {folder_for_ft} in total, {file_for_ft} for each file")

        make_dir_if_not_ex(os.path.join(target_folder, "for_prune"))
        make_dir_if_not_ex(os.path.join(target_folder, "for_ft"))

        prune_out = MDSWriter(
            columns={"tokens": "bytes", "set": "str"},
            out=os.path.join(target_folder, "for_prune", folder),
            compression=None,
        )
        ft_out = MDSWriter(
            columns={"tokens": "bytes", "set": "str"},
            out=os.path.join(target_folder, "for_ft", folder),
            compression=None,
        )

        total_prune = 0
        total_ft = 0

        for fname in tqdm(selected):
            data = np.load(fname)
            if len(data) % 2 != 0:
                data = data[:-1]
            data = data.reshape(-1, seq_length)

            indices = np.arange(len(data))
            np.random.shuffle(indices)
            prune_indices = indices[:file_for_prune]
            ft_indices = indices[file_for_prune : file_for_prune + file_for_ft]

            prune_data = data[prune_indices]
            for sample in prune_data:
                prune_out.write({"tokens": sample.tobytes(), "set": folder})
                total_prune += 1
                if total_prune >= folder_for_prune:
                    break
            ft_data = data[ft_indices]
            for sample in ft_data:
                ft_out.write({"tokens": sample.tobytes(), "set": folder})
                total_ft += 1
                if total_ft >= folder_for_ft:
                    break

            if total_prune >= folder_for_prune and total_ft >= folder_for_ft:
                break

        prune_out.finish()
        ft_out.finish()

    print("Done.")


if __name__ == "__main__":
    raw_dir = "sample_redpajama"
    tokenized_dir = "tokenized_sample_redpajama"
    mds_dir = "mds_sample_redpajama"

    jsonl_path = write_jsonl_list(raw_dir)

    with open(jsonl_path, "r") as file:
        for i, _ in enumerate(file):
            tokenize_single_file(i, raw_dir=raw_dir, target_dir=tokenized_dir)

    # SLURM_ARRAY_TASK_ID=$i python sample.py --target_dir mds_sample_redpajama --tokenized_dir tokenized_sample_redpajama
    for domain_idx in range(7):
        sample_domain(
            domain_idx,
            target_dir=mds_dir,
            tokenized_dir=tokenized_dir,
            for_prune=0.42,
            for_ft=50,
        )
