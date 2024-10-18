import logging
from pathlib import Path
from typing import Any, List, Mapping

import fire
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

DS_NAME_MAPPING = {
    "RedPajamaC4": "c4-rp",
    "RedPajamaCommonCrawl": "cc",
    "RedPajamaGithub": "github",
    "RedPajamaWikipedia": "wiki",
    "RedPajamaStackExchange": "stackexchange",
    "RedPajamaBook": "book",
    "RedPajamaArXiv": "arxiv",
}
DS_SAMPLING_RATES = {
    "arxiv": 0.025,
    "book": 0.045,
    "c4-rp": 0.15,
    "cc": 0.67,
    "github": 0.045,
    "stackexchange": 0.02,
    "wiki": 0.045,
}


def open_dataset(arrow_path: str):
    from datasets import load_dataset

    return load_dataset("arrow", data_files=arrow_path, split="train")


class StatusReporter:
    def __init__(self):
        self.counts_pbars = {}

    def update(
        self,
        name: str,
        n: float | None = None,
        total: int | None = None,
        unit: str = "it",
        unit_scale: bool = False,
    ):
        if name not in self.counts_pbars:
            self.counts_pbars[name] = tqdm(
                total=total, desc=name, unit=unit, unit_scale=unit_scale
            )
        if n is not None:
            original_n = self.counts_pbars[name].n
            if self.counts_pbars[name].total is not None:
                if original_n is not None:
                    if original_n >= self.counts_pbars[name].total:
                        return
                n = min(n, self.counts_pbars[name].total)
            self.counts_pbars[name].n = n
        if total is not None:
            self.counts_pbars[name].total = total
        self.counts_pbars[name].refresh()

    def finish(self):
        for pbar in self.counts_pbars.values():
            pbar.close()


reporter = StatusReporter()


class TokenizerWorker:
    def __init__(self, tokenizer_path="tokenizer.model"):
        from llama_tokenizer import Tokenizer

        self.tokenizer = Tokenizer(tokenizer_path)

    def tokenize(self, text):
        return self.tokenizer.encode(text, bos=True, eos=True)


class SequenceBreaker:
    def __init__(self, seq_length: int):
        self.seq_length = seq_length
        self.reset()

    def reset(self):
        self.buffer = []

    def process(self, tokens: List[int]):
        tokens = self.buffer + tokens
        self.buffer = []
        for start_id in range(0, len(tokens), self.seq_length):
            if start_id + self.seq_length < len(tokens):
                yield tokens[start_id : start_id + self.seq_length]
            else:
                self.buffer = tokens[start_id:]
                break


class MDSWriterWorker:
    def __init__(self, data_source: str, seq_length: int = 4096, total_seqs=0):
        from streaming import MDSWriter

        self.output_dir: Path | None = None
        self.writer: MDSWriter | None = None
        self.data_source = data_source
        self.seq_length = seq_length
        self.seq_breaker = SequenceBreaker(seq_length)
        self.seqs_so_far = 0
        self.seqs_total = total_seqs
        self.writing = False

    def reset(self, output_dir: str):
        from streaming import MDSWriter

        if self.writer is not None:
            self.finish()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = MDSWriter(
            columns={"tokens": "bytes", "set": "str"},
            out=str(self.output_dir),
            compression=None,
        )
        self.writing = True
        self.seq_breaker.reset()

    def write_seq(self, seq: List[int]):
        """Writes a sequence of tokens that satisfies the seq_length."""
        assert self.writer is not None
        self.writer.write(
            {
                "tokens": np.array(seq, dtype=np.uint16).tobytes(),
                "set": self.data_source,
            }
        )
        self.seqs_so_far += 1
        if self.seqs_total and self.seqs_so_far >= self.seqs_total:
            self.finish()
            self.writing = False

    def write(self, tokens: List[int]):
        """Writes arbitrary length tokens."""
        for seq in self.seq_breaker.process(tokens):
            self.write_seq(seq)

    def finish(self):
        if self.writer is not None:
            self.writer.finish()
            self.writer = None
            self.buffer = []


class TokensCacher:
    def __init__(self, seq_length: int = 4096, block_length: int = 300):
        self.cache_root: Path | None = None
        self.seq_breaker = SequenceBreaker(seq_length)
        self.seq_length = seq_length
        self.block_length = block_length
        self.buffer = np.zeros([block_length, seq_length], dtype=np.uint16)
        self.copied_seqs = 0
        self.seqs_so_far = 0
        self.blocks_so_far = 0

    def reset_cache_root(self, cache_root: str):
        self.finish()
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.buffer = np.zeros([self.block_length, self.seq_length], dtype=np.uint16)
        self.copied_seqs = 0
        self.blocks_so_far = 0

    def cache(self, tokens: List[int]):
        assert self.cache_root is not None, f"Must call reset_cache_root() first"
        num_tokens_processed = 0
        for seq in self.seq_breaker.process(tokens):
            self.buffer[self.copied_seqs, :] = seq
            self.copied_seqs += 1
            self.seqs_so_far += 1
            if self.copied_seqs == self.buffer.shape[0]:
                self.write_block(self.buffer)
                self.copied_seqs = 0
            num_tokens_processed += len(seq)
        return num_tokens_processed

    def write_block(self, block: np.ndarray):
        assert self.cache_root is not None, f"Must call reset_cache_root() first"
        assert block.shape[1] == self.buffer.shape[1]
        path = self.cache_root / f"{self.blocks_so_far:05d}.npy"
        if self.copied_seqs > 0:
            np.save(path, block[: self.copied_seqs, :])
            self.blocks_so_far += 1
            self.copied_seqs = 0

    def finish(self):
        if self.copied_seqs > 0:
            self.write_block(self.buffer[: self.copied_seqs, :])


class MultiTokensCacher:
    def __init__(
        self,
        cache_root: str,
        total_tokens_in_billions: float,
        seq_length: int = 4096,
        block_length: int = 300,
        sampling_rates: Mapping[str, float] = DS_SAMPLING_RATES,
    ):
        self.cache_root = Path(cache_root)
        self.cachers = {
            ds: TokensCacher(
                seq_length=seq_length,
                block_length=block_length,
            )
            for ds in DS_NAME_MAPPING.values()
        }
        self.seq_length = seq_length
        self.block_length = block_length
        self.total_tokens = int(total_tokens_in_billions * 1_000_000_000)
        self.total_seqs = self.total_tokens // seq_length
        self.sampling_rates = sampling_rates
        self.total_seqs_per_ds = {
            ds: int(self.total_seqs * sampling_rate)
            for ds, sampling_rate in sampling_rates.items()
        }
        self.tokens_so_far = 0
        self.unfinished_ds = set(DS_NAME_MAPPING.values())

    def reset_dataset_id(self, dataset_id: int):
        for ds, cacher in self.cachers.items():
            cacher.reset_cache_root(
                str(self.cache_root / ds / f"dataset_{dataset_id:05d}")
            )

    def cache(self, tokens: List[int], ds: str):
        if ds not in self.unfinished_ds:
            return
        self.tokens_so_far += self.cachers[ds].cache(tokens)
        if self.cachers[ds].seqs_so_far >= self.total_seqs_per_ds[ds]:
            self.cachers[ds].finish()
            self.unfinished_ds.remove(ds)

    def is_finished(self):
        return len(self.unfinished_ds) == 0

    def finish(self):
        for cacher in self.cachers.values():
            cacher.finish()


class ArrowsDataset:
    def __init__(self, arrow_root: str):
        self.arrow_root = Path(arrow_root)
        self.arrow_list_path = Path(self.arrow_root) / "arrow_list.txt"
        self.arrow_datasets = self.get_arrow_filenames()

    def get_arrow_filenames(self):
        files = []
        if not self.arrow_list_path.exists():
            with open(self.arrow_list_path, "w") as f:
                for p in sorted(Path(self.arrow_root).rglob("*train*.arrow")):
                    rel_path = p.relative_to(self.arrow_root)
                    f.write(f"{rel_path}\n")
                    files.append(rel_path)
        else:
            with open(self.arrow_list_path, "r") as f:
                for line in f:
                    path = line.strip()
                    if path:
                        files.append(path)
        return files

    def get_arrow_paths(self):
        for filename in self.arrow_datasets:
            yield str(self.arrow_root / filename)


class App:
    def __init__(self):
        self.arrow_root = "/nvmefs1/mk/datasets/cerebras___slim_pajama-627_b/default/0.0.0/2d0accdd58c5d5511943ca1f5ff0e3eb5e293543/"
        self.output_root = "/nvmefs1/daranhe/llm-shearing/out/data_preparation"
        self.arrow_dataset = ArrowsDataset(self.arrow_root)

    def tokenize(self, billions=50.4):
        output_tokens_root = Path(self.output_root) / "tokens"
        tokenizer = TokenizerWorker()
        cacher = MultiTokensCacher(
            str(output_tokens_root), total_tokens_in_billions=billions
        )
        for arrow_id, arrow_path in enumerate(self.arrow_dataset.get_arrow_paths()):
            if cacher.is_finished():
                break
            dataset = open_dataset(arrow_path)
            cacher.reset_dataset_id(arrow_id)
            for row_id, row in enumerate(dataset):  # type: ignore
                if cacher.is_finished():
                    break

                row: Mapping[str, Any]
                ds = DS_NAME_MAPPING.get(row["meta"]["redpajama_set_name"], None)
                if ds is None:
                    logger.warning(f"Unknown dataset: {row['meta']}")
                    continue
                if ds not in cacher.unfinished_ds:
                    continue
                tokens = tokenizer.tokenize(row["text"])
                logger.debug(
                    ds, len(row["text"]), len(tokens), len(row["text"]) / len(tokens)
                )
                cacher.cache(tokens, ds)

                for ds in DS_NAME_MAPPING.values():
                    reporter.update(
                        f"cached: {ds:<15}",
                        n=cacher.cachers[ds].seqs_so_far * 4096,
                        total=cacher.total_seqs_per_ds[ds] * 4096,
                        unit="token",
                        unit_scale=True,
                    )
                reporter.update(
                    f"num datasets processed",
                    n=arrow_id,
                    total=len(self.arrow_dataset.arrow_datasets),
                    unit="dataset",
                )
                reporter.update(
                    f"current dataset processed",
                    n=row_id,
                    total=len(dataset),
                    unit="row",
                )
                reporter.update(
                    f"tokens processed",
                    n=cacher.tokens_so_far,
                    total=cacher.total_tokens,
                    unit="token",
                    unit_scale=True,
                )
            cacher.finish()
        reporter.finish()
        print(f"Successfully cached {billions}b tokens to {output_tokens_root}")


if __name__ == "__main__":
    fire.Fire(App)
