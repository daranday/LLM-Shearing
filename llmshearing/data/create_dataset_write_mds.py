import logging
import sys
import time
from pathlib import Path
from typing import List, Mapping

import fire
import numpy as np
import ray
from ray.util.actor_pool import ActorPool
from status_tracker import Status
from tqdm import tqdm

logger = logging.getLogger(__name__)

status = Status(cache_dir="/nvmefs1/daranhe/llm-shearing/out/data_preparation/status")

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


class TqdmToFile:
    def __init__(self, file=sys.stderr):
        self.file = file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        self.file.flush()


original_stdout = sys.stdout
sys.stdout = TqdmToFile(sys.stdout)


def open_dataset(arrow_path: str):
    from datasets import load_dataset

    return load_dataset("arrow", data_files=arrow_path, split="train")


def map_ds_name(ds_name: str) -> str | None:
    name = DS_NAME_MAPPING.get(ds_name, None)
    if name is None:
        logger.warning(f"Unknown dataset: {ds_name}")
    return name


def get_ds_tokens_cached(ds: str) -> int:
    return status.get(StatusType.tokens_cached(ds))


def get_ds_tokens_total(ds: str) -> int:
    return status.get(StatusType.tokens_cached(ds), type="total")


def get_ds_finished(ds: str):
    return status.get(StatusType.ds_finished(ds))


class IsFinishedException(Exception):
    pass


class StatusType:
    total_tokens_cached = "total tokens cached"
    num_datasets_processed = "datasets processed"

    @staticmethod
    def tokens_cached(ds: str):
        return f"tokens cached: {ds:<15}"

    @staticmethod
    def ds_finished(ds: str):
        return f"ds finished: {ds:<15}"

    @staticmethod
    def dataset_processed(id: int):
        return f"dataset finished: {id}"


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

    def write(self, tokens: List[int]) -> bool:
        """Writes arbitrary length tokens."""
        for seq in self.seq_breaker.process(tokens):
            if not self.write_seq(seq):
                break
        return self.writing

    def write_seq(self, seq: List[int]) -> bool:
        """Writes a sequence of tokens that satisfies the seq_length."""
        return self.write_seq_binary(np.array(seq, dtype=np.uint16))

    def write_seq_binary(self, seq: np.ndarray) -> bool:
        """Writes a sequence of tokens that satisfies the seq_length."""
        if not self.writing:
            return False

        assert self.writer is not None
        assert seq.dtype == np.uint16
        self.writer.write(
            {
                "tokens": seq.tobytes(),
                "set": self.data_source,
            }
        )
        self.seqs_so_far += 1
        if self.seqs_total and self.seqs_so_far >= self.seqs_total:
            self.finish()

        return self.writing

    def finish(self):
        if self.writer is not None:
            self.writer.finish()
            self.writer = None
            self.writing = False
            self.buffer = []


class TokensCacher:
    def __init__(self, ds: str, seq_length: int = 4096, block_length: int = 300):
        self.ds = ds
        self.cache_root: Path | None = None
        self.seq_breaker = SequenceBreaker(seq_length)
        self.seq_length = seq_length
        self.block_length = block_length
        self.buffer = np.zeros([block_length, seq_length], dtype=np.uint16)
        self.copied_seqs = 0
        self.seqs_so_far = 0
        self.blocks_so_far = 0
        self.finished = True

    def reset_cache_root(self, cache_root: str):
        self.finish()
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_root_unfinished = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.buffer = np.zeros([self.block_length, self.seq_length], dtype=np.uint16)
        self.copied_seqs = 0
        self.blocks_so_far = 0
        self.finished = False

    def cache(self, tokens: List[int]):
        if self.finished:
            return
        if get_ds_finished(self.ds):
            self.finish()
            return
        num_tokens_processed = self.do_cache(tokens)
        self.update_status(num_tokens_processed)

    def update_status(self, num_tokens_processed: int):
        status.incr(StatusType.tokens_cached(self.ds), num_tokens_processed)
        status.incr(StatusType.total_tokens_cached, num_tokens_processed)
        if get_ds_tokens_cached(self.ds) >= get_ds_tokens_total(self.ds):
            self.finish()
            status.set(StatusType.ds_finished(self.ds), 1)

    def do_cache(self, tokens: List[int]):
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
        if self.finished:
            return
        if self.copied_seqs > 0:
            self.write_block(self.buffer[: self.copied_seqs, :])
        self.finished = True


class DatasetTokensCacher:
    def __init__(
        self,
        cache_root: str,
        seq_length: int = 4096,
        block_length: int = 300,
    ):
        self.cache_root = Path(cache_root)
        self.cachers = {
            ds: TokensCacher(
                ds=ds,
                seq_length=seq_length,
                block_length=block_length,
            )
            for ds in DS_NAME_MAPPING.values()
        }
        self.seq_length = seq_length
        self.block_length = block_length

    def reset_dataset_id(self, dataset_id: int):
        for ds, cacher in self.cachers.items():
            cacher.reset_cache_root(
                str(self.cache_root / ds / f"dataset_{dataset_id:05d}")
            )

    def cache(self, tokens: List[int], ds: str):
        self.cachers[ds].cache(tokens)

    def finish(self):
        for cacher in self.cachers.values():
            cacher.finish()


class DatasetTokenizingPipeline:
    def __init__(self, arrow_root: str, output_root: str):
        self.arrow_root = Path(arrow_root)
        self.output_root = Path(output_root)
        self.arrow_dataset = ArrowsDataset(str(self.arrow_root))
        self.output_tokens_root = Path(self.output_root) / "tokens"
        self.tokenizer = TokenizerWorker()
        self.cacher = DatasetTokensCacher(
            str(self.output_tokens_root),
        )

    def tokenize_dataset(self, arrow_id: int, arrow_path: str):
        dataset = open_dataset(arrow_path)
        self.cacher.reset_dataset_id(arrow_id)
        for row in dataset:
            ds = map_ds_name(row["meta"]["redpajama_set_name"])  # type: ignore
            if ds and not self.cacher.cachers[ds].finished:
                tokens = self.tokenizer.tokenize(row["text"])  # type: ignore
                self.cacher.cache(tokens, ds)
        status.incr(StatusType.num_datasets_processed)
        status.set(StatusType.dataset_processed(arrow_id), 1, type="store")


@ray.remote
class DatasetTokenizingActor:
    def __init__(self, arrow_root: str, output_root: str):
        self.pipeline = DatasetTokenizingPipeline(arrow_root, output_root)

    def tokenize(self, arrow_id: int, arrow_path: str):
        self.pipeline.tokenize_dataset(arrow_id, arrow_path)


@ray.remote
class MDSWriterActor:
    def __init__(
        self,
        npys_root: str,
        output_root: str,
        data_source: str,
        total_seqs: int,
        skip_seqs: int = 0,
    ):
        self.npys_root = Path(npys_root)
        self.output_root = Path(output_root)
        self.data_source = data_source
        self.total_seqs = total_seqs
        self.skip_seqs = skip_seqs
        self.writer = MDSWriterWorker(
            data_source=data_source,
            total_seqs=total_seqs,
        )
        self.writer.reset(str(self.output_root))
        self.skipped_so_far = 0

    def run(self):
        for npy_file in sorted(self.npys_root.rglob("*.npy")):
            block: np.ndarray = np.load(npy_file)
            num_seqs_processed = 0
            for i in range(block.shape[0]):
                if self.skipped_so_far < self.skip_seqs:
                    self.skipped_so_far += 1
                    continue
                self.writer.write_seq_binary(block[i, :])
                num_seqs_processed += 1
            status.incr(StatusType.tokens_cached(self.data_source), num_seqs_processed * 4096)
            status.incr(StatusType.total_tokens_cached, num_seqs_processed * 4096)
            if not self.writer.writing:
                break
        status.set(StatusType.ds_finished(self.data_source), 1)


class ShearingDatasetCreator:
    def __init__(
        self,
        arrow_root="/nvmefs1/mk/datasets/cerebras___slim_pajama-627_b/default/0.0.0/2d0accdd58c5d5511943ca1f5ff0e3eb5e293543/",
        output_root="/nvmefs1/daranhe/llm-shearing/out/data_preparation",
        arrow_dataset: ArrowsDataset | None = None,
        sampling_rates: Mapping[str, float] = DS_SAMPLING_RATES,
        seq_length: int = 4096,
        block_length: int = 300,
    ):
        self.arrow_root = Path(arrow_root)
        self.output_root = Path(output_root)
        self.arrow_dataset = arrow_dataset or ArrowsDataset(str(self.arrow_root))
        self.output_tokens_root = Path(self.output_root) / "tokens"

        self.sampling_rates = sampling_rates
        self.seq_length = seq_length
        self.block_length = block_length

    def run(self, num_tokenizers: int = 1):
        actors = [
            DatasetTokenizingActor.remote(str(self.arrow_root), str(self.output_root))
            for _ in range(num_tokenizers)
        ]
        pool = ActorPool(actors)
        pool.map(
            lambda actor, args: actor.tokenize.remote(*args),
            list(enumerate(self.arrow_dataset.get_arrow_paths())),
        )

        while self.in_progress():
            time.sleep(1)

        status.close()

    def reset_job(self, billions: float):
        self.billions = billions
        self.total_tokens = int(billions * 1_000_000_000)
        self.total_seqs = self.total_tokens // self.seq_length
        self.total_seqs_per_ds = {
            ds: int(self.total_seqs * sampling_rate)
            for ds, sampling_rate in self.sampling_rates.items()
        }
        self.initialize_status()

    def initialize_status(self):
        status.reset()
        for ds in DS_NAME_MAPPING.values():
            status.track(
                StatusType.tokens_cached(ds),
                total=self.total_seqs_per_ds[ds] * 4096,
                unit="token",
                unit_scale=True,
            )
        for ds in DS_NAME_MAPPING.values():
            status.track(
                StatusType.ds_finished(ds), total=1, unit_scale=True, disable=False
            )
        status.track(
            StatusType.num_datasets_processed,
            total=len(self.arrow_dataset.arrow_datasets),
            unit="dataset",
            unit_scale=True,
        )
        status.track(
            StatusType.total_tokens_cached,
            total=self.total_tokens,
            unit="token",
            unit_scale=True,
        )

    def in_progress(self):
        status.show_progress()
        all_done = all(get_ds_finished(ds) for ds in DS_NAME_MAPPING.values())
        return not all_done

    def write_mds(self):
        jobs = [
            {"purpose": "for_prune", "billions": 0.4},
            {"purpose": "for_ft", "billions": 50.0},
        ]
        skip_seqs_per_ds = {ds: 0 for ds in DS_NAME_MAPPING.values()}
        for job in jobs:
            logging.info(f"Starting job: {job}")
            self.reset_job(job["billions"])

            logging.info(f"Creating actors")
            actors = [
                MDSWriterActor.remote(
                    str(self.output_tokens_root / ds),
                    str(self.output_root / "mds" / job["purpose"] / ds),
                    ds,
                    self.total_seqs_per_ds[ds],
                    skip_seqs_per_ds[ds],
                )
                for ds in DS_NAME_MAPPING.values()
            ]

            logging.info(f"Starting actors...")
            for actor in actors:
                actor.run.remote()

            logging.info(f"Monitoring progress...")
            while self.in_progress():
                time.sleep(1)

            status.close()

            for ds in DS_NAME_MAPPING.values():
                skip_seqs_per_ds[ds] += self.total_seqs_per_ds[ds]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(ShearingDatasetCreator)
