import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Dict, List

import fire
import numpy as np
import ray
from joblib import Memory
from llama_tokenizer import Tokenizer
from ray.util.queue import Empty, Queue
from tqdm import tqdm

job_memory = Memory(location="/nvmefs1/daranhe/llm-shearing/out/joblib_cache")


class SkipRowSignal(Exception):
    pass


class SkipDatasetSignal(Exception):
    pass


class SkipRunSignal(Exception):
    pass


class Task:
    input_queue: Queue
    thread: Thread

    def start(self):
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.input_queue.put(None)
        self.thread.join()

    def _run(self):
        print(f"Starting {self.__class__.__name__}")
        self.run()
        print(f"Stopping {self.__class__.__name__}")

    def run(self):
        pass


@dataclass
class Status:
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    seqs_so_far_per_writer: Dict[str, int] = field(default_factory=dict)
    seqs_total_per_writer: Dict[str, int] = field(default_factory=dict)


class StatusReporter:
    def __init__(self, data_sources: List[str]):
        self.reading_rows_pbar = tqdm(total=0, desc="Op: Reading rows")
        self.tokenizing_pbar = tqdm(total=0, desc="Op: Tokenizing")
        self.counts_pbars = {
            name: tqdm(total=0, desc=name)
            for name in [
                "q_size_read_row",
                "q_put_read_row",
                "q_put_tokenize",
            ]
        }
        self.writer_pbars = {
            ds: tqdm(total=0, desc=f"Writer: {ds}") for i, ds in enumerate(data_sources)
        }

    def update(self, status: Status):
        for name, n in status.counts.items():
            pbar = self.counts_pbars[name]
            pbar.n = n
            pbar.refresh()
        for source, n_seqs in status.seqs_so_far_per_writer.items():
            pbar = self.writer_pbars[source]
            pbar.n = n_seqs
            pbar.total = status.seqs_total_per_writer[source]
            pbar.refresh()


@ray.remote
class StatusActor:
    def __init__(self):
        self.status = Status()

    def update_writer(self, source: str, n_seqs: int, total: int):
        self.status.seqs_so_far_per_writer[source] = n_seqs
        self.status.seqs_total_per_writer[source] = total

    def update(self, name: str, n: int | None = None):
        self.status.counts[name] = self.status.counts[name] + 1 if n is None else n

    def get_status(self) -> Status:
        return self.status


@ray.remote
class Arr(Task):
    def __init__(self, input_queue: Queue, output_queue: Queue, status_actor):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.status_actor = status_actor
        self.ds_name_mapping = {
            "RedPajamaC4": "c4-rp",
            "RedPajamaCommonCrawl": "cc",
            "RedPajamaGithub": "github",
            "RedPajamaWikipedia": "wiki",
            "RedPajamaStackExchange": "stackexchange",
            "RedPajamaBook": "book",
            "RedPajamaArXiv": "arxiv",
        }

    def get_ds(self, row) -> str | None:
        raw_ds = row["meta"]["redpajama_set_name"]
        return self.ds_name_mapping.get(raw_ds, None)

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is None:
                break
            arrow_path = item
            dataset = open_dataset(arrow_path)
            for row in tqdm(dataset, desc=f"Reading {arrow_path}"):
                pass
                # self.output_queue.put((row["text"], self.get_ds(row)))  # type: ignore
                # self.status_actor.update.remote(
                #     "q_size_read_row", n=self.output_queue.qsize()
                # )
                # self.status_actor.update.remote("q_put_read_row")


@ray.remote
class TokenizerActor(Task):
    def __init__(self, input_queue, output_queues, tokenizer_path="tokenizer.model"):
        self.tokenizer = Tokenizer(tokenizer_path)
        self.input_queue = input_queue
        self.output_queues = output_queues

    def tokenize(self, text):
        return self.tokenizer.encode(text, bos=True, eos=True)

    ### component behaviour

    def process(self, text, source):
        # print(f"Tokenizing {len(text)} tokens from {source}")
        tokens = self.tokenize(text)
        self.output_queues[source].put(tokens)

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is None:
                break
            text, source = item
            self.process(text, source)


@ray.remote
class MDSWriterActor(Task):
    def __init__(
        self,
        output_path,
        data_source,
        input_queue,
        status_queue,
        tokens_total,
        status_actor,
        seq_length=4096,
    ):
        self.output_dir = Path(output_path) / data_source
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer: "MDSWriter" = None  # type: ignore
        self.data_source = data_source
        self.input_queue = input_queue
        self.status_queue = status_queue
        self.status_actor = status_actor
        self.seq_length = seq_length
        self.seqs_total = int(tokens_total // seq_length)
        self.seqs_so_far = 0
        self.buffer = []

    def write(self, tokens):
        self.writer.write(
            {
                "tokens": np.array(tokens, dtype=np.uint16).tobytes(),
                "set": self.data_source,
            }
        )

    def finish(self):
        self.writer.finish()

    ### component behaviour

    def process(self, tokens):
        tokens = self.buffer + tokens
        self.buffer = []
        for start_id in range(0, len(tokens), self.seq_length):
            if start_id + self.seq_length < len(tokens):
                self.write(tokens[start_id : start_id + self.seq_length])
                self.seqs_so_far += 1
                self.status_actor.update_writer.remote(
                    self.data_source, self.seqs_so_far, self.seqs_total
                )
            else:
                self.buffer = tokens[start_id:]
                break

    def run(self):
        from streaming import MDSWriter

        try:
            self.writer = MDSWriter(
                columns={"tokens": "bytes", "set": "str"},
                out=str(self.output_dir),
                compression=None,
            )
        except Exception as e:
            print(f"MDSWriter already finished for {self.data_source}")
            self.status_queue.put(True)
            with suppress(Empty):
                self.input_queue.get_nowait()
            return
        while True:
            tokens = self.input_queue.get()
            # print(f"Writing {len(tokens)} tokens from {self.data_source}")
            self.process(tokens)
            if self.seqs_so_far >= self.seqs_total:
                self.finish()
                self.status_queue.put(True)
                with suppress(Empty):
                    self.input_queue.get_nowait()
                break


@job_memory.cache
def write_redpajama_arrow_list(arrow_root: str, output_root: str) -> str:
    output_path = Path(output_root) / "arrow_list.txt"
    with open(output_path, "w") as f:
        for p in sorted(Path(arrow_root).rglob("*train*.arrow")):
            f.write(f"{p.relative_to(arrow_root)}\n")
    print(f"Wrote output to {output_path}")
    return str(output_path)


def open_dataset(arrow_path: str):
    from datasets import load_dataset

    print(f"Opening dataset from {arrow_path}")
    return load_dataset("arrow", data_files=arrow_path, split="train")


class App:
    def __init__(
        self,
        num_arrow_readers: int = 5,
        num_tokenizers: int = 4,
        queue_capacity: int = 100,
    ):
        self.arrow_root = "/nvmefs1/mk/datasets/cerebras___slim_pajama-627_b/default/0.0.0/2d0accdd58c5d5511943ca1f5ff0e3eb5e293543/"
        self.output_root = "/nvmefs1/daranhe/llm-shearing/out/data_preparation"
        self.purposes = [
            "for_prune",
            # "for_ft",
        ]
        self.num_billion_tokens = {
            "for_prune": 0.4,
            "for_ft": 50,
        }
        self.sampling_rates = {
            "arxiv": 0.025,
            "book": 0.045,
            "c4-rp": 0.15,
            "cc": 0.67,
            "github": 0.045,
            "stackexchange": 0.02,
            "wiki": 0.045,
        }

        # Status
        self.status_actor = StatusActor.remote()
        self.status_reporter = StatusReporter(list(self.sampling_rates.keys()))

        # Queues
        self.arrows_queue = Queue(maxsize=1000)
        self.text_queue = Queue(maxsize=queue_capacity)
        self.tokens_queues = {
            ds: Queue(maxsize=queue_capacity) for ds in self.sampling_rates
        }  # a queue per data source

        # Workers
        self.arrow_readers = [
            Arr.remote(self.arrows_queue, self.text_queue, self.status_actor)
            for _ in range(num_arrow_readers)
        ]
        self.tokenizer_actors = [
            TokenizerActor.remote(self.text_queue, self.tokens_queues)
            for _ in range(num_tokenizers)
        ]
        self.writer_actors = {}
        self.writer_done_queues = {}
        self.purpose_idx = -1
        self.set_next_writers()

    def set_next_writers(self):
        print(f"Setting next writers")
        self.purpose_idx += 1
        if self.purpose_idx >= len(self.purposes):
            raise SkipRunSignal()
        self.purpose = self.purposes[self.purpose_idx]

        # Kill existing
        for queue in self.writer_done_queues.values():
            queue.shutdown()
        for ds, writer in self.writer_actors.items():
            ray.kill(writer)

        # Create new.
        self.writer_done_queues = {
            ds: Queue(maxsize=100) for ds in self.sampling_rates
        }  # a queue per data source
        self.writer_actors = {
            ds: MDSWriterActor.remote(
                str(Path(self.output_root) / "mds_dataset" / self.purpose),
                ds,
                self.tokens_queues[ds],
                self.writer_done_queues[ds],
                self.num_billion_tokens[self.purpose]
                * 1000000000
                * self.sampling_rates[ds],
                self.status_actor,
            )
            for ds in self.sampling_rates
        }
        return True

    def run(self):
        arrows_manifest_path = write_redpajama_arrow_list(
            self.arrow_root, self.output_root
        )

        actors = [
            *self.arrow_readers,
            *self.writer_actors.values(),
            *self.tokenizer_actors,
        ]
        futures = [actor.start.remote() for actor in actors]
        ray.wait(futures)

        max_datasets = self.arrows_queue.maxsize or None
        for line in open(arrows_manifest_path).readlines()[:max_datasets]:
            arrow_path = str(Path(self.arrow_root) / line.strip())
            # self.arrows_queue.put(arrow_path)
            dataset = open_dataset(arrow_path)
            list(tqdm(dataset, desc=f"Opening dataset rows"))

        while True:
            self.update_status()
            time.sleep(1)

        # for line in open(arrows_manifest_path).readlines():
        #     arrow_path = str(Path(self.arrow_root) / line.strip())
        #     dataset = open_dataset(arrow_path)
        #     for row in dataset:
        #         self.update_status()
        #         continue
        #         try:
        #             self.process_row(row)
        #         except SkipRowSignal:
        #             continue
        #         except SkipDatasetSignal:
        #             print("Warning: Skipping dataset")
        #             break
        #         except SkipRunSignal:
        #             print("Warning: Skipping run")
        #             return

    def update_status(self):
        self.status_reporter.update(ray.get(self.status_actor.get_status.remote()))
        q_sizes = [q.size() for q in self.writer_done_queues.values()]
        if all(q_size > 0 for q_size in q_sizes):
            self.set_next_writers()


class Args:
    arrow_root: str = (
        "/nvmefs1/daranhe/llm-shearing/out/data_preparation/mds_dataset/for_prune"
    )
    output_root: str = "/nvmefs1/daranhe/llm-shearing/out/data_preparation"


def tokenize(args: Args):
    arrows_manifest_path = write_redpajama_arrow_list(args.arrow_root, args.output_root)
    tokenizer = TokenizerActor()

    for line in open(arrows_manifest_path).readlines():
        arrow_path = str(Path(args.arrow_root) / line.strip())
        dataset = open_dataset(arrow_path)
        for row in dataset:
            self.update_status()
            continue
            try:
                self.process_row(row)
            except SkipRowSignal:
                continue


if __name__ == "__main__":
    fire.Fire(App)
