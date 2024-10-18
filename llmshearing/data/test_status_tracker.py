import tempfile
from unittest.mock import patch

import pytest
from status_tracker import Status


@pytest.fixture
def temp_cache_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def status(temp_cache_dir):
    return Status(cache_dir=temp_cache_dir)


def test_track(status: Status):
    status.track("test", n=0, total=100, unit="tokens", unit_scale=True)
    assert "test" in status.data
    assert status.data["test"]["n"] == 0
    assert status.data["test"]["total"] == 100
    assert status.data["test"]["unit"] == "tokens"
    assert status.data["test"]["unit_scale"] == True


def test_incr(status: Status):
    status.track("test", n=0, total=100)
    status.incr("test", 10)
    status.read_all()
    assert status.data["test"]["n"] == 10


def test_set(status: Status):
    status.track("test", n=0, total=100)
    status.set("test", 50)
    status.read_all()
    assert status.data["test"]["n"] == 50


def test_read_all(status: Status):
    status.track("test1", n=10, total=100)
    status.track("test2", n=20, total=200)
    status.incr("test1", 5)
    status.set("test2", 30)
    status.read_all()
    assert status.data["test1"]["n"] == 15
    assert status.data["test2"]["n"] == 30


@patch("status_tracker.tqdm")
def test_show_progress(mock_tqdm, status: Status):
    status.track("test", n=0, total=100)
    status.incr("test", 50)
    status.show_progress()
    mock_tqdm.assert_called_once()
    mock_tqdm.return_value.n = 50
    mock_tqdm.return_value.refresh.assert_called_once()


def test_close(status: Status):
    status.track("test", n=0, total=100)
    status.show_progress()  # This creates a progress bar
    assert len(status.progress_bars) == 1
    status.close()
    assert all(bar.disable for bar in status.progress_bars.values())


def test_multiple_processes(temp_cache_dir):
    import multiprocessing as mp

    def worker_process(source):
        worker_status = Status(cache_dir=temp_cache_dir)
        worker_status.incr(source, 10)

    status = Status(cache_dir=temp_cache_dir)
    status.track("arxiv", total=100)
    status.track("wiki", total=100)

    processes = [
        mp.Process(target=worker_process, args=("arxiv",)),
        mp.Process(target=worker_process, args=("wiki",)),
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    status.read_all()
    assert status.data["arxiv"]["n"] == 10
    assert status.data["wiki"]["n"] == 10


if __name__ == "__main__":
    pytest.main()
