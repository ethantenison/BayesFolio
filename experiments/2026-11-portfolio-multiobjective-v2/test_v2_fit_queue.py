"""Local queue-control tests; no GP fits or worker pools are launched."""

from __future__ import annotations

import fcntl
import json
from concurrent.futures import Future
from pathlib import Path

import pytest
import v2_fit_queue as queue


def test_global_compute_lock_prevents_any_second_pool(tmp_path, monkeypatch):
    """A policy/fit queue holding the shared lock blocks this queue before setup."""
    monkeypatch.setattr(queue, "HERE", tmp_path)
    monkeypatch.setattr(queue, "prepare_contract", lambda: pytest.fail("Contract setup passed an active global lock"))
    with (tmp_path / "compute_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            queue.run_missing()


@pytest.mark.parametrize("workers", [0, 15, -1])
def test_worker_limit_checked_before_compute_or_file_access(workers):
    """Worker settings outside the approved shared cap are rejected immediately."""
    with pytest.raises(ValueError, match="at most14"):
        queue.run_missing(workers)


def test_attempt_inventory_counts_started_retries_and_prestart_crashes_once(tmp_path, monkeypatch):
    """Admission/start records for one run count once; abandoned work also counts."""
    monkeypatch.setattr(queue, "HERE", tmp_path)
    admissions = tmp_path / "outputs/fit_admissions"
    admissions.mkdir(parents=True)
    paths = [tmp_path / "outputs/fits/example/attempt_01", tmp_path / "outputs/fits/example/attempt_02"]
    paths[0].mkdir(parents=True)
    (paths[0] / "started.json").write_text("{}\n")
    for i, path in enumerate(paths):
        (admissions / f"admission_{i}.json").write_text(json.dumps({"run": str(path)}))
    report = queue.attempt_inventory()
    assert report["started_attempts"] == 1
    assert report["admitted_or_started_attempts"] == 2
    assert report["attempts_remaining"] == 1198


def test_attempt_inventory_rejects_out_of_scope_admission(tmp_path, monkeypatch):
    """An unrelated absolute destination cannot silently consume this queue."""
    monkeypatch.setattr(queue, "HERE", tmp_path)
    path = tmp_path / "outputs/fit_admissions/outside.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run": "/private/tmp/not-this-experiment"}))
    with pytest.raises(ValueError, match="Out-of-scope"):
        queue.attempt_inventory()


def test_final_reserve_cannot_be_consumed_by_nonfinal_origin():
    """Nonfinal work stops at1104 total attempts; final work retains the96 reserve."""
    count = {"admitted_or_started_attempts": 1104}
    with pytest.raises(RuntimeError, match="reserve preserved"):
        queue._check_budget({"date": "2026-09-30"}, count)
    queue._check_budget({"date": "2026-10-30"}, count)
    with pytest.raises(RuntimeError, match="budget"):
        queue._check_budget({"date": "2026-10-30"}, {"admitted_or_started_attempts": 1200})


def test_existing_contract_is_repeat_stable_and_protected(tmp_path, monkeypatch):
    """Resume verifies the original contract and never redefines its missing jobs."""
    monkeypatch.setattr(queue, "HERE", tmp_path)
    source = tmp_path / "source.py"
    source.write_text("original\n")
    contract = {"configs": {"compact_r1_i15_exp": {"rank": 1}}, "jobs": ["same"],
                "protected_sha256": {str(source): queue.digest(source)}}
    path = tmp_path / "FIT_CONTRACT.json"
    path.write_text(json.dumps(contract, indent=2) + "\n")
    (tmp_path / "GP_VARIANTS.json").write_text(json.dumps({"specs": contract["configs"]}))
    original = path.read_bytes()
    assert queue.prepare_contract() == contract
    assert path.read_bytes() == original
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="Protected source changed"):
        queue.prepare_contract()


def test_single_thread_environment_is_set_before_numpy():
    """Every numerical backend is explicitly configured for one thread."""
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        assert queue.os.environ[name] == "1"
    source = Path(queue.__file__).read_text()
    assert source.index('os.environ[_name] = "1"') < source.index("import numpy as np")


def test_identical_retry_preserves_failure_and_counts_all_seven_attempts(tmp_path, monkeypatch):
    """A fake executor exercises six-job retry/resume bookkeeping without fits."""
    jobs = [{"component": "compact_r1_i15_exp", "date": date, "seed": seed, "index": index, "audit": False}
            for date, index in [("2026-09-30", 28), ("2026-10-30", 29)] for seed in [27, 127, 227]]
    contract = {"jobs": jobs, "configs": {"compact_r1_i15_exp": {"rank": 1}}, "protected_sha256": {}}
    (tmp_path / "FIT_CONTRACT.json").write_text(json.dumps(contract))
    monkeypatch.setattr(queue, "HERE", tmp_path)
    monkeypatch.setattr(queue, "prepare_contract", lambda: contract)
    monkeypatch.setattr(queue, "checkpoint_path", lambda job: tmp_path / "outputs/fit_checkpoints" /
                        f"{job['component']}_s{job['seed']}_{job['date']}.json")

    def local_save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

    monkeypatch.setattr(queue, "save", local_save)
    monkeypatch.setattr(queue, "status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(queue, "validate_fit", lambda *_args: None)
    monkeypatch.setattr(queue, "_export_complete_panels", lambda _contract: [])
    submitted = []

    class FakePool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 6
            assert kwargs["initializer"] is queue.initialize

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, job):
            assert function is queue.fit_job
            submitted.append(job)
            run = Path(job["run"])
            run.mkdir(parents=True)
            (run / "started.json").write_text(json.dumps(job))
            future = Future()
            if len(submitted) == 1:
                (run / "raw_partial.txt").write_text("preserve exact failure context")
                future.set_exception(RuntimeError("deliberate synthetic solver failure"))
            else:
                artifact = run / "artifact.json"
                artifact.write_text("{}")
                future.set_result({"component": job["component"], "seed": job["seed"],
                                   "paths": {"artifact": str(artifact)}})
            return future

    monkeypatch.setattr(queue, "ProcessPoolExecutor", FakePool)
    result = queue.run_missing()
    assert result["passed"]
    assert result["completed_new_fits"] == 6
    assert result["fit_attempts_used"] == 7
    failures = list((tmp_path / "outputs/fit_failures").glob("*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text())
    assert "deliberate synthetic solver failure" in failure["traceback"]
    assert any(name.endswith("raw_partial.txt") for name in failure["partial_artifact_sha256"])
    retried = [job for job in submitted if job["date"] == "2026-09-30" and job["seed"] == 27]
    assert len(retried) == 2
    assert {key: value for key, value in retried[0].items() if key != "run"} == {
        key: value for key, value in retried[1].items() if key != "run"}
    assert len(list((tmp_path / "outputs/fit_checkpoints").glob("*.json"))) == 6
