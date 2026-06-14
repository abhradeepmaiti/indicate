"""Batch-mode LLM transliteration via provider Batch APIs (LiteLLM).

The synchronous :class:`~indicate.llm_indic.IndicLLMTransliterator` issues one
``litellm.completion`` call per request. For transliterating the millions of unique
tokens in an electoral roll, that is slow and expensive. This module routes the work
through a provider's asynchronous **Batch API** instead (~50% cheaper), with
checkpointing and resume.

Because a batch can take up to 24h to finish, *submit* and *collect* are separate
steps; :func:`transliterate_tokens_batched` is a convenience driver that submits then
polls to completion. State is durable, so a killed process resumes from the checkpoint
rather than resubmitting finished work:

* ``checkpoint_path`` -- a JSONL file of resolved ``{"token", "translit"}`` pairs
  (append-only; the durable result map). Kept dependency-free (no pandas/pyarrow).
* ``checkpoint_path + ".batchstate.json"`` -- in-flight batch ids and the
  ``custom_id -> [tokens]`` mapping needed to align results back to tokens.

Provider support is LiteLLM's batch support: **openai, azure, vertex_ai, bedrock,
vllm** -- *not* native Anthropic. To use Claude in batch mode, go through Bedrock
(``provider="bedrock"``, model ``"anthropic.claude-sonnet-4-6"``) or add a native
Anthropic ``messages.batches`` adapter later. The default provider is ``openai``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import litellm

from .llm_indic import IndicLLMTransliterator
from .logging import get_logger

logger = get_logger()

DEFAULT_GROUP_SIZE = 25
DEFAULT_MAX_REQUESTS_PER_BATCH = 50_000
BATCH_ENDPOINT = "/v1/chat/completions"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@dataclass
class BatchJob:
    """One submitted batch (a provider batch id + the tokens it covers)."""

    batch_id: str
    input_file_id: str
    custom_id_to_tokens: dict[str, list[str]]
    status: str = "submitted"  # "submitted" | "done"
    output_file_id: str | None = None


@dataclass
class BatchState:
    """Durable record of an in-flight transliteration run."""

    provider: str
    model: str
    source_lang: str
    target_lang: str
    group_size: int
    temperature: float
    use_few_shot: bool
    jobs: list[BatchJob] = field(default_factory=list)
    submitted_at: float | None = None


def _state_path(checkpoint_path: Path) -> Path:
    return Path(str(checkpoint_path) + ".batchstate.json")


def _save_state(checkpoint_path: Path, state: BatchState) -> None:
    data = asdict(state)
    _state_path(checkpoint_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_state(checkpoint_path: Path) -> BatchState | None:
    path = _state_path(checkpoint_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = [BatchJob(**job) for job in data.get("jobs", [])]
    return BatchState(
        provider=data["provider"],
        model=data["model"],
        source_lang=data["source_lang"],
        target_lang=data["target_lang"],
        group_size=data["group_size"],
        temperature=data["temperature"],
        use_few_shot=data["use_few_shot"],
        jobs=jobs,
        submitted_at=data.get("submitted_at"),
    )


# --------------------------------------------------------------------------- #
# Resolved-pairs checkpoint (JSONL)
# --------------------------------------------------------------------------- #
def _load_resolved(checkpoint_path: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    if not checkpoint_path.exists():
        return resolved
    with open(checkpoint_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            resolved[record["token"]] = record["translit"]
    return resolved


def _append_resolved(checkpoint_path: Path, pairs: dict[str, str]) -> None:
    if not pairs:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a", encoding="utf-8") as handle:
        for token, translit in pairs.items():
            handle.write(
                json.dumps({"token": token, "translit": translit}, ensure_ascii=False)
                + "\n"
            )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _chunk(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _make_transliterator(
    source_lang: str,
    target_lang: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    temperature: float,
) -> IndicLLMTransliterator:
    return IndicLLMTransliterator(
        source_lang,
        target_lang,
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def _content_to_text(content) -> str:
    """Normalise litellm.file_content's return value to a UTF-8 string."""
    if isinstance(content, (bytes, bytearray)):
        return bytes(content).decode("utf-8")
    if isinstance(content, str):
        return content
    for attr in ("text", "content"):
        value = getattr(content, attr, None)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8")
        if isinstance(value, str):
            return value
    return str(content)


def _parse_output_jsonl(content) -> dict[str, str]:
    """Map each request's ``custom_id`` to the model's text output.

    Output lines follow the OpenAI batch schema:
    ``{"custom_id", "response": {"body": {"choices": [{"message": {"content"}}]}}, "error"}``.
    """
    results: dict[str, str] = {}
    for line in _content_to_text(content).splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        custom_id = record.get("custom_id")
        if not custom_id or record.get("error"):
            continue
        body = (record.get("response") or {}).get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            continue
        message = choices[0].get("message") or {}
        results[custom_id] = message.get("content") or ""
    return results


# --------------------------------------------------------------------------- #
# Submit
# --------------------------------------------------------------------------- #
def submit_transliteration_batches(
    tokens: list[str],
    source_lang: str,
    target_lang: str,
    *,
    checkpoint_path: str | Path,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    completion_window: str = "24h",
    use_few_shot: bool = True,
    temperature: float = 0.3,
    max_requests_per_batch: int = DEFAULT_MAX_REQUESTS_PER_BATCH,
) -> BatchState:
    """Submit unique ``tokens`` to the provider Batch API. Does not block.

    Already-resolved tokens (present in the checkpoint) and duplicates/blanks are
    skipped. Returns the persisted :class:`BatchState`.
    """
    checkpoint_path = Path(checkpoint_path)
    resolved = _load_resolved(checkpoint_path)

    seen: set[str] = set()
    todo: list[str] = []
    for token in tokens:
        if not token or token in seen or token in resolved:
            continue
        seen.add(token)
        todo.append(token)

    transliterator = _make_transliterator(
        source_lang, target_lang, provider, model, api_key, temperature
    )
    provider = provider or transliterator.provider
    model = model or transliterator.model

    state = _load_state(checkpoint_path) or BatchState(
        provider=provider,
        model=model,
        source_lang=source_lang,
        target_lang=target_lang,
        group_size=group_size,
        temperature=temperature,
        use_few_shot=use_few_shot,
    )

    if not todo:
        _save_state(checkpoint_path, state)
        return state

    examples = transliterator.generate_few_shot_examples() if use_few_shot else []

    # One request per group of tokens; custom_id maps back to the group.
    requests: list[dict] = []
    custom_id_to_tokens: dict[str, list[str]] = {}
    for index, group in enumerate(_chunk(todo, group_size)):
        custom_id = f"grp-{index}"
        custom_id_to_tokens[custom_id] = group
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": {
                    "model": model,
                    "messages": transliterator.build_group_messages(group, examples),
                    "temperature": temperature,
                    "max_tokens": transliterator.default_max_tokens_for(group),
                },
            }
        )

    for request_chunk in _chunk(requests, max_requests_per_batch):
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        try:
            for request in request_chunk:
                tmp.write(json.dumps(request, ensure_ascii=False) + "\n")
            tmp.close()
            with open(tmp.name, "rb") as handle:
                file_obj = litellm.create_file(
                    file=handle, purpose="batch", custom_llm_provider=provider
                )
            batch = litellm.create_batch(
                completion_window=completion_window,
                endpoint=BATCH_ENDPOINT,
                input_file_id=file_obj.id,
                custom_llm_provider=provider,
            )
        finally:
            os.unlink(tmp.name)

        chunk_map = {
            request["custom_id"]: custom_id_to_tokens[request["custom_id"]]
            for request in request_chunk
        }
        state.jobs.append(
            BatchJob(
                batch_id=batch.id,
                input_file_id=file_obj.id,
                custom_id_to_tokens=chunk_map,
            )
        )
        logger.info(
            "Submitted batch %s (%d requests) for %s->%s",
            batch.id,
            len(request_chunk),
            source_lang,
            target_lang,
        )

    if state.submitted_at is None:
        state.submitted_at = time.time()
    _save_state(checkpoint_path, state)
    return state


# --------------------------------------------------------------------------- #
# Collect
# --------------------------------------------------------------------------- #
def collect_transliteration_batches(
    checkpoint_path: str | Path,
    *,
    transliterator: IndicLLMTransliterator | None = None,
) -> tuple[bool, dict[str, str]]:
    """Poll in-flight batches once and append any completed results.

    Returns ``(all_done, resolved_map)``. A group whose result count does not match
    the request, or whose batch failed, is left out of ``resolved_map`` (the driver
    requeues such tokens). Safe to call repeatedly.
    """
    checkpoint_path = Path(checkpoint_path)
    state = _load_state(checkpoint_path)
    if state is None:
        return True, _load_resolved(checkpoint_path)

    if transliterator is None:
        transliterator = _make_transliterator(
            state.source_lang,
            state.target_lang,
            state.provider,
            state.model,
            None,
            state.temperature,
        )

    all_done = True
    newly_resolved: dict[str, str] = {}
    for job in state.jobs:
        if job.status == "done":
            continue
        retrieved = litellm.retrieve_batch(
            batch_id=job.batch_id, custom_llm_provider=state.provider
        )
        status = getattr(retrieved, "status", None)
        output_file_id = getattr(retrieved, "output_file_id", None)

        if status == "completed" and output_file_id:
            job.output_file_id = output_file_id
            content = litellm.file_content(
                file_id=output_file_id, custom_llm_provider=state.provider
            )
            by_custom_id = _parse_output_jsonl(content)
            for custom_id, group in job.custom_id_to_tokens.items():
                text = by_custom_id.get(custom_id)
                if text is None:
                    continue  # missing -> requeued by the driver
                # Validate the raw line count *before* parsing: _parse_batch_response
                # pads/truncates to len(group), which would silently mis-align an
                # over- or under-count. Each token yields exactly one numbered line.
                raw_lines = [ln for ln in text.splitlines() if ln.strip()]
                parsed = transliterator._parse_batch_response(text, len(group))
                if len(raw_lines) != len(group) or any(not p for p in parsed):
                    logger.warning(
                        "Batch %s group %s: %d output line(s) for %d tokens; requeueing",
                        job.batch_id,
                        custom_id,
                        len(raw_lines),
                        len(group),
                    )
                    continue
                for token, translit in zip(group, parsed, strict=True):
                    newly_resolved[token] = translit
            job.status = "done"
        elif status in ("failed", "cancelled", "expired"):
            logger.warning(
                "Batch %s ended as %s; tokens will be requeued", job.batch_id, status
            )
            job.status = "done"
        else:
            all_done = False  # validating / in_progress / finalizing

    _append_resolved(checkpoint_path, newly_resolved)
    _save_state(checkpoint_path, state)
    resolved = _load_resolved(checkpoint_path)
    return all_done, resolved


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def transliterate_tokens_batched(
    tokens: list[str],
    source_lang: str,
    target_lang: str,
    *,
    checkpoint_path: str | Path,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    completion_window: str = "24h",
    use_few_shot: bool = True,
    temperature: float = 0.3,
    poll_interval: float = 60.0,
    max_wait: float | None = None,
    requeue_passes: int = 2,
    max_requests_per_batch: int = DEFAULT_MAX_REQUESTS_PER_BATCH,
) -> dict[str, str]:
    """Submit ``tokens`` in batch mode and poll to completion; return token->translit.

    Resumable: if a batch is already in flight for ``checkpoint_path`` this skips
    submission and resumes polling. Tokens whose group output was malformed are
    requeued one-per-request (up to ``requeue_passes``). If ``max_wait`` elapses with
    batches still running, returns what is resolved so far -- rerun later to resume.
    """
    checkpoint_path = Path(checkpoint_path)
    unique_tokens = [token for token in dict.fromkeys(tokens) if token]
    start = time.time()

    transliterator = _make_transliterator(
        source_lang, target_lang, provider, model, api_key, temperature
    )

    def _poll_to_done() -> tuple[dict[str, str], bool]:
        while True:
            done, resolved = collect_transliteration_batches(
                checkpoint_path, transliterator=transliterator
            )
            if done:
                return resolved, True
            if max_wait is not None and time.time() - start > max_wait:
                return resolved, False
            time.sleep(poll_interval)

    if _load_state(checkpoint_path) is None:
        submit_transliteration_batches(
            unique_tokens,
            source_lang,
            target_lang,
            checkpoint_path=checkpoint_path,
            provider=provider,
            model=model,
            api_key=api_key,
            group_size=group_size,
            completion_window=completion_window,
            use_few_shot=use_few_shot,
            temperature=temperature,
            max_requests_per_batch=max_requests_per_batch,
        )

    resolved, done = _poll_to_done()
    if not done:
        logger.warning("max_wait exceeded; rerun to resume from checkpoint")
        return resolved
    _state_path(checkpoint_path).unlink(missing_ok=True)

    unresolved = [token for token in unique_tokens if token not in resolved]
    passes = 0
    while unresolved and passes < requeue_passes:
        passes += 1
        logger.info(
            "Requeueing %d unresolved token(s) at group_size=1 (pass %d/%d)",
            len(unresolved),
            passes,
            requeue_passes,
        )
        submit_transliteration_batches(
            unresolved,
            source_lang,
            target_lang,
            checkpoint_path=checkpoint_path,
            provider=provider,
            model=model,
            api_key=api_key,
            group_size=1,
            completion_window=completion_window,
            use_few_shot=use_few_shot,
            temperature=temperature,
            max_requests_per_batch=max_requests_per_batch,
        )
        resolved, done = _poll_to_done()
        if not done:
            logger.warning("max_wait exceeded during requeue; rerun to resume")
            return resolved
        _state_path(checkpoint_path).unlink(missing_ok=True)
        unresolved = [token for token in unique_tokens if token not in resolved]

    if unresolved:
        logger.warning(
            "%d token(s) unresolved after %d requeue pass(es): %s",
            len(unresolved),
            requeue_passes,
            unresolved[:10],
        )
    return resolved
