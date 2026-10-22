# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint-owned, default-off replay of detached TP-local KPool selections.

MCORE_KPOOL_CHECKPOINT_REPLAY: off, observe, replay. Observe recomputes and checks
input contents and ordered indices; replay checks layouts/bounds and reuses only
local indices. Neither mode changes preprocessing, pooling or collective order.
Replay requires observation gates before training use. Cache limits are per process:
MCORE_KPOOL_REPLAY_MAX_BYTES defaults to 1 GiB; MCORE_KPOOL_REPLAY_MAX_ENTRIES to 64.
MCORE_KPOOL_VALIDATE=all still performs reference scoring; use first for timing only
after validation. Nested checkpoints, CUDA capture and a second backward traversal
are deliberately unsupported when enabled.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import logging
import os
import threading
import weakref

import torch

MODES = ('off', 'observe', 'replay')
_ACTIVE = ContextVar('kpool_checkpoint_replay', default=None)
_LOCK = threading.Lock()
_STATS = dict(
    original=0,
    recompute=0,
    hits=0,
    observed=0,
    bypassed=0,
    failures=0,
    live_bytes=0,
    peak_bytes=0,
    live_entries=0,
    peak_entries=0,
    completed=0)
_LOGGER = logging.getLogger(__name__)


class ReplayError(RuntimeError):
    """An enabled replay contract was violated; never silently use stale indices."""


def replay_mode():
    value = os.environ.get('MCORE_KPOOL_CHECKPOINT_REPLAY', 'off')
    if value not in MODES:
        raise ValueError(f'MCORE_KPOOL_CHECKPOINT_REPLAY must be one of {MODES}, got {value!r}')
    return value


def _limit(name, default):
    value = os.environ.get(name, str(default))
    if not value.isdecimal() or int(value) <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def validate_configuration():
    mode = replay_mode()
    if mode != 'off':
        _limit('MCORE_KPOOL_REPLAY_MAX_BYTES', 1024**3)
        _limit('MCORE_KPOOL_REPLAY_MAX_ENTRIES', 64)
    return mode


def _count(name, value=1):
    with _LOCK:
        _STATS[name] += value


def replay_stats():
    """Scalar-only process-local counters; no tensor or checkpoint references."""
    with _LOCK:
        return dict(_STATS)


def _release(allocation):
    with _LOCK:
        _STATS['live_bytes'] -= allocation[0]
        _STATS['live_entries'] -= allocation[1]
        allocation[:] = [0, 0]


class _Owner:

    def __init__(self, mode):
        self.mode = mode
        self.phase = 'original'
        self.entries = []
        self.cursor = 0
        self.closed = False
        self.failure = None
        self.caller = None
        self.allocation = [0, 0]
        self.max_bytes = _limit('MCORE_KPOOL_REPLAY_MAX_BYTES', 1024**3)
        self.max_entries = _limit('MCORE_KPOOL_REPLAY_MAX_ENTRIES', 64)
        self.finalizer = weakref.finalize(self, _release, self.allocation)

    def reserve(self, size):
        with _LOCK:
            if (_STATS['live_bytes'] + size > self.max_bytes or _STATS['live_entries'] + 1 > self.max_entries):
                raise ReplayError('KPool replay process-local cache budget exceeded')
            _STATS['live_bytes'] += size
            _STATS['live_entries'] += 1
            _STATS['peak_bytes'] = max(_STATS['peak_bytes'], _STATS['live_bytes'])
            _STATS['peak_entries'] = max(_STATS['peak_entries'], _STATS['live_entries'])
            self.allocation[0] += size
            self.allocation[1] += 1

    def close(self):
        self.entries.clear()
        self.closed = True
        self.finalizer()


def current_contract():
    owner = _ACTIVE.get()
    if owner is None:
        return ('off', None, 0)
    return (owner.mode, owner.phase, owner.cursor)


def failure_pending():
    owner = _ACTIVE.get()
    return bool(owner is not None and owner.failure)


@contextmanager
def checkpoint_region(ctx, phase):
    """Called around the entire CheckpointFunction forward/backward, including errors."""
    owner = getattr(ctx, '_kpool_replay_owner', None)
    if phase == 'original':
        mode = validate_configuration()
        if mode == 'off':
            yield
            return
        if _ACTIVE.get() is not None:
            raise ReplayError('Nested KPool replay checkpoints are unsupported')
        owner = _Owner(mode)
        ctx._kpool_replay_owner = owner
    elif phase == 'recompute':
        if owner is None:
            if replay_mode() != 'off':
                raise ReplayError('KPool replay was enabled after checkpoint forward')
            yield
            return
        if owner.closed:
            raise ReplayError('KPool replay does not support repeated backward/retain_graph')
        if _ACTIVE.get() is not None:
            raise ReplayError('Nested KPool replay checkpoints are unsupported')
    else:
        raise ValueError(f'Unknown checkpoint phase {phase!r}')
    token = None
    try:
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise ReplayError('KPool replay is unsupported during CUDA graph capture')
        if replay_mode() != owner.mode:
            raise ReplayError('KPool replay mode changed across checkpoint phases')
        owner.phase, owner.cursor = phase, 0
        token = _ACTIVE.set(owner)
        _count(phase)
        yield
        if owner.failure:
            raise ReplayError(owner.failure)
        if phase == 'recompute' and owner.cursor != len(owner.entries):
            raise ReplayError('KPool selection call count changed across checkpoint phases')
    except BaseException:
        _count('failures')
        owner.close()
        raise
    finally:
        if token is not None:
            _ACTIVE.reset(token)
        if phase == 'recompute':
            owner.close()
            _count('completed')
            stats = replay_stats()
            if stats['completed'] == 1 or stats['completed'] % 128 == 0:
                _LOGGER.info('KPool checkpoint replay mode=%s stats=%s', owner.mode, stats)


def finish_recompute():
    """Stop exposing selection state before the inner autograd backward executes."""
    owner = _ACTIVE.get()
    if owner is not None:
        if owner.failure or owner.cursor != len(owner.entries):
            raise ReplayError(owner.failure or 'KPool selection call count changed across checkpoint phases')
        _ACTIVE.set(None)


def validate_eligibility(indexer, *, loss_coeff, index_share, selection_mode):
    owner = _ACTIVE.get()
    if owner is None:
        return
    owner.caller = id(indexer)
    if loss_coeff or index_share or selection_mode != 'tp_dedup':
        raise ReplayError('KPool replay requires loss_coeff=0, no index sharing and tp_dedup')
    if any(parameter.requires_grad for parameter in indexer.parameters()):
        raise ReplayError('KPool replay requires a frozen indexer')


def eligibility_contract(indexer, **kwargs):
    try:
        validate_eligibility(indexer, **kwargs)
    except ReplayError as error:
        mark_failure(error)
        return str(error)
    return None


def communication_layout(default):
    owner = _ACTIVE.get()
    if owner is not None and owner.phase == 'recompute' and owner.cursor < len(owner.entries):
        entry = owner.entries[owner.cursor]
        if isinstance(entry, dict):
            contract = entry['descriptor'][0]
            return contract[3], contract[4]
    return default


def _descriptor(value):
    if value is None:
        return None
    return (tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))


def _fingerprint(inputs):
    result = []
    for name, value in inputs.items():
        if value is None:
            result.append((name, None))
        else:
            raw = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
            result.append((name, _descriptor(value), hashlib.sha256(raw).hexdigest()))
    return tuple(result)


def bypass_selection(reason):
    """Record non-dedup execution without caching or skipping its computations."""
    owner = _ACTIVE.get()
    if owner is None:
        return
    index = owner.cursor
    owner.cursor += 1
    entry = ('bypass', reason)
    if owner.phase == 'original':
        owner.entries.append(entry)
    elif index >= len(owner.entries) or owner.entries[index] != entry:
        raise ReplayError('KPool fallback path changed across checkpoint phases')
    _count('bypassed')


def local_selection(compute, *, contract, inputs, bounds):
    """Return local indices only. Adapter propagates ReplayError via its existing gather."""
    owner = _ACTIVE.get()
    if owner is None:
        return compute()
    index = owner.cursor
    owner.cursor += 1
    if owner.failure:
        raise ReplayError(owner.failure)
    descriptor = (contract, tuple((name, _descriptor(value)) for name, value in inputs.items()), owner.caller)
    if owner.phase == 'original':
        local = compute()
        if local.dtype != torch.int32 or local.requires_grad or local.ndim != 3:
            raise ReplayError('Expected detached int32 local KPool indices')
        size = local.numel() * local.element_size() + sum(x.numel() * x.element_size() for x in bounds)
        owner.reserve(size)
        try:
            clones = tuple(value.detach().clone() for value in bounds)
            indices = local.detach().clone()
            fingerprint = _fingerprint(inputs) if owner.mode == 'observe' else None
            owner.entries.append(dict(descriptor=descriptor, bounds=clones, indices=indices, fingerprint=fingerprint))
        except Exception as error:
            owner.failure = 'KPool cache allocation or fingerprint failed'
            raise ReplayError(owner.failure) from error
        return local
    if index >= len(owner.entries) or not isinstance(owner.entries[index], dict):
        raise ReplayError('Missing checkpoint-local KPool cache entry')
    saved = owner.entries[index]
    if descriptor != saved['descriptor']:
        raise ReplayError('KPool input layout or selection contract changed on recompute')
    if len(bounds) != len(saved['bounds']) or any(not torch.equal(a, b) for a, b in zip(bounds, saved['bounds'])):
        raise ReplayError('KPool bounds, pool bases or positions changed on recompute')
    if owner.mode == 'observe':
        local = compute()
        try:
            fingerprint = _fingerprint(inputs)
        except Exception as error:
            raise ReplayError('KPool observation fingerprint failed') from error
        if fingerprint != saved['fingerprint'] or not torch.equal(local, saved['indices']):
            raise ReplayError('KPool observation input contents or ordered indices differ')
        _count('observed')
        return local
    _count('hits')
    return saved['indices']


def mark_failure(error):
    owner = _ACTIVE.get()
    if owner is not None:
        owner.failure = str(error)
