#!/usr/bin/env python3
"""Share an untouched policy column by whole prompt, then publish full cells.

Existing Trainer columns finish unchanged. Inference A starts its prompt subset now;
Trainer scores the complement after its existing columns finish. Only the final
cache-only pass publishes ordinary 100-prompt cell manifests.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.phase1.audit_policy import load_run_contract
from dynamic_rubric.phase1.audit_scoring import AuditScoreConfig, score_pool
from dynamic_rubric.phase1.probe_adjacent_scoring import (
    endpoint_identity, load_evaluator_rubrics, load_pool_b, score_regular_adjacent,
)
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter, VLLMChatError


def split_responses(responses, assignments):
    ids = {str(row['prompt_id']) for row in responses}
    if set(assignments) != {'trainer', 'inference_a'}:
        raise ValueError('exactly two prompt owners are required')
    owners = {key: set(value) for key, value in assignments.items()}
    if any(len(owners[k]) != len(assignments[k]) for k in owners):
        raise ValueError('duplicate prompt assignment')
    if not all(owners.values()) or owners['trainer'] & owners['inference_a']:
        raise ValueError('prompt owners must be nonempty and disjoint')
    if owners['trainer'] | owners['inference_a'] != ids:
        raise ValueError('prompt assignment must cover the entire Pool B')
    return {k: [r for r in responses if str(r['prompt_id']) in v] for k, v in owners.items()}


def retry_grading(call, *, retries=3, delay=30):
    for attempt in range(retries + 1):
        try:
            return call()
        except VLLMChatError:
            if attempt == retries:
                raise
            time.sleep(delay)


def run(journal):
    spec = read_json(journal / 'assignment.json')
    command = spec['trainer_existing_command']
    restricted = read_json(Path(command[command.index('--cell-plan') + 1]))
    original = read_json(Path(spec['original_trainer_plan']))
    expected = [c for c in original['cells'] if c['policy_step'] != spec['policy_step']]
    if restricted['cells'] != expected:
        raise ValueError('existing scorer plan does not exclude exactly the shared column')
    evaluators = sorted(c['evaluator_step'] for c in original['cells']
                        if c['policy_step'] == spec['policy_step'])
    if spec['evaluator_steps'] != evaluators:
        raise ValueError('shared evaluator inventory differs from original plan')
    artifact = Path(spec['artifact_root'])
    output = Path(spec['output_root'])
    contract = load_run_contract(Path(spec['run_dir']))
    judge = read_json(contract.config_path)['models']['judge']
    responses, _ = load_pool_b(contract, artifact, spec['policy_step'])
    groups = split_responses(responses, spec['prompt_ids'])
    if len(responses) != 1600 or len({r['response_id'] for r in responses}) != 1600:
        raise ValueError('expected the existing immutable 100 x 16 Pool B')
    identities = {b: endpoint_identity(url, judge['model'], judge['revision'])
                  for b, url in spec['judge_urls'].items()}
    if identities['trainer']['version'] != identities['inference_a']['version']:
        raise ValueError('judge version mismatch')
    write_json_atomic(journal / 'judge-endpoints.json', identities, immutable=False)
    rubrics = {e: load_evaluator_rubrics(contract, artifact, e)[0]
               for e in spec['evaluator_steps']}
    config = AuditScoreConfig(domain=contract.domain, method=contract.method,
                              seed=contract.seed, judge_model=judge['model'],
                              judge_revision=judge['revision'],
                              max_output_tokens=int(judge['max_output_tokens']), concurrency=32)

    def status(backend, **values):
        write_json_atomic(journal / f'{backend}-status.json', {
            **values, 'policy_step': spec['policy_step'],
            'backend': backend, 'prompt_count': len(spec['prompt_ids'][backend]),
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }, immutable=False)

    def grade(backend):
        grader = VLLMChatAdapter(spec['judge_urls'][backend], judge['model'],
                                 journal / f'{backend}-provider-cache',
                                 timeout_seconds=600, max_retries=4,
                                 bounded_grading_whitespace=True)
        try:
            for e in spec['evaluator_steps']:
                status(backend, state='scoring', evaluator_step=e)
                cell = output / f"policy-{spec['policy_step']:06d}" / f'evaluator-{e:06d}'
                rows = retry_grading(lambda: score_pool(
                    groups[backend], rubrics[e], evaluator_checkpoint=str(e),
                    policy_checkpoint=str(spec['policy_step']), pool='probe_B',
                    config=config, grader=grader, cache_dir=cell / 'grade_cache'))
                for row in rows:
                    url = row['judge']['transport']['selected_base_url'].rstrip('/')
                    if url != spec['judge_urls'][backend].rstrip('/') + '/v1':
                        raise ValueError('saved grade contradicts immutable prompt ownership')
            status(backend, state='complete', score_count=len(groups[backend]) * len(rubrics))
        except Exception as error:
            status(backend, state='failed', error=type(error).__name__)
            raise

    # The helper never writes cell manifests and never touches existing columns.
    with ThreadPoolExecutor(max_workers=1) as executor:
        inference_a = executor.submit(grade, 'inference_a')
        with (journal / 'trainer-existing.log').open('a') as log:
            subprocess.run(spec['trainer_existing_command'], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        with (output / 'scorer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            grade('trainer')
            inference_a.result()
            # Missing/corrupt cache is a hard failure, never silently regraded
            # on a backend different from its assigned physical judge.
            with patch.object(VLLMChatAdapter, 'generate', side_effect=RuntimeError(
                    'cache-only finalization attempted inference')):
                result = score_regular_adjacent(
                    run_dir=Path(spec['run_dir']), artifact_root=artifact,
                    output_root=output, steps=[], judge_urls=[spec['judge_urls']['trainer']],
                    concurrency=32, wait_timeout_seconds=0,
                    cell_plan_path=Path(spec['original_trainer_plan']),
                    bounded_grading_whitespace=True)
            write_json_atomic(journal / 'completion.json', result, immutable=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', required=True, type=Path)
    args = parser.parse_args()
    with (args.journal / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.journal)


if __name__ == '__main__':
    main()
