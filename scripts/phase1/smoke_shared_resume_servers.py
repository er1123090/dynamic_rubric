#!/usr/bin/env python3
"""Isolated concurrent inference check; never starts training or writes its state."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

from dynamic_rubric.artifacts import write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    services = {
        'policy': ('http://127.0.0.1:28010', 'phase1-policy-checkpoint-45'),
        'extractor': ('http://127.0.0.1:28011', 'openai/gpt-oss-120b'),
        'judge': ('http://127.0.0.1:28002', 'Qwen/Qwen3-32B'),
    }
    schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
              'required': ['ok'], 'additionalProperties': False}
    def call(name, index):
        url, model = services[name]
        data = {'model': model, 'messages': [
            {'role': 'system', 'content': 'Return only JSON matching the supplied schema.'},
            {'role': 'user', 'content': 'Confirm readiness with ok=true.'}],
            'temperature': 0, 'top_p': 1, 'seed': 11 + index, 'max_tokens': 256,
            'chat_template_kwargs': {'enable_thinking': False},
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'phase1_launch_smoke_v1', 'schema': schema, 'strict': True}}}
        if name == 'extractor':
            data['reasoning_effort'] = 'low'
        started = time.monotonic()
        request = urllib.request.Request(url + '/v1/chat/completions',
                                         data=json.dumps(data).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
        assert result['model'] == model
        assert json.loads(result['choices'][0]['message']['content']) == {'ok': True}
        return {'service': name, 'request_index': index, 'request': data,
                'response': result, 'elapsed_seconds': time.monotonic() - started}
    with ThreadPoolExecutor(max_workers=12) as executor:
        jobs = [executor.submit(call, name, index) for name in services for index in range(4)]
        records = [job.result() for job in jobs]
    result = {'state': 'all_three_services_concurrent_smoke_passed',
              'request_count': len(records), 'training_updates': 0,
              'full_training_batch_validated': False, 'requests': records}
    write_json_atomic(args.output, result, immutable=True)
    print(json.dumps({'state': result['state'], 'request_count': len(records),
                      'max_elapsed_seconds': max(r['elapsed_seconds'] for r in records)}))


if __name__ == '__main__':
    main()
