# Stage-0 source evidence

Retrieved on 2026-08-21 from primary upstream sources.

| Asset | Immutable revision | Evidence |
| --- | --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/.gitattributes> |
| `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | <https://huggingface.co/Qwen/Qwen3-32B/tree/9216db5781bf21249d130ec9da846c4624c16137> |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | <https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e> |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | <https://huggingface.co/BAAI/bge-m3/commit/5617a9f61b028005a4858fdac845db406aefb181> |

Model and tokenizer are loaded from the same Qwen revision. `BAAI/bge-m3` is
the pinned multilingual criterion embedding adapter.

The canonical veRL repository is <https://github.com/verl-project/verl> and the
custom reward contract is documented at
<https://verl.readthedocs.io/en/latest/preparation/reward_function.html>.
`v0.5.0` is the evaluated compatibility baseline, but its peeled 40-character
Git SHA has not been captured in this environment. The simple-evals HealthBench
loader and MIT license are at
<https://github.com/openai/simple-evals/blob/main/healthbench_eval.py> and
<https://github.com/openai/simple-evals/blob/main/LICENSE>. Its full source
commit and official Consensus blob checksum also remain unresolved.

For those reasons `upstream-lock.json` remains blocked. A tag, branch, shortened
SHA, or mutable model alias is not accepted as an immutable source revision.
The OpenAI requested aliases are experiment inputs rather than source pins;
their returned identities are captured and monitored per run.
