# Environment declaration and verification

`requirements.txt` was copied from the recovered workflow. Its pinned versions
are retained for traceability; they are not a validated lockfile and were not
installed or tested as a complete environment during this packaging operation.
The recovered scripts do not import TensorFlow and TensorFlow is not a declared
dependency. Do not copy an unrelated Python/TensorFlow/GPU configuration into the
paper as evidence of the environment used for these analyses.

Record the actual versions in the successful private Kaggle run:

```bash
python tools/capture_environment.py --out /kaggle/working/chapter2_private_outputs/run_environment.json
```

The command records Python, operating system and the installed versions of the
declared analysis packages. It does not inspect research input files. A version
record alone does not verify the results. Keep it with the run logs and compare
the final outputs with the manuscript before freezing a public release tag.
