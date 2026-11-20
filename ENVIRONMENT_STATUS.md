

Record the actual versions in the successful private Kaggle run:

```bash
python tools/capture_environment.py --out /kaggle/working/chapter2_private_outputs/run_environment.json
```

The command records Python, operating system and the installed versions of the
declared analysis packages. It does not inspect research input files. A version
record alone does not verify the results. Keep it with the run logs and compare
the final outputs with the manuscript before freezing a public release tag.
