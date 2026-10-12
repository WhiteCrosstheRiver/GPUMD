# fix shell mode grow (GPU)

```text
python3 make_model.py
../../../../src/gpumd | tee gpumd.log
python3 check.py
```

Deposits 60 pulses of 5 Si atoms on a 6x6x4 Si(001) slab at 800 K with a sliding
frozen zone (`fix shell ... mode grow`) and the tuned skip path (`fix_skip`). `check.py`
verifies the ratchet, that frozen atoms never move, and the depth rule.
The last block runs with `fix 1` only; the log should then show a fresh `full`
classification if a grow `fix shell` follows (state dropped after that run).
