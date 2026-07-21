# Verified dims (deployed checkpoint, prod artifact 2026-07-17)

Source: `model.pth` state-dict shapes (`tower_dnns.*`, `tower_heads.*`) +
`attention_input_layer.skip_output_size()` + v3_q3a config.

| item | problem.py has | ground truth | verdict |
|---|---|---|---|
| iap tower | [384, 384] | (384,1322)→(384,384) | OK |
| adrev tower | [192, 192] | (192,1322)→(192,192) | OK |
| retention tower | [192, 192] | **(256,1322)→(256,256)** | **FIX → [256, 256]** |
| adrev_bce tower | [192, 192] | **(128,1322)→(128,128)** | **FIX → [128, 128]** |
| SKIP_DIM | 64 | `skip_features` unset in v3_q3a → `skip_output_size() == 0` | **FIX → 0** |
| trunk width into towers | — | 1322 | note |
| tower heads | — | gating.gate (square, tower width) then linear; outs iap 13 / adrev 4 / retention 6 / adrev_bce 7; mains iap_main (6,659), adrev_main (3,593) | note |

Do NOT edit problem.py while an optimization run is live (workers re-read it
per round). Apply the fixes after the run completes, re-run test.py, and
re-verify the winner against the corrected reference before integration.
