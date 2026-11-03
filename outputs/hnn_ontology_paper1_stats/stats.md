| factor | metric | control | mean diff [95% CI] | t(4) | p | p (Holm) | sign-flip p | d_z | wins |
|---|---|---|---|---|---|---|---|---|---|
| `v1_baseline` | retrieval.recall@5 | `reified_binary` | +0.193 [+0.121, +0.265] | 7.43 | 0.0018 | 0.0158 | 0.0625 | 3.32 | 5/5 |
| `v1_baseline` | completion.acc@1 | `reified_binary` | +0.078 [+0.046, +0.110] | 6.78 | 0.0025 | 0.0197 | 0.0625 | 3.03 | 5/5 |
| `v2a_inductive` | retrieval.recall@5 | `reified_binary` | +0.147 [+0.100, +0.194] | 8.67 | 0.0010 | 0.0098 | 0.0625 | 3.88 | 5/5 |
| `v2a_inductive` | completion.acc@1 | `reified_binary` | +0.098 [+0.058, +0.139] | 6.74 | 0.0025 | 0.0197 | 0.0625 | 3.02 | 5/5 |
| `v2b_head` | retrieval.recall@5 | `reified_binary` | +0.164 [+0.095, +0.233] | 6.57 | 0.0028 | 0.0197 | 0.0625 | 2.94 | 5/5 |
| `v2b_head` | completion.acc@1 | `reified_binary` | +0.098 [+0.079, +0.117] | 14.28 | 0.0001 | 0.0017 | 0.0625 | 6.39 | 5/5 |
| `v2c_text` | retrieval.recall@5 | `reified_binary` | +0.262 [+0.064, +0.459] | 3.67 | 0.0214 | 0.0427 | 0.0625 | 1.64 | 5/5 |
| `v2c_text` | completion.acc@1 | `reified_binary` | +0.154 [+0.127, +0.181] | 15.72 | 0.0001 | 0.0012 | 0.0625 | 7.03 | 5/5 |
| `v2d_real` | completion.mrr | `reified_binary` | +0.029 [+0.022, +0.036] | 11.97 | 0.0003 | 0.0031 | 0.0625 | 5.35 | 5/5 |
| `v2d_real` | completion.acc@10 | `reified_binary` | +0.074 [+0.065, +0.083] | 22.26 | 0.0000 | 0.0003 | 0.0625 | 9.96 | 5/5 |
| `v2_combined` | retrieval.recall@5 | `reified_binary` | +0.210 [+0.089, +0.332] | 4.80 | 0.0086 | 0.0331 | 0.0625 | 2.15 | 5/5 |
| `v2_combined` | completion.acc@1 | `reified_binary` | +0.056 [+0.003, +0.109] | 2.93 | 0.0426 | 0.0427 | 0.0625 | 1.31 | 5/5 |
| `v2e_depth_parity` | retrieval.recall@5 | `reified_binary_deep` | +0.191 [+0.101, +0.282] | 5.89 | 0.0042 | 0.0208 | 0.0625 | 2.63 | 5/5 |
| `v2e_depth_parity` | completion.acc@1 | `reified_binary_deep` | +0.102 [+0.044, +0.160] | 4.86 | 0.0083 | 0.0331 | 0.0625 | 2.17 | 5/5 |

Holm-significant at alpha=0.05: 14/14
