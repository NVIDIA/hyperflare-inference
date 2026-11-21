The three SGC1 packets were produced by the round-three 64-pivot CPU reference (`reference.cpp::ref_frames`) and the round-two canonical packet writer (`hostpacket.cpp::cpu_encode`). Each uses a single kind-0 frame of 4,096 bytes, with base, sub, rank, layer and page zero. For fixture index i, byte j of the original is `(j / (8 + 2*i) + i) % (2+i)` using integer division. The reference decoder reproduced every input byte. These fixtures exercise transport and storage without linking any codec into Selene.

| i | Unit | Packet bytes | Stored CRC32 | Original CRC32 |
|---|---|---:|---|---|
| 0 | page0 | 704 | 6c3ff1e0 | 476cc8d3 |
| 1 | page1 | 994 | ba0478f7 | ed7164fa |
| 2 | row | 1208 | c9292e62 | b2b29b40 |

`large-row.sgc` covers 9 MiB of zero bytes in 144 consecutive kind-0 frames of 65,536 bytes. Each uses its own base offset and sub zero. The same encoder produced 6,352 bytes (CRC 96bd12e4) and the reference decoder reproduced all 9 MiB (CRC 63214bad).

`suffix-directory.pb` is the fixed outer-directory vector for the mixed-root scanner test. Its first page is 3 and its units use the first three packets above. Its directory CRC is 500789bb. The test can run unchanged against a pre-v4 scanner.
