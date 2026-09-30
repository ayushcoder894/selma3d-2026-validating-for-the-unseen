# Container resources

`scripts/build_container.sh` fills this folder; the weights are not stored in git.

```
resources/
  fold0.pt, fold1.pt        v15a, v15b  (96^3; isolated path, router input, half of the contiguous blend)
  cont/                     only in the routed (Task 1) entry
    fold0.pt, fold1.pt      B128_all1, B128_all2  (128^3 contiguous experts)
    route.json              {"blend": 0.5, "threshold": 0.45, "min_size": 4, "router_threshold": 0.45,
                             "ttt_steps": 0, "anchor": true}
```

Each packaged checkpoint is an fp16 state dict plus its settings (`norm`, `patch`, `threshold`,
`minimum_component_size`, `sparse_threshold`, `sparse_minimum_component_size`), so the container
applies the same input normalisation and patch size the network was trained with.

The trained weights are attached to the repository's GitHub release (`v15a.pt`, `v15b.pt`,
`B128_all1.pt`, `B128_all2.pt`, and the pretraining checkpoint `ssl_v2_final.pt`).
