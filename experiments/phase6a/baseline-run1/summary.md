# Phase 4B Moonlight run: experiments\phase6a\baseline-run1

correctness: **PASS** · A: **PASS** · B: **PASS** · C: **PASS** · D: **PASS** · E: **PASS** · F: **PASS**

Model: moonshotai/Moonlight-16B-A3B@476b36a4 · 26 MoE layers × 64 experts · 28.79 GB of routed experts (16.5 MiB each) · GPU 8.59 GB, cap 6.00 GB · non-expert weights 3.21 GB on the device (shared experts 0.90 GB) · prompts 16

## Per step (fractions of all routed expert bytes: mean / median / p95)

| Configuration | Phase | Steps | All equal | Expert checks | Experts / layer | Drive | Drive / token | H2D | Hit rate | Reads | Chunked calls | Compact MB (max) | Expert MB (max) | Peak device MB | Step ms* | I/O ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stream | prefill | 16 | 16 | 416/416 | 50.7 | 0.7926 / 0.8228 / 0.9570 | 0.0096 | 0.7922 | – | 23727 | 416/416 | 173 | 173 | 3916 | 9049 | 6899 |
| stream | decode | 128 | 128 | 3328/3328 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | – | 2808 | 0/3328 | 138 | 138 | 3400 | 1314 | 839 |
| compact | prefill | 8 | 8 | 208/208 | 51.2 | 0.8001 / 0.8312 / 0.9435 | 0.0097 | 0.7997 | – | 23954 | 0/208 | 1107 | 1107 | 4579 | 9126 | 6940 |
| compact | decode | 64 | 64 | 1664/1664 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | – | 2808 | 0/1664 | 104 | 104 | 3401 | 1312 | 839 |
| chunk-1 | prefill | 4 | 4 | 104/104 | 43.3 | 0.6769 / 0.7224 / 0.7712 | 0.0175 | 0.6765 | – | 20264 | 104/104 | 12 | 12 | 3312 | 18759 | 6605 |
| chunk-1 | decode | 32 | 32 | 832/832 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | – | 2808 | 832/832 | 12 | 12 | 3276 | 2756 | 924 |
| lru-40 | prefill | 16 | 16 | 52/416 | 50.7 | 0.7926 / 0.8228 / 0.9570 | 0.0096 | 0.7922 | 0.000 | 23727 | 416/416 | 173 | 865 | 4676 | 9278 | 6961 |
| lru-40 | decode | 128 | 128 | 3328/3328 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | 0.000 | 2808 | 0/3328 | 104 | 796 | 4159 | 1335 | 844 |
| lru-80 | prefill | 16 | 16 | 52/416 | 50.7 | 0.7926 / 0.8228 / 0.9570 | 0.0096 | 0.7922 | 0.000 | 23727 | 416/416 | 173 | 1557 | 5426 | 9323 | 6956 |
| lru-80 | decode | 128 | 128 | 3328/3328 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | 0.000 | 2808 | 0/3328 | 104 | 1488 | 4912 | 1334 | 841 |
| hotness-80 | prefill | 16 | 16 | 52/416 | 50.7 | 0.7814 / 0.8114 / 0.9459 | 0.0094 | 0.7811 | 0.014 | 23395 | 416/416 | 173 | 1557 | 5423 | 9132 | 6804 |
| hotness-80 | decode | 128 | 128 | 3328/3328 | 6.0 | 0.0796 / 0.0788 / 0.0938 | 0.0796 | 0.0796 | 0.151 | 2383 | 0/3328 | 104 | 1488 | 4908 | 1207 | 707 |
| shared-streamed | prefill | 4 | 4 | 52/104 | 43.3 | 0.6769 / 0.7224 / 0.7712 | 0.0175 | 0.6765 | – | 20264 | 104/104 | 173 | 173 | 2653 | 7906 | 5880 |
| shared-streamed | decode | 32 | 32 | 832/832 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | – | 2808 | 0/832 | 104 | 104 | 2373 | 1753 | 837 |
| all-experts | prefill | 2 | 2 | 52/52 | 64.0 | 1.0005 / 1.0005 / 1.0005 | 0.0469 | 1.0000 | – | 29952 | 52/52 | 173 | 173 | 3575 | 11178 | 8847 |
| all-experts | decode | 4 | 4 | 104/104 | 64.0 | 1.0005 / 1.0005 / 1.0005 | 1.0005 | 1.0000 | – | 29952 | 104/104 | 173 | 173 | 3740 | 11158 | 8933 |

\* Step times include the benchmark's digests and checks; see the profile for un-instrumented times.

## Host and device memory

```json
{
 "stream": {
  "before_setup": {
   "resident_bytes": 632803328,
   "peak_resident_bytes": 632803328,
   "private_bytes": 1194565632
  },
  "after_non_expert_load": {
   "resident_bytes": 1532940288,
   "peak_resident_bytes": 2208333824,
   "private_bytes": 5689843712
  },
  "peak_prefill": {
   "resident_bytes": 3282239488,
   "private_bytes": 9919475712,
   "host_commit_bytes": 4562243584
  },
  "peak_decode": {
   "resident_bytes": 3289260032,
   "private_bytes": 9917763584,
   "host_commit_bytes": 4436172800
  },
  "peak_by_configuration": {
   "stream": {
    "prefill": {
     "resident_bytes": 3282239488,
     "private_bytes": 8377372672,
     "host_commit_bytes": 4562243584,
     "device_bytes": 3915995648
    },
    "decode": {
     "resident_bytes": 3289260032,
     "private_bytes": 8601116672,
     "host_commit_bytes": 4436172800,
     "device_bytes": 3400220160
    }
   },
   "compact": {
    "prefill": {
     "resident_bytes": 2757320704,
     "private_bytes": 9740701696,
     "host_commit_bytes": 3995443200,
     "device_bytes": 4578760704
    },
    "decode": {
     "resident_bytes": 2732310528,
     "private_bytes": 9741357056,
     "host_commit_bytes": 3854241792,
     "device_bytes": 3400710144
    }
   },
   "chunk-1": {
    "prefill": {
     "resident_bytes": 2747125760,
     "private_bytes": 7370346496,
     "host_commit_bytes": 4058943488,
     "device_bytes": 3311813120
    },
    "decode": {
     "resident_bytes": 2752888832,
     "private_bytes": 7275515904,
     "host_commit_bytes": 3873935360,
     "device_bytes": 3275921408
    }
   },
   "lru-40": {
    "prefill": {
     "resident_bytes": 2769842176,
     "private_bytes": 9354002432,
     "host_commit_bytes": 3918036992,
     "device_bytes": 4676213248
    },
    "decode": {
     "resident_bytes": 2769186816,
     "private_bytes": 9352888320,
     "host_commit_bytes": 3896107008,
     "device_bytes": 4159445504
    }
   },
   "lru-80": {
    "prefill": {
     "resident_bytes": 2777956352,
     "private_bytes": 9919475712,
     "host_commit_bytes": 4150874112,
     "device_bytes": 5425945088
    },
    "decode": {
     "resident_bytes": 2775859200,
     "private_bytes": 9917763584,
     "host_commit_bytes": 3930959872,
     "device_bytes": 4911793152
    }
   },
   "hotness-80": {
    "prefill": {
     "resident_bytes": 2785013760,
     "private_bytes": 9865506816,
     "host_commit_bytes": 4115128320,
     "device_bytes": 5422799360
    },
    "decode": {
     "resident_bytes": 2784305152,
     "private_bytes": 9826918400,
     "host_commit_bytes": 3914342400,
     "device_bytes": 4907866624
    }
   },
   "shared-streamed": {
    "prefill": {
     "resident_bytes": 3057577984,
     "private_bytes": 9107726336,
     "host_commit_bytes": 4427386880,
     "device_bytes": 2653307392
    },
    "decode": {
     "resident_bytes": 3058126848,
     "private_bytes": 9108643840,
     "host_commit_bytes": 4193738752,
     "device_bytes": 2372997632
    }
   },
   "all-experts": {
    "prefill": {
     "resident_bytes": 3068379136,
     "private_bytes": 8547676160,
     "host_commit_bytes": 4430151680,
     "device_bytes": 3574940160
    },
    "decode": {
     "resident_bytes": 3078557696,
     "private_bytes": 8736714752,
     "host_commit_bytes": 4200755200,
     "device_bytes": 3740409856
    }
   }
  },
  "lifetime_peak_resident_bytes": 3289296896,
  "pinned_staging_bytes_per_backend": 134217728,
  "pinned_host_memory": {
   "active_bytes.allocated": 2953301148,
   "active_bytes.current": 536870912,
   "active_bytes.freed": 2416430236,
   "active_bytes.peak": 1073741836,
   "allocated_bytes.allocated": 1073741836,
   "allocated_bytes.current": 1073741836,
   "allocated_bytes.freed": 0,
   "allocated_bytes.peak": 1073741836
  },
  "system": {
   "start": {
    "physical_available_bytes": 26262511616,
    "system_cache_bytes": 1198981120,
    "commit_total_bytes": 22670700544
   },
   "end": {
    "physical_available_bytes": 21342396416,
    "system_cache_bytes": 2615926784,
    "commit_total_bytes": 30545522688
   }
  }
 },
 "reference": {
  "before_setup": {
   "resident_bytes": 631898112,
   "peak_resident_bytes": 631898112,
   "private_bytes": 1193820160
  },
  "after_non_expert_load": {
   "resident_bytes": 823386112,
   "peak_resident_bytes": 2832162816,
   "private_bytes": 4979322880
  },
  "peak_prefill": {
   "resident_bytes": 2220388352,
   "private_bytes": 10857095168,
   "host_commit_bytes": 3273793536
  },
  "peak_decode": {
   "resident_bytes": 2217340928,
   "private_bytes": 10850766848,
   "host_commit_bytes": 3269562368
  },
  "lifetime_peak_resident_bytes": 2946822144,
  "system": {
   "start": {
    "physical_available_bytes": 18231377920,
    "system_cache_bytes": 15414235136,
    "commit_total_bytes": 22690607104
   },
   "end": {
    "physical_available_bytes": 17078657024,
    "system_cache_bytes": 17591660544,
    "commit_total_bytes": 32374603776
   }
  }
 }
}
```

## Reference stage

- Load: 3.8 s (3.13 GB of non-expert weights).
- Streamed run: 3744 experts-layer loads, 4145.7 GB converted in 4098 s (1.01 GB/s); peak device 5.15 GB, experts at most 2.21 GB.

## Routing (reference)

- Experts per layer: decode 6.00, prefill 50.70 (of 64); by prompt length: {"16": 32.0, "32": 41.9, "64": 47.9, "128": 47.1, "256": 56.2, "512": 58.8, "768": 60.8, "1024": 60.9}
- Share of decode routings received by the most used (layer, expert) slots: {"top_10pct_slots": 0.303, "top_25pct_slots": 0.551, "top_50pct_slots": 0.834}
- Of a decode step's experts, routed at the previous step in the same layer: 0.412

## Cache replay

```json
{
 "replay_checked_against_measured": {
  "lru-40": {
   "replay_equals_measured_hits_every_step": true,
   "steps": 144
  },
  "lru-80": {
   "replay_equals_measured_hits_every_step": true,
   "steps": 144
  },
  "hotness-80": {
   "replay_equals_measured_hits_every_step": true,
   "steps": 144
  }
 },
 "capacities": {
  "lru-40": {
   "capacity_experts": 40,
   "capacity_gb": 0.69206016,
   "fraction_of_experts": 0.02403846153846154,
   "decode_hit_rate": 0.0,
   "prefill_hit_rate": 0.0
  },
  "lru-80": {
   "capacity_experts": 80,
   "capacity_gb": 1.38412032,
   "fraction_of_experts": 0.04807692307692308,
   "decode_hit_rate": 0.0,
   "prefill_hit_rate": 0.0
  },
  "lru-160": {
   "capacity_experts": 160,
   "capacity_gb": 2.76824064,
   "fraction_of_experts": 0.09615384615384616,
   "decode_hit_rate": 0.3616286057692308,
   "prefill_hit_rate": 0.0005689630648143758
  },
  "lru-320": {
   "capacity_experts": 320,
   "capacity_gb": 5.53648128,
   "fraction_of_experts": 0.19230769230769232,
   "decode_hit_rate": 0.4801432291666667,
   "prefill_hit_rate": 0.018159404485325496
  },
  "lru-640": {
   "capacity_experts": 640,
   "capacity_gb": 11.07296256,
   "fraction_of_experts": 0.38461538461538464,
   "decode_hit_rate": 0.6747546073717948,
   "prefill_hit_rate": 0.11087667725570148
  },
  "lru-960": {
   "capacity_experts": 960,
   "capacity_gb": 16.60944384,
   "fraction_of_experts": 0.5769230769230769,
   "decode_hit_rate": 0.8200620993589743,
   "prefill_hit_rate": 0.2286994452610118
  },
  "lru-1248": {
   "capacity_experts": 1248,
   "capacity_gb": 21.592276992,
   "fraction_of_experts": 0.75,
   "decode_hit_rate": 0.9263571714743589,
   "prefill_hit_rate": 0.3983215589587976
  },
  "hotness-40": {
   "capacity_experts": 40,
   "capacity_gb": 0.69206016,
   "fraction_of_experts": 0.02403846153846154,
   "decode_hit_rate": 0.06372696314102565,
   "prefill_hit_rate": 0.005215494760798444
  },
  "hotness-80": {
   "capacity_experts": 80,
   "capacity_gb": 1.38412032,
   "fraction_of_experts": 0.04807692307692308,
   "decode_hit_rate": 0.15129206730769232,
   "prefill_hit_rate": 0.013892181499217676
  },
  "hotness-160": {
   "capacity_experts": 160,
   "capacity_gb": 2.76824064,
   "fraction_of_experts": 0.09615384615384616,
   "decode_hit_rate": 0.30213341346153844,
   "prefill_hit_rate": 0.03238348110568489
  },
  "hotness-320": {
   "capacity_experts": 320,
   "capacity_gb": 5.53648128,
   "fraction_of_experts": 0.19230769230769232,
   "decode_hit_rate": 0.5001252003205128,
   "prefill_hit_rate": 0.05974112180550946
  },
  "hotness-640": {
   "capacity_experts": 640,
   "capacity_gb": 11.07296256,
   "fraction_of_experts": 0.38461538461538464,
   "decode_hit_rate": 0.6883513621794872,
   "prefill_hit_rate": 0.15046702384903513
  },
  "hotness-960": {
   "capacity_experts": 960,
   "capacity_gb": 16.60944384,
   "fraction_of_experts": 0.5769230769230769,
   "decode_hit_rate": 0.8304286858974359,
   "prefill_hit_rate": 0.26843203262054904
  },
  "hotness-1248": {
   "capacity_experts": 1248,
   "capacity_gb": 21.592276992,
   "fraction_of_experts": 0.75,
   "decode_hit_rate": 0.9335186298076923,
   "prefill_hit_rate": 0.44272438480868614
  }
 }
}
```

## Gates

```json
{
  "correctness": {
    "hard_failures": 0,
    "reference_steps": 144,
    "reference_steps_expected": 144,
    "source_files_verified_against_hub_sha256": true,
    "source_files_verified": 27,
    "index_rows_audited": 3328,
    "index_rows_expected": 3328,
    "index_rows_differing": 0,
    "residency_checked_steps": 18,
    "residency_check_all_equal": true,
    "steps_all_equal": {
      "stream": 144,
      "compact": 72,
      "chunk-1": 36,
      "lru-40": 144,
      "lru-80": 144,
      "hotness-80": 144,
      "shared-streamed": 36,
      "all-experts": 6
    },
    "steps_expected": {
      "stream": 144,
      "compact": 72,
      "chunk-1": 36,
      "lru-40": 144,
      "lru-80": 144,
      "hotness-80": 144,
      "shared-streamed": 36,
      "all-experts": 6
    },
    "quantities_compared": [
      "attention",
      "dense_mlp",
      "expert_outputs",
      "experts_output",
      "kv_cache",
      "logits",
      "moe_output",
      "routed",
      "router_indices",
      "router_logits",
      "router_scores",
      "router_weights",
      "shared_output",
      "token",
      "top_k_index",
      "top_k_weights"
    ],
    "expert_outputs_checked_layers": 17732,
    "expert_outputs_layers": 18876,
    "audit_failures": 0,
    "os_counters_match_all_steps": true,
    "poisoned_steps": 27,
    "passed": true
  },
  "A": {
    "expert_bytes": 28789702656,
    "gpu_budget_bytes": 6000000000,
    "gpu_total_bytes": 8585084928,
    "experts_over_budget": 4.798283776,
    "experts_over_gpu": 3.3534557779508085,
    "min_experts_over_budget": 4.0,
    "completed_steps": {
      "stream": 144,
      "compact": 72,
      "chunk-1": 36,
      "lru-40": 144,
      "lru-80": 144,
      "hotness-80": 144,
      "shared-streamed": 36,
      "all-experts": 6
    },
    "max_peak_device_bytes": 5425945088,
    "max_peak_reserved_bytes": 5999951872,
    "passed": true
  },
  "B": {
    "worst_process_bytes": {
      "stream": {
        "working_set": 3289296896,
        "host_commit": 4562243584,
        "private_bytes_with_device_memory": 9919475712
      },
      "reference": {
        "working_set": 2946822144,
        "host_commit": 3273793536,
        "private_bytes_with_device_memory": 10857095168
      }
    },
    "fraction_of_experts": {
      "stream": {
        "working_set": 0.1142525483956148,
        "host_commit": 0.15846789522326632
      },
      "reference": {
        "working_set": 0.10235681066979894,
        "host_commit": 0.11371404474431818
      }
    },
    "max_fraction_of_experts": 0.25,
    "passed": true
  },
  "C": {
    "steps_with_a_budget": 654,
    "steps_over_budget": 0,
    "max_compact_fraction_of_layer": 0.15625,
    "max_compact_fraction_of_layer_allowed": 0.25,
    "prefill_steps_completed": 74,
    "prefill_max_compact_bytes": 173015040,
    "prefill_max_routed_experts_per_layer": 64,
    "layer_bytes": 1107296256,
    "working_set_within_capacity_plus_budget": true,
    "unbounded_compact_max_bytes": 1107296256,
    "passed": true
  },
  "D": {
    "decode_drive_fraction_without_cache": 0.09379438920454543,
    "max_decode_drive_fraction_without_cache": 0.11,
    "all_experts_decode_drive_fraction": 1.0004734848484849,
    "decode_drive_ratio_to_all_experts": 0.09374999999999997,
    "max_decode_drive_ratio_to_all_experts": 0.12,
    "decode_drive_fraction_by_configuration": {
      "stream": 0.09379438920454543,
      "compact": 0.09379438920454544,
      "chunk-1": 0.09379438920454546,
      "lru-40": 0.09379438920454543,
      "lru-80": 0.09379438920454543,
      "hotness-80": 0.07961343218396594,
      "shared-streamed": 0.09379438920454546,
      "all-experts": 1.0004734848484849
    },
    "prefill_drive_fraction_by_configuration": {
      "stream": 0.7925550732023511,
      "compact": 0.8001082271406978,
      "chunk-1": 0.6768527797885708,
      "lru-40": 0.7925550732023511,
      "lru-80": 0.7925550732023511,
      "hotness-80": 0.7814445940208881,
      "shared-streamed": 0.6768527797885708,
      "all-experts": 1.0004734848484849
    },
    "passed": true
  },
  "E": {
    "configurations": {
      "compact": {
        "steps": 72,
        "equal": 72,
        "chunked_calls": 0
      },
      "chunk-1": {
        "steps": 36,
        "equal": 36,
        "chunked_calls": 936
      },
      "stream": {
        "steps": 144,
        "equal": 144,
        "chunked_calls": 416
      }
    },
    "passed": true
  },
  "F": {
    "test_layering": "6 passed in 0.17s",
    "passed": true
  }
}
```

## Reproducibility

```json
{
  "other": "experiments/phase4b/moonlight-run1",
  "same_source_tree": false,
  "python_hash_seeds": [
    "1",
    "1"
  ],
  "digests_equal": {
    "index_sha256": true,
    "reference_rows_sha256": true,
    "prompts_sha256": true,
    "reference_sha256": true,
    "records_sha256": true,
    "ranges_sha256": true
  }
}
```
