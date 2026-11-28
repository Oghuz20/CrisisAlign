# Humanitarian-label disagreement results on val

Results are mean ± sample SD when multiple seeds contribute. Parentheses give metric support in seeds.

Fusion F1, precision and recall use disagreement score > 0.5. The independent pair uses hard class disagreement for those metrics and the soft score for AP and AUROC.

A missing-input warning predicts the original pair’s label disagreement from the visible input. The corresponding hidden-label classification result measures label inference.

## Clean classification

| objective | model | output | subset | n | macro_f1 | accuracy | critical_recall | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | multimodal | image | agree | 1 | 0.7137 (1) | 0.8225 (1) | 0.4375 (1) | 0.0504 (1) |
| classification | multimodal | image | all | 1 | 0.6757 (1) | 0.7662 (1) | 0.4023 (1) | 0.0328 (1) |
| classification | multimodal | image | conflict | 1 | 0.6015 (1) | 0.7195 (1) | 0.3944 (1) | 0.0331 (1) |
| classification | multimodal | text | agree | 1 | 0.7285 (1) | 0.7732 (1) | 0.8125 (1) | 0.0328 (1) |
| classification | multimodal | text | all | 1 | 0.6948 (1) | 0.7202 (1) | 0.6503 (1) | 0.0417 (1) |
| classification | multimodal | text | conflict | 1 | 0.5951 (1) | 0.6762 (1) | 0.6299 (1) | 0.0677 (1) |
| classification | multimodal_swap0.25 | image | agree | 1 | 0.6819 (1) | 0.8047 (1) | 0.2500 (1) | 0.0408 (1) |
| classification | multimodal_swap0.25 | image | all | 1 | 0.6572 (1) | 0.7662 (1) | 0.2529 (1) | 0.0363 (1) |
| classification | multimodal_swap0.25 | image | conflict | 1 | 0.6039 (1) | 0.7343 (1) | 0.2535 (1) | 0.0370 (1) |
| classification | multimodal_swap0.25 | text | agree | 1 | 0.7205 (1) | 0.7623 (1) | 0.7500 (1) | 0.0283 (1) |
| classification | multimodal_swap0.25 | text | all | 1 | 0.6922 (1) | 0.7135 (1) | 0.6364 (1) | 0.0239 (1) |
| classification | multimodal_swap0.25 | text | conflict | 1 | 0.6000 (1) | 0.6729 (1) | 0.6220 (1) | 0.0379 (1) |
| warning | multimodal | image | agree | 1 | 0.6685 (1) | 0.8195 (1) | 0.1250 (1) | 0.0495 (1) |
| warning | multimodal | image | all | 1 | 0.6698 (1) | 0.7738 (1) | 0.2759 (1) | 0.0411 (1) |
| warning | multimodal | image | conflict | 1 | 0.6152 (1) | 0.7359 (1) | 0.3099 (1) | 0.0521 (1) |
| warning | multimodal | text | agree | 1 | 0.7314 (1) | 0.7761 (1) | 0.6875 (1) | 0.0430 (1) |
| warning | multimodal | text | all | 1 | 0.6964 (1) | 0.7237 (1) | 0.5944 (1) | 0.0266 (1) |
| warning | multimodal | text | conflict | 1 | 0.5976 (1) | 0.6803 (1) | 0.5827 (1) | 0.0413 (1) |
| warning | multimodal_swap0.25 | image | agree | 1 | 0.6819 (1) | 0.8047 (1) | 0.2500 (1) | 0.0408 (1) |
| warning | multimodal_swap0.25 | image | all | 1 | 0.6572 (1) | 0.7662 (1) | 0.2529 (1) | 0.0363 (1) |
| warning | multimodal_swap0.25 | image | conflict | 1 | 0.6039 (1) | 0.7343 (1) | 0.2535 (1) | 0.0370 (1) |
| warning | multimodal_swap0.25 | text | agree | 1 | 0.7205 (1) | 0.7623 (1) | 0.7500 (1) | 0.0283 (1) |
| warning | multimodal_swap0.25 | text | all | 1 | 0.6922 (1) | 0.7135 (1) | 0.6364 (1) | 0.0239 (1) |
| warning | multimodal_swap0.25 | text | conflict | 1 | 0.6000 (1) | 0.6729 (1) | 0.6220 (1) | 0.0379 (1) |

## Disagreement warning

| objective | model | condition | n | conflict_f1 | conflict_precision | conflict_recall | conflict_ap | conflict_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | multimodal | clean | 1 | 0.7499 (1) | 0.7034 (1) | 0.8029 (1) | 0.7981 (1) | 0.7731 (1) |
| classification | multimodal | image_missing | 1 | 0.7058 (1) | 0.6473 (1) | 0.7760 (1) | 0.7299 (1) | 0.7050 (1) |
| classification | multimodal | image_swapped | 1 | 0.7401 (1) | 0.8739 (1) | 0.6418 (1) | 0.8877 (1) | 0.5309 (1) |
| classification | multimodal | text_missing | 1 | 0.6804 (1) | 0.6605 (1) | 0.7016 (1) | 0.7076 (1) | 0.6874 (1) |
| classification | multimodal | text_swapped | 1 | 0.7525 (1) | 0.8767 (1) | 0.6591 (1) | 0.8930 (1) | 0.6044 (1) |
| classification | multimodal_swap0.25 | clean | 1 | 0.7616 (1) | 0.6880 (1) | 0.8528 (1) | 0.7968 (1) | 0.7765 (1) |
| classification | multimodal_swap0.25 | image_missing | 1 | 0.7171 (1) | 0.6208 (1) | 0.8487 (1) | 0.7292 (1) | 0.7060 (1) |
| classification | multimodal_swap0.25 | image_swapped | 1 | 0.7792 (1) | 0.8755 (1) | 0.7020 (1) | 0.8980 (1) | 0.5594 (1) |
| classification | multimodal_swap0.25 | text_missing | 1 | 0.6868 (1) | 0.6596 (1) | 0.7163 (1) | 0.7078 (1) | 0.6901 (1) |
| classification | multimodal_swap0.25 | text_swapped | 1 | 0.8003 (1) | 0.8815 (1) | 0.7329 (1) | 0.8981 (1) | 0.6218 (1) |
| warning | multimodal | clean | 1 | 0.7601 (1) | 0.7046 (1) | 0.8250 (1) | 0.8029 (1) | 0.7807 (1) |
| warning | multimodal | image_missing | 1 | 0.7307 (1) | 0.6429 (1) | 0.8463 (1) | 0.7431 (1) | 0.7217 (1) |
| warning | multimodal | image_swapped | 1 | 0.7575 (1) | 0.8747 (1) | 0.6680 (1) | 0.8905 (1) | 0.5433 (1) |
| warning | multimodal | text_missing | 1 | 0.6582 (1) | 0.6819 (1) | 0.6361 (1) | 0.7108 (1) | 0.6913 (1) |
| warning | multimodal | text_swapped | 1 | 0.7722 (1) | 0.8790 (1) | 0.6886 (1) | 0.8915 (1) | 0.6025 (1) |
| warning | multimodal_swap0.25 | clean | 1 | 0.7616 (1) | 0.6880 (1) | 0.8528 (1) | 0.7968 (1) | 0.7765 (1) |
| warning | multimodal_swap0.25 | image_missing | 1 | 0.7171 (1) | 0.6208 (1) | 0.8487 (1) | 0.7292 (1) | 0.7060 (1) |
| warning | multimodal_swap0.25 | image_swapped | 1 | 0.7792 (1) | 0.8755 (1) | 0.7020 (1) | 0.8980 (1) | 0.5594 (1) |
| warning | multimodal_swap0.25 | text_missing | 1 | 0.6868 (1) | 0.6596 (1) | 0.7163 (1) | 0.7078 (1) | 0.6901 (1) |
| warning | multimodal_swap0.25 | text_swapped | 1 | 0.8003 (1) | 0.8815 (1) | 0.7329 (1) | 0.8981 (1) | 0.6218 (1) |

## Stress-test classification

| objective | model | output | condition | n | macro_f1 |
| --- | --- | --- | --- | --- | --- |
| classification | multimodal | image | image_missing | 1 | 0.3717 (1) |
| classification | multimodal | image | image_swapped | 1 | 0.6270 (1) |
| classification | multimodal | image | text_missing | 1 | 0.6365 (1) |
| classification | multimodal | image | text_swapped | 1 | 0.5987 (1) |
| classification | multimodal | text | image_missing | 1 | 0.6872 (1) |
| classification | multimodal | text | image_swapped | 1 | 0.6846 (1) |
| classification | multimodal | text | text_missing | 1 | 0.2263 (1) |
| classification | multimodal | text | text_swapped | 1 | 0.7162 (1) |
| classification | multimodal_swap0.25 | image | image_missing | 1 | 0.2333 (1) |
| classification | multimodal_swap0.25 | image | image_swapped | 1 | 0.6110 (1) |
| classification | multimodal_swap0.25 | image | text_missing | 1 | 0.6446 (1) |
| classification | multimodal_swap0.25 | image | text_swapped | 1 | 0.5998 (1) |
| classification | multimodal_swap0.25 | text | image_missing | 1 | 0.6889 (1) |
| classification | multimodal_swap0.25 | text | image_swapped | 1 | 0.6855 (1) |
| classification | multimodal_swap0.25 | text | text_missing | 1 | 0.2704 (1) |
| classification | multimodal_swap0.25 | text | text_swapped | 1 | 0.7085 (1) |
| warning | multimodal | image | image_missing | 1 | 0.2781 (1) |
| warning | multimodal | image | image_swapped | 1 | 0.6026 (1) |
| warning | multimodal | image | text_missing | 1 | 0.6381 (1) |
| warning | multimodal | image | text_swapped | 1 | 0.6030 (1) |
| warning | multimodal | text | image_missing | 1 | 0.6976 (1) |
| warning | multimodal | text | image_swapped | 1 | 0.6941 (1) |
| warning | multimodal | text | text_missing | 1 | 0.2363 (1) |
| warning | multimodal | text | text_swapped | 1 | 0.7130 (1) |
| warning | multimodal_swap0.25 | image | image_missing | 1 | 0.2333 (1) |
| warning | multimodal_swap0.25 | image | image_swapped | 1 | 0.6110 (1) |
| warning | multimodal_swap0.25 | image | text_missing | 1 | 0.6446 (1) |
| warning | multimodal_swap0.25 | image | text_swapped | 1 | 0.5998 (1) |
| warning | multimodal_swap0.25 | text | image_missing | 1 | 0.6889 (1) |
| warning | multimodal_swap0.25 | text | image_swapped | 1 | 0.6855 (1) |
| warning | multimodal_swap0.25 | text | text_missing | 1 | 0.2704 (1) |
| warning | multimodal_swap0.25 | text | text_swapped | 1 | 0.7085 (1) |
