# Results on val (experiment v2)

## Clean input: humanitarian labels (macro-F1 overall and on agree / conflict posts)

                                        all            agree         conflict  critical_recall              ECE
output model                                                                                                   
image  image                0.6533 ± 0.0078  0.6366 ± 0.0147  0.6194 ± 0.0065  0.3295 ± 0.0404  0.0248 ± 0.0018
       multimodal           0.6793 ± 0.0059  0.6966 ± 0.0123  0.6234 ± 0.0164  0.3716 ± 0.0176  0.0232 ± 0.0040
       multimodal_md0.3     0.6784 ± 0.0019  0.7170 ± 0.0038  0.6052 ± 0.0029  0.3602 ± 0.0066  0.0285 ± 0.0046
       multimodal_swap0.25  0.6713 ± 0.0107  0.6924 ± 0.0236  0.6186 ± 0.0066  0.3678 ± 0.0345  0.0313 ± 0.0118
text   multimodal           0.6947 ± 0.0059  0.7178 ± 0.0067  0.6012 ± 0.0059  0.6294 ± 0.0185  0.0568 ± 0.0177
       multimodal_md0.3     0.7011 ± 0.0036  0.7545 ± 0.0034  0.5872 ± 0.0015  0.6503 ± 0.0484  0.0390 ± 0.0124
       multimodal_swap0.25  0.6936 ± 0.0044  0.7116 ± 0.0113  0.6069 ± 0.0026  0.6224 ± 0.0252  0.0444 ± 0.0295
       text                 0.7004 ± 0.0088  0.7252 ± 0.0212  0.6128 ± 0.0083  0.6410 ± 0.0040  0.0269 ± 0.0029

## Clean input: conflict warning

                         conflict_F1  conflict_recall            AUROC
model                                                                 
always conflict      0.7069 ± 0.0000  1.0000 ± 0.0000  0.5000 ± 0.0000
text+image pair      0.7821 ± 0.0043  0.8425 ± 0.0025  0.8208 ± 0.0038
multimodal           0.7463 ± 0.0101  0.7950 ± 0.0221  0.7728 ± 0.0052
multimodal_md0.3     0.7410 ± 0.0087  0.7771 ± 0.0297  0.7752 ± 0.0022
multimodal_swap0.25  0.7572 ± 0.0009  0.8517 ± 0.0161  0.7765 ± 0.0063

## Stress tests: label macro-F1 by condition (mean over seeds)

condition                    clean  text_missing  image_missing  text_swapped  image_swapped
output model                                                                                
image  image                0.6533        0.6533         0.1309        0.6533         0.6573
       multimodal           0.6793        0.6505         0.3288        0.6072         0.6060
       multimodal_md0.3     0.6784        0.6463         0.3773        0.6030         0.6036
       multimodal_swap0.25  0.6713        0.6561         0.2184        0.6191         0.6364
text   multimodal           0.6947        0.2295         0.6947        0.6946         0.6904
       multimodal_md0.3     0.7011        0.2856         0.6950        0.6700         0.6749
       multimodal_swap0.25  0.6936        0.2444         0.6920        0.6935         0.6872
       text                 0.7004        0.0874         0.7004        0.7084         0.7004

## Stress tests: conflict-warning AUROC by condition

condition             clean  text_missing  image_missing  text_swapped  image_swapped
model                                                                                
multimodal           0.7728        0.6829         0.7089        0.6000         0.5194
multimodal_md0.3     0.7752        0.6972         0.7266        0.5971         0.5073
multimodal_swap0.25  0.7765        0.6893         0.7109        0.6129         0.5488
text+image pair      0.8208        0.5925         0.6412        0.8105         0.7887