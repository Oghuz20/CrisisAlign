# Results on test

## Clean input: humanitarian labels (macro-F1 overall and on agree / conflict posts)

                                     all            agree         conflict  critical_recall              ECE
output model                                                                                                
image  image             0.6778 ± 0.0071  0.7069 ± 0.0071  0.6281 ± 0.0029  0.4686 ± 0.0446  0.0831 ± 0.0451
       multimodal        0.6848 ± 0.0171  0.7674 ± 0.0035  0.6019 ± 0.0233  0.4323 ± 0.0702  0.0633 ± 0.0346
       multimodal_md0.3  0.6946 ± 0.0062  0.7664 ± 0.0076  0.6132 ± 0.0140  0.4257 ± 0.0551  0.0697 ± 0.0230
text   multimodal        0.6761 ± 0.0126  0.7090 ± 0.0232  0.5892 ± 0.0054  0.6200 ± 0.0283  0.1072 ± 0.0656
       multimodal_md0.3  0.6774 ± 0.0087  0.7261 ± 0.0257  0.5775 ± 0.0132  0.6200 ± 0.0565  0.1125 ± 0.0226
       text              0.6671 ± 0.0078  0.6951 ± 0.0099  0.5922 ± 0.0096  0.6084 ± 0.0252  0.1169 ± 0.0585

## Clean input: conflict warning

                      conflict_F1  conflict_recall            AUROC
model                                                              
always conflict   0.7230 ± 0.0000  1.0000 ± 0.0000  0.5000 ± 0.0000
text+image pair   0.7850 ± 0.0011  0.8399 ± 0.0051  0.8175 ± 0.0057
multimodal        0.7634 ± 0.0135  0.8102 ± 0.0405  0.7799 ± 0.0087
multimodal_md0.3  0.7579 ± 0.0165  0.7967 ± 0.0480  0.7761 ± 0.0060

## Stress tests: label macro-F1 by condition (mean over seeds)

condition                 clean  text_missing  image_missing  text_swapped  image_swapped
output model                                                                             
image  image             0.6778        0.6778         0.1282        0.6778         0.6714
       multimodal        0.6848        0.6617         0.2635        0.6106         0.5928
       multimodal_md0.3  0.6946        0.6712         0.3617        0.6067         0.5891
text   multimodal        0.6761        0.2609         0.6687        0.6589         0.6585
       multimodal_md0.3  0.6774        0.2921         0.6700        0.6365         0.6424
       text              0.6671        0.0873         0.6671        0.6737         0.6671

## Stress tests: conflict-warning AUROC by condition

condition          clean  text_missing  image_missing  text_swapped  image_swapped
model                                                                             
multimodal        0.7799        0.7019         0.7091        0.6133         0.5688
multimodal_md0.3  0.7761        0.7045         0.7164        0.6007         0.5566