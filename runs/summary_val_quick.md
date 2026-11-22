# Results on val

## Clean input: humanitarian labels (macro-F1 overall and on agree / conflict posts)

                            all   agree conflict critical_recall     ECE
output model                                                            
image  image             0.5325  0.4945   0.4667          0.7750  0.1397
       multimodal        0.6014  0.6462   0.4922          0.7000  0.1208
       multimodal_md0.3  0.5804  0.5761   0.4992          0.7750  0.1438
text   multimodal        0.3320  0.4208   0.2631          0.0625  0.1082
       multimodal_md0.3  0.2714  0.3057   0.2484          0.0000  0.0591
       text              0.3012  0.3131   0.2678          0.0625  0.0904

## Clean input: conflict warning

                 conflict_F1 conflict_recall   AUROC
model                                               
always conflict       0.7500          1.0000  0.5000
text+image pair       0.7630          0.8583  0.7294
multimodal            0.7742          0.9000  0.6979
multimodal_md0.3      0.7801          0.9167  0.6895

## Stress tests: label macro-F1 by condition (mean over seeds)

condition                 clean  text_missing  image_missing  text_swapped  image_swapped
output model                                                                             
image  image             0.5325        0.5325         0.0667        0.5325         0.5255
       multimodal        0.6014        0.5039         0.2705        0.5116         0.5014
       multimodal_md0.3  0.5804        0.5305         0.3475        0.4961         0.4830
text   multimodal        0.3320        0.2385         0.3581        0.2600         0.2574
       multimodal_md0.3  0.2714        0.2204         0.3457        0.2134         0.2034
       text              0.3012        0.0667         0.3012        0.2845         0.3012

## Stress tests: conflict-warning AUROC by condition

condition          clean  text_missing  image_missing  text_swapped  image_swapped
model                                                                             
multimodal        0.6979        0.7069         0.5911        0.4101         0.5937
multimodal_md0.3  0.6895        0.6959         0.5897        0.4126         0.5853