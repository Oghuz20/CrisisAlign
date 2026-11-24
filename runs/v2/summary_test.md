# Results on test (experiment v2)

## Clean input: humanitarian labels (macro-F1 overall and on agree / conflict posts)

                                        all            agree         conflict  critical_recall              ECE
output model                                                                                                   
image  image                0.6841 ± 0.0072  0.7054 ± 0.0113  0.6322 ± 0.0066  0.4257 ± 0.0396  0.0212 ± 0.0107
       multimodal           0.6916 ± 0.0133  0.7534 ± 0.0093  0.6150 ± 0.0160  0.4422 ± 0.0375  0.0240 ± 0.0076
       multimodal_md0.3     0.7019 ± 0.0030  0.7700 ± 0.0080  0.6218 ± 0.0040  0.4455 ± 0.0198  0.0242 ± 0.0029
       multimodal_swap0.25  0.6904 ± 0.0089  0.7445 ± 0.0166  0.6187 ± 0.0074  0.4488 ± 0.0488  0.0400 ± 0.0063
text   multimodal           0.6694 ± 0.0083  0.6973 ± 0.0113  0.5906 ± 0.0075  0.6061 ± 0.0283  0.0774 ± 0.0200
       multimodal_md0.3     0.6750 ± 0.0076  0.7176 ± 0.0115  0.5787 ± 0.0095  0.6061 ± 0.0359  0.0554 ± 0.0091
       multimodal_swap0.25  0.6741 ± 0.0092  0.7044 ± 0.0159  0.5978 ± 0.0077  0.6037 ± 0.0291  0.0544 ± 0.0380
       text                 0.6730 ± 0.0041  0.6944 ± 0.0103  0.6020 ± 0.0060  0.5967 ± 0.0283  0.0300 ± 0.0034

## Clean input: conflict warning

                         conflict_F1  conflict_recall            AUROC
model                                                                 
always conflict      0.7230 ± 0.0000  1.0000 ± 0.0000  0.5000 ± 0.0000
text+image pair      0.7922 ± 0.0031  0.8541 ± 0.0037  0.8248 ± 0.0039
multimodal           0.7638 ± 0.0070  0.8002 ± 0.0220  0.7768 ± 0.0048
multimodal_md0.3     0.7577 ± 0.0118  0.7873 ± 0.0335  0.7792 ± 0.0054
multimodal_swap0.25  0.7784 ± 0.0057  0.8623 ± 0.0135  0.7828 ± 0.0036

## Stress tests: label macro-F1 by condition (mean over seeds)

condition                    clean  text_missing  image_missing  text_swapped  image_swapped
output model                                                                                
image  image                0.6841        0.6841         0.1282        0.6841         0.6743
       multimodal           0.6916        0.6727         0.3258        0.6492         0.6359
       multimodal_md0.3     0.7019        0.6814         0.3770        0.6414         0.6174
       multimodal_swap0.25  0.6904        0.6839         0.2310        0.6659         0.6495
text   multimodal           0.6694        0.2352         0.6706        0.6703         0.6667
       multimodal_md0.3     0.6750        0.2676         0.6724        0.6473         0.6493
       multimodal_swap0.25  0.6741        0.2484         0.6700        0.6759         0.6678
       text                 0.6730        0.0873         0.6730        0.6793         0.6730

## Stress tests: conflict-warning AUROC by condition

condition             clean  text_missing  image_missing  text_swapped  image_swapped
model                                                                                
multimodal           0.7768        0.6978         0.7057        0.6082         0.5620
multimodal_md0.3     0.7792        0.7122         0.7205        0.5972         0.5406
multimodal_swap0.25  0.7828        0.7031         0.7099        0.6179         0.5829
text+image pair      0.8248        0.6018         0.6342        0.8056         0.7918