# GPU-da tam öyrətmə üçün təlimat

Bu təlimat v2 resepti ilə 6 təlim üçündür: sadə multimodal model və
`--swap-aug 0.25`, hər biri 42, 43 və 44 seed-ləri ilə.
Yeni nəticələr `runs/disagreement/v2/` və `report/generated/` qovluqlarına yazılır.

## 1. Kodu götür
GitHub-dan:
```
git clone https://github.com/Oghuz20/CrisisAlign.git
cd CrisisAlign
```
və ya göndərilən `DL_final_project_3_code.zip` faylını aç.

## 2. Virtual mühit
```
python -m venv .venv
.venv\Scripts\activate
```

## 3. ƏVVƏLCƏ PyTorch-un GPU versiyası
pytorch.org → "Get Started" səhifəsində öz CUDA versiyanı seç və verilən `pip install torch torchvision ...` əmrini işlət.
Adi `pip install torch` CPU versiyasını qurur, GPU işləməz.

**RTX 50 seriyası (məsələn, 5070 Ti) üçün vacib:** bu kartlar (Blackwell) **CUDA 12.8 və ya daha yeni**
PyTorch versiyası tələb edir. pytorch.org-da CUDA 12.8+ variantını seçin. Köhnə CUDA versiyası ilə
"no kernel image is available" xətası çıxa bilər.

## 4. Qalan paketlər
`requirements.txt`-i BU MƏRHƏLƏDƏ işlətmə (GPU torch-u CPU versiyası ilə əvəz edə bilər). Bunu işlət:
```
pip install transformers pandas numpy scikit-learn pillow ftfy matplotlib huggingface_hub
```

## 5. GPU-nu yoxla (True çıxmalıdır)
```
python -c "import torch; print(torch.cuda.is_available())"
```

## 6. Datanı yüklə (~1.9 GB)
```
hf download QCRI/CrisisMMD --repo-type dataset --local-dir CrisisMMD
```

## 7. İstəyə bağlı qısa yoxlama və əsas təlim

```
python -m src.reproduce --recipe v2 --matrix fusion --quick --tag smoke --workers 6
python -m src.reproduce --recipe v2 --matrix fusion --seeds 42 43 44 --tag track2 --workers 6
```

İkinci əmr hər seed üçün təsnifat makro-F1 və humanitar etiket uyğunsuzluğu AP
göstəricilərinə görə ayrı checkpoint seçir. Təlim yarımçıq qalıbsa, eyni əmrə
`--resume` əlavə et. Eyni eksperimentdə worker sayı və başqa parametrlər sabit qalsın.

## 8. Yekun test

Validation nəticələrinə əsasən müqayisə qaydasını təsdiqlədikdən sonra:

```
python -m src.prepare_data --splits test
python -m src.reproduce --recipe v2 --matrix fusion --seeds 42 43 44 --tag track2 --workers 6 --split test
```

Test əmri yalnız hazırlanmış checkpoint-ləri qiymətləndirir. Çatışmayan
checkpoint olduqda xəta verir.

## 9. Nəticələri geri göndər

`runs/disagreement/v2/` və `report/generated/` qovluqlarındakı nəticə fayllarını
paylaş. `.pt` fayllarını GPU kompüterində və ya ortaq yaddaşda saxla.
`config.json` faylında hər checkpoint-in epoch və SHA-256 məlumatı var.
