# GPU-da tam öyrətmə üçün təlimat

Bu təlimat layihənin bütün eksperimentlərini GPU olan kompüterdə işlətmək üçündür:
5 model (text, image, multimodal, multimodal + modality dropout, multimodal + swap augmentation) × 3 seed = 15 öyrətmə.
Bu, **versiya 3**-dür (v2 + şəbəkələrin daxilində daha güclü regularizasiya): nəticələr `runs/v3/` və
`report/figures/v3/` qovluqlarına yazılır, versiya 1 və 2-nin nəticələri toxunulmaz qalır.
RTX 5070 Ti (16 GB) kimi kartda təxminən 1.5–2.5 saat çəkir.

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

## 7. Əvvəl qısa yoxlama (~10 dəqiqə)
```
python -m src.reproduce --quick --extra --workers 6
```

## 8. Tam iş (~1.5–2.5 saat)
Kod artıq klonlanıbsa, əvvəlcə yeniləmələri gətir:
```
git pull
```
Sonra iki əmr, ARDICIL (birincisi öyrədir və validation-da yoxlayır, ikincisi eyni modelləri test-də yoxlayır):
```
python -m src.reproduce --extra --workers 6
python -m src.reproduce --split test --extra --workers 6
```
İkinci əmr modelləri yenidən öyrətmir, yalnız test qiymətləndirməsi edir (~10 dəqiqə).
Yarımçıq kəsilsə, eyni əmri təkrar işlət: bitmiş öyrətmələri atlayır.
GPU yaddaşı çatmasa (out of memory): `--extra --workers 6 --batch-size 16`.
Prosessor zəifdirsə və ya kompüter donursa, `--workers 6` əvəzinə `--workers 2` yazın.

## 9. Nəticələri geri göndər
GitHub ilə: `runs/v3/` və `report/figures/v3/` qovluqlarını commit edib push et:
```
git add runs/v3 report/figures/v3
git commit -m "Version 3 runs: validation and test results"
git push
```
(`.pt` faylları `.gitignore`-a görə avtomatik çıxarılır).
Və ya bu iki qovluğu zip edib göndər (`.pt` faylları lazım deyil).
Xəta çıxsa, xəta mətnini olduğu kimi göndər.
