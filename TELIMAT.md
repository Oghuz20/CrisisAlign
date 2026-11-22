# GPU-da tam öyrətmə üçün təlimat

Bu təlimat layihənin bütün eksperimentlərini GPU olan kompüterdə işlətmək üçündür:
4 model (text, image, multimodal, multimodal + modality dropout) × 3 seed = 12 öyrətmə.
16 GB GPU-da təxminən 4–6 saat çəkir.

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
python -m src.reproduce --quick --extra --workers 4
```

## 8. Tam iş (4–6 saat)
```
python -m src.reproduce --extra --workers 4
```
Yarımçıq kəsilsə, eyni əmri təkrar işlət: bitmiş öyrətmələri atlayır.
GPU yaddaşı çatmasa (out of memory): `--extra --workers 4 --batch-size 16`.

## 9. Nəticələri geri göndər
GitHub ilə: `runs/` və `report/figures/` qovluqlarını commit edib push et
(`.pt` faylları `.gitignore`-a görə avtomatik çıxarılır).
Və ya bu iki qovluğu zip edib göndər (`.pt` faylları lazım deyil).
Xəta çıxsa, xəta mətnini olduğu kimi göndər.
