# STen Yeniden Üretim – Rapor Notları

Bu dosyaya raporda yazılacak her şey eklenir: ortam, sapmalar, bulgular, engeller.
İşaretler: ✅ = bizim bilgisayarlarımızdan birinde görüldü | 🔬 = henüz doğrulanmadı

## 1. Kaynaklar
- Makale: Ivanov vd., "STen: Productive and Efficient Sparsity in PyTorch", ACM TACO 23(2), Art. 74, Haziran 2026. doi:10.1145/3815424
- Orijinal kod: https://github.com/spcl/sten , commit 9110245 (2025-03-22, "doublesparse")
- Bizim fork: https://github.com/yagmursavran/sten-reproduce

## 2. Donanım
| Makine | İşlemci | GPU | RAM | Rol |
|---|---|---|---|---|
| Victus 16 | AMD Ryzen 7 7840HS (8 çekirdek, AVX2 + AVX-512) | RTX 4050 6 GB | ? | CPU ölçümleri |
| 2. laptop | AMD Ryzen 7 7840HS | RTX 3050 6 GB | ? | Eğitim deneyleri |
| Mac | Apple M4 (ARM) | dahili | ? | ARM uyarlaması (aday) |
Makaledeki donanım: AMD EPYC 7742 (64 çekirdek, AVX2), Intel Xeon Gold 6130 (16 çekirdek, AVX-512), 3x RTX 3090, V100, Piz Daint (P100).

## 3. Yazılım ortamı (makale vs biz)
| Bileşen | Makale | Biz | Not |
|---|---|---|---|
| İşletim sistemi | belirtilmemiş | Ubuntu 24.04 (WSL2) | |
| Python | belirtilmemiş | 3.10.22 (uv ile) | DaCe 0.14.4 <3.12 istiyor, DeepSparse 1.1.0 <=3.10 istiyor |
| PyTorch | 1.12 | 2.4.1 | 2.15 STen'in yedek yolunu bozuyor; 1.12 ileride denenebilir |
| CUDA | 11.4 | (GPU kurulumuna göre doldurulacak) | |
| GCC | 8.4.1 | 13.x (Ubuntu 24.04) | |
| DaCe | belirtilmemiş | 0.14.4 | 1.0+ n:m:g kod üreticisini bozuyor |
| networkx | belirtilmemiş | 3.4.2 | 3.5+ DaCe 0.14'ü bozuyor |
| transformers | belirtilmemiş | 4.44.2 | yeni sürümler PyTorch >=2.6 istiyor |
| numpy | belirtilmemiş | 1.26.4 | |

## 4. Kurulum sırasında karşılaşılan engeller
1. ✅ STen'in pyproject.toml dosyasında hiçbir paket sürümü sabitlenmemiş. `pip install .` her seferinde en yeni sürümleri kuruyor; iki farklı denemede farklı sürümler geldi (DaCe 1.0.2 / 0.16).
2. ✅ `pip install .` (GPU'lu PyTorch dahil) yavaş internette 40 dakikayı aştı.
3. ✅ En yeni sürümlerle (torch 2.15.0, dace 1.0.2, networkx 3.7, transformers 5.22.1): STen testlerinde 3 başarısız:
   - test_modify_bert_encoder: torch.hub, HuggingFace'in hubconf.py dosyasını arıyor; dosya HuggingFace reposundan kaldırılmış (FileNotFoundError). Bu test artık hiçbir sürümle geçemez.
   - test_fallback_implementations: torch 2.15'te torch.mm'in imzası değişmiş; STen'in yedek yolu fazladan None gönderiyor (TypeError). torch 2.4.1 ile geçiyor (🔬 bizde tekrar doğrulanacak).
   - test_clone: yazarların kodunda TODO olarak bırakılmış eksik; torch 2.4.1'de de düşüyor.
4. ✅ STen eski PyTorch API'leri kullanıyor (TypedStorage, torch.distributed.reduce_op): yeni sürümlerde deprecation uyarıları.
5. ✅ DaCe 0.14.4 Python 3.12'yi desteklemiyor (pip: "Requires-Python >=3.6,<3.12"). Bu yüzden Python 3.10 ortamı kuruldu.
6. 🔬 DaCe 1.0.2 + networkx 3.6+: n:m:g kodu derlenirken "KeyError: SDFGState". Sebep networkx 3.5+; networkx 3.4.2 ile çözülüyor.
7. 🔬 DaCe 0.16 + sympy 1.14: n:m:g derlemesinde "TypeError: 'Attr' object is not callable".
8. 🔬 TVM 0.10.0 (makaledeki) PyPI'de yok; PyPI'de sadece 0.25+ var. TVM 0.20'de Relay ve te.Schedule kaldırılmış; makaledeki TVM deneyi yeni sürümde birebir tekrarlanamaz.
9. 🔬 DeepSparse 1.1.0 sadece Linux x86_64 ve Python 3.6–3.10 için paket sunuyor. Son sürüm 1.9.0 (Haziran 2025) sadece kaynak kod olarak yayınlanmış.

## 5. Kodda bulunan hata: n:m:g çarpımı girdiyi transpoze okuyor
- Yer: src/sten/grouped_nm/sten_impls.py, satır 53: `trans_b = not dense.is_contiguous()`
- linear() fonksiyonu trans_b=True gönderiyor; bu satır onu eziyor. PyTorch'tan gelen girdi contiguous olduğu için trans_b=False oluyor ve çekirdek girdiyi transpoze okuyor. Satırın üstündeki yorum ile kodun yaptığı şey çelişiyor.
- Kanıt (hata_testi.py, 64x64 girdi, 2:4:2):
  - STen vs doğru sonuç: ~2.75 (yanlış)
  - STen vs linear(x.T, W): ~4.8e-07 (yani girdi ters okunuyor)
- Hız etkilenmiyor (aynı sayıda işlem); makale doğruluğu maskeli dense ile ölçtüğü için muhtemelen makaledeki sonuçları etkilemiyor.
- Önerilen düzeltme: `trans_b = trans_b and dense.is_contiguous()` → hata ~1e-6'ya iniyor.
- Durum: 🔬 Claude'un ortamında (Python 3.10/3.11/3.12) görüldü; bizim bilgisayarlarda doğrulanacak.
- Önemli: STen'in kendi testleri (tests/test_nmg.py) sonucun doğruluğunu kontrol etmiyor, bu yüzden hata yakalanmamış.

## 6. Makaledeki eksikler (reproduce açısından)
- Sadece PyTorch, CUDA ve GCC sürümleri belirtilmiş; diğer bağımlılıkların sürümü yok.
- Figür 12 (energy), Figür 15 (DeepSparse/TVM karşılaştırması) ve Tablo 3 (OPT + SparseGPT) için betikler repoda yok.
- Özet "best available implementation" diyor; kodda maliyet modeli yok, sadece tam eşleşme aranıyor.

## 7. Deney günlüğü
(Her deney için: tarih, kim, makine, komut, sonuç)

