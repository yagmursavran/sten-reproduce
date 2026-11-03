# n:m:g hızlı çarpım kodunun doğruluk testi
# Aynı budanmış ağırlıkla: STen'in n:m:g kodu vs normal PyTorch çarpımı
import torch, sten
F = torch.nn.functional
torch.manual_seed(0)                       # her çalıştırmada aynı rastgele sayılar

lin = torch.nn.Linear(64, 96)              # 64 girdi, 96 çıktılı normal bir katman
sb = sten.SparsityBuilder()
sb.set_weight(name="weight",
              initial_sparsifier=sten.GroupedNMSparsifier(n=2, m=4, g=2),   # kural: 2:4, grup 2
              out_format=sten.GroupedNMTensor)                              # format: n:m:g
seyrek = sb.get_sparse_model(lin)          # ağırlığı budanmış, n:m:g formatında katman
W = seyrek.weight.wrapped_tensor.to_dense()   # aynı budanmış ağırlığın normal matris hali

x = torch.randn(64, 64)                    # 64 satırlık rastgele girdi
with torch.no_grad():
    stenin_sonucu = seyrek(x)                       # Yol A: STen'in n:m:g kodu
    dogru_sonuc   = F.linear(x, W, lin.bias)        # Yol B: normal PyTorch (doğru cevap)
    ters_okunmus  = F.linear(x.T, W, lin.bias)      # girdi ters okunsaydı ne çıkardı?

print("STen vs doğru sonuç:       ", (stenin_sonucu - dogru_sonuc).abs().max().item())
print("STen vs ters okunmuş girdi:", (stenin_sonucu - ters_okunmus).abs().max().item())
