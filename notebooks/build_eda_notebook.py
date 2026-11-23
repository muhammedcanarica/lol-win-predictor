"""EDA notebook'unu üretir. Çalıştırma: python notebooks/build_eda_notebook.py
Ardından: jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 01 – Keşifsel Veri Analizi (EDA)\n\n**Veri:** ChainCC Open LoL Esports Dataset 2026 (`chaincc-teams-2026.csv.gz`, CC BY 4.0)\n\n"
       "Amaç: 10. dakika istatistiklerinin Blue takımının galibiyetini tahmin etmek için kullanılabilirliğini incelemek ve temizleme kararlarını gerekçelendirmek."),
    code("import sys, pathlib\nROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\nimport pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns\n"
         "from src.data import load_raw, clean, FEATURES, AT10_COLS\nFIG = ROOT / 'reports' / 'figures'\nFIG.mkdir(parents=True, exist_ok=True)\n"
         "sns.set_theme(style='whitegrid')\ndf = load_raw()\ndf.shape"),
    md("## 1. Boyutlar, benzersiz maçlar ve veri tipleri"),
    code("print('Satır:', len(df), '| Sütun:', df.shape[1], '| Benzersiz maç:', df.game_id.nunique())\n"
         "print('Maç başına satır sayısı:', df.groupby('game_id').size().value_counts().to_dict())\n"
         "print('Taraf dağılımı:', df.side.value_counts().to_dict())\n"
         "print('Her maçta tam bir kazanan var mı?', (df.groupby('game_id').result.sum() == 1).all())\n"
         "df.dtypes.value_counts()"),
    code("df.dtypes.to_frame('tip').T"),
    md("## 2. Eksik değerler ve tekrarlanan kayıtlar"),
    code("na = df.isna().sum(); na[na > 0].sort_values(ascending=False).to_frame('eksik')"),
    code("print('Tam tekrar eden satır:', df.duplicated().sum())\nprint('Tekrar eden (game_id, side):', df.duplicated(['game_id', 'side']).sum())"),
    md("Eksikler yalnızca `split`, `bo_format`, `match_in_series`, `game_in_match`, `team_id` ve ban sütunlarında. "
       "Bunların hiçbiri modelde kullanılmıyor, bu yüzden silme/doldurma gerekmiyor."),
    md("## 3. 10. dakika istatistiklerinin kullanılabilirliği\n\n`goldat10` bir takım için 0 olamaz (oyun başlangıç altınıyla başlar). "
       "Bu değerin 0 olduğu satırlarda tüm 10. dk sütunları da 0 mı?"),
    code("zero = (df[AT10_COLS] == 0).all(axis=1)\nprint('Tüm 10. dk değerleri 0 olan satır:', zero.sum(), '| maç:', df.loc[zero, 'game_id'].nunique())\n"
         "print('goldat10 == 0 olan satır:', (df.goldat10 == 0).sum())\n"
         "print('Bu maçlarda gerçek maç sonu verisi var mı? (ortalama kills):', df.loc[zero, 'kills'].mean().round(2))\n"
         "df.loc[zero, 'league'].value_counts().to_frame('eksik 10.dk satırı')"),
    md("**Karar:** Bu 175 maçta maç sonu istatistikleri dolu, fakat 10. dk değerleri hiç kaydedilmemiş (0 = eksik). "
       "Gerçek bir 0 altın değeri mümkün olmadığı için bunlar **eksik veri** olarak kabul edilip çıkarıldı. "
       "Bu durum büyük ölçüde LPL ve DCGI maçlarında yoğunlaşıyor."),
    md("## 4. Temizlenmiş veri seti (Blue bakış açısı, maç başına 1 satır)"),
    code("data, report = clean(df)\nreport"),
    code("blue, red = df[df.side=='Blue'].set_index('game_id'), df[df.side=='Red'].set_index('game_id')\n"
         "print('golddiffat10 simetrik mi (Blue = -Red)?', (blue.golddiffat10 + red.golddiffat10 == 0).all())\n"
         "print('Blue killsat10 == Red deathsat10 oranı:', round((blue.killsat10 == red.deathsat10).mean(), 4))"),
    md("Fark sütunları iki satırda simetrik olduğundan yalnızca Blue satırı yeterlidir; Red satırı bilgi eklemez ve aynı maçın iki kez "
       "kullanılmasına (sızıntıya) yol açardı. Kill/death eşleşmesindeki küçük (%1) uyumsuzluk kaynak verideki kayıt farklarından gelir."),
    md("## 5. Hedef sınıf dağılımı"),
    code("ax = data.result.map({1: 'Blue kazandı', 0: 'Red kazandı'}).value_counts().plot.bar(color=['#0a6ebd', '#c62828'], rot=0)\n"
         "ax.set_title('Hedef dağılımı (temiz veri)'); ax.set_ylabel('Maç sayısı'); plt.tight_layout(); plt.savefig(FIG/'target_distribution.png', dpi=120); plt.show()\n"
         "data.result.value_counts(normalize=True).round(4)"),
    md("Blue tarafı ~%53.7 kazanıyor: hafif bir dengesizlik var ama sınıf ağırlıklandırması gerektirecek kadar değil. "
       "Referans (baseline) olarak \"her zaman Blue\" tahmini bu oranda doğruluk verir."),
    md("## 6. Özelliklerin dağılımı ve aykırı değerler"),
    code("data[FEATURES].describe().T.round(1)"),
    code("fig, axes = plt.subplots(1, 5, figsize=(18, 4))\nfor ax, f in zip(axes, FEATURES):\n    sns.boxplot(y=data[f], ax=ax, color='#c89b3c'); ax.set_title(f)\n"
         "plt.tight_layout(); plt.savefig(FIG/'boxplots.png', dpi=120); plt.show()\n"
         "q1, q3 = data[FEATURES].quantile(.25), data[FEATURES].quantile(.75); iqr = q3 - q1\n"
         "((data[FEATURES] < q1 - 1.5*iqr) | (data[FEATURES] > q3 + 1.5*iqr)).sum().to_frame('IQR aykırı sayısı')"),
    md("**Karar:** IQR yöntemine göre aykırı görünen değerler (ör. 10. dakikada +10.000 altın, 24 kill) gerçek tek taraflı maçlardır; "
       "ölçüm hatası değildir ve tahmin için en bilgilendirici örneklerdendir. Bu yüzden çıkarılmadı. Logistic Regression için "
       "`StandardScaler`, ağaç modelleri ise aykırılardan zaten az etkilenir."),
    md("## 7. Gold, XP ve CS farklarının galibiyetle ilişkisi"),
    code("fig, axes = plt.subplots(1, 3, figsize=(16, 4))\nfor ax, f in zip(axes, ['golddiffat10', 'xpdiffat10', 'csdiffat10']):\n"
         "    sns.histplot(data=data, x=f, hue=data.result.map({1:'Blue kazandı',0:'Red kazandı'}), bins=60, ax=ax, element='step', stat='density', common_norm=False,\n"
         "                 palette={'Blue kazandı':'#0a6ebd','Red kazandı':'#c62828'})\n    ax.set_title(f)\n"
         "plt.tight_layout(); plt.savefig(FIG/'diff_distributions.png', dpi=120); plt.show()"),
    code("bins = [-np.inf, -3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000, np.inf]\n"
         "wr = data.groupby(pd.cut(data.golddiffat10, bins), observed=True).result.agg(['mean', 'size'])\n"
         "ax = wr['mean'].plot.bar(color='#c89b3c', rot=45, figsize=(10, 4)); ax.axhline(.5, ls='--', c='k')\n"
         "ax.set_ylabel('Blue galibiyet oranı'); ax.set_title('10. dk altın farkına göre Blue galibiyet oranı')\n"
         "plt.tight_layout(); plt.savefig(FIG/'winrate_by_golddiff.png', dpi=120); plt.show()\nwr.round(3)"),
    md("Altın farkı arttıkça galibiyet oranı monoton biçimde artıyor: 10. dakikada +3000 altın önde olan Blue takımı maçların büyük çoğunluğunu kazanıyor."),
    md("## 8. Korelasyonlar"),
    code("corr = data[FEATURES + ['result']].corr()\nplt.figure(figsize=(7, 6)); sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)\n"
         "plt.title('Korelasyon matrisi'); plt.tight_layout(); plt.savefig(FIG/'correlation.png', dpi=120); plt.show()"),
    md("`golddiffat10`, `xpdiffat10` ve `csdiffat10` birbiriyle güçlü korelasyonlu (çoklu doğrusallık). Bu, ağaç modellerini ve tahmin gücünü "
       "bozmaz; ancak Logistic Regression katsayılarını tek tek yorumlarken dikkat edilmelidir."),
    md("## 9. Tarih ve patch dağılımı"),
    code("fig, axes = plt.subplots(1, 2, figsize=(16, 4))\ndata.date.dt.strftime('%Y-%m').value_counts().sort_index().plot.bar(ax=axes[0], color='#0a6ebd', rot=45)\n"
         "axes[0].set_title('Aylara göre maç sayısı')\n"
         "data.patch.map('{:.2f}'.format).value_counts().sort_index().plot.bar(ax=axes[1], color='#c89b3c'); axes[1].set_title('Patch dağılımı')\n"
         "plt.tight_layout(); plt.savefig(FIG/'date_patch.png', dpi=120); plt.show()"),
    code("data.league.value_counts().head(15).to_frame('maç')"),
    md("Veri 2026 Ocak–Kasım arasını kapsıyor. Meta değiştikçe (patch) ilişkiler değişebileceğinden modelde **kronolojik bölme** "
       "kullanıldı: eski %80 maç eğitim, en yeni %20 maç test. Böylece model gerçek kullanımdaki gibi \"geleceği\" tahmin ediyor."),
    md("## 10. Temizleme kararlarının özeti\n\n"
       "| Adım | Karar | Gerekçe |\n|---|---|---|\n"
       "| Maç başına satır | Yalnız Blue satırı | Fark sütunları simetrik; aynı maç iki kez girmez |\n"
       "| Tüm 10. dk değerleri 0 (175 maç) | Çıkarıldı | `goldat10=0` imkânsız → eksik kayıt |\n"
       "| Eksik değerli sütunlar (ban, split, bo_format…) | Kullanılmadı | Modelle ilgisiz |\n"
       "| Aykırı değerler | Tutuldu | Gerçek tek taraflı maçlar |\n"
       "| Maç sonu sütunları (kills, towers, gamelength…) | Kullanılmadı | 10. dakikada bilinmez → veri sızıntısı |\n"
       "| `game_id`, takım adları | Kullanılmadı | Kimlik bilgisi, genellenebilir değil |\n"
       "| 15. dk sütunları | Kullanılmadı | Problem 10. dk ile sınırlı |"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, Path(__file__).with_name("01_eda.ipynb"))
