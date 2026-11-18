# Sunum konuşma notları

Hedef süre: yaklaşık 10–12 dakika. İngilizce slaytlardaki sayıları okumak yerine soruyu, deneyi ve sonucu anlat.

## 1. Başlık — 30 saniye
Bu projede FreeCP adlı, görüntü ve metni birlikte kullanan bir segmentasyon yöntemini inceledim. Önce makalenin bir sonucunu yeniden ürettim; sonra sınıf seçiminin hatalarını anlamak için farklı müdahaleler denedim. Son aşamada Y2 adını verdiğimiz seçici bir sınıf kurtarma kuralını kontrollü olarak değerlendirdim. Küçük bir artış bulduk, ancak sonuçların belirsizliğini ve bağımsızlık sınırlarını özellikle ayırıyorum.

## 2. Araştırma sorusu — 40 saniye
Modelin doğru sınıfı tanıması ile doğru pikselleri bulması farklı sorunlar. Bir sınıf yanlışlıkla elenirse nesneyi kaybedebiliriz. Fakat onu geri almak da yanlış bir bölgeyi boyayabilir. Sorum şu: zaten hesaplanmış bir tanıma sinyaliyle, yeni görüntü kodlaması yapmadan, yanlışlıkla atılan sınıfları seçici olarak geri alabilir miyiz? CLIP ViT-B/16 ve SCLIP sabit tutuldu; model eğitilmedi.

## 3. Literatür — 45 saniye
Raporda altı merkezi çalışmayı, hocanın istediği yedi alanlı matrisle karşılaştırdım. MaskCLIP, SCLIP, ClearCLIP ve NACLIP yoğun görsel temsilleri iyileştiriyor. FreeCP hangi sınıf haritalarının kalacağına müdahale ediyor. Lin ve arkadaşlarının TagCLIP çalışması ise çok etiketli tanıma açısından yakın bir bağlantı sunuyor. Y2, görüntü etiketlerinden yararlanma fikrini ilk kez ortaya koyduğu iddiasında değil; FreeCP içindeki dar bir müdahalenin kontrollü deneyi.

## 4. FreeCP mekanizması — 55 saniye
Görüntü ve metin sorguları CLIP özelliklerine dönüştürülüyor. SCLIP yoğun haritalar üretiyor; refinement görsel benzerlikler üzerinden aktivasyonları yayıyor. RP gereksiz sınıfları eliyor; AP kalan belirsizlikleri ele alıyor. RP skorunun büyüklüğe duyarlı olduğunu gördük. Önemli bir reprodüksiyon ayrıntısı var: released kodun RP paydası ile makalenin soft-IoU denklemi aynı değil. Makalenin sonucunu yeniden üretmek için released kodu koruduk ve farkı raporda açıkça belirttik.

## 5. Reprodüksiyon — 35 saniye
Makale SCLIP tabanlı VOC21 için 65,8 bildiriyor. Biz aynı kapsamda 65,763253 mIoU ölçtük; yuvarlayınca aynı değer. Bu, makalenin bütün sonuçlarının değil, seçtiğimiz tek konfigürasyonun reprodüksiyonu. Kod revizyonu, ortam, model ve veri ayrımları kayıtlı. Son çalıştırmanın baseline matrisi, önceki çalıştırmanın 1449 görüntüsünün her birinde birebir aynı çıktı.

## 6. Veri ayrımı — 50 saniye
İlk 50 görüntü, 200 geliştirme görüntüsü ve 300 doğrulama görüntüsü VOC train içinden seçildi; birbirleriyle ve resmi val kümesiyle kesişmiyorlar. Ancak resmi val1449 önceki keşif deneylerinde incelendi. Bu yüzden son val sonucunu yeni ve bağımsız bir doğrulama saymıyoruz. K=21 ile K=124 çıktı sınıfı sayılarıdır; metin sorguları sırasıyla 46 ve 149. ADE'den yalnızca sınıf adları eklendi. VOC bu ek sınıfları tam etiketlemediği için gerçek varlıklarını bilinmiyor kabul ettik.

## 7. İlk denemeler — 50 saniye
RP eşiğini 0,15'ten 0,10'a indirmek ilk 50 görüntüde yaklaşık 0,726 puan kazandırdı. Fakat 1449 görüntüde 1,359 puan kaybettirdi. Kosinüs skoru büyüklük bilgisini kaldırınca, görüntüde olmayan sınıfların geçiş oranı yaklaşık yüzde 1,4'ten yüzde 60,7'ye yükseldi ve sonuç ciddi biçimde düştü. Bu, büyüklüğün yararlı olduğunu destekliyor; RP'nin yalnızca güven filtresi olduğunu kanıtlamıyor. Arka plan sorgularını değiştirmek de sonuçları tutarsız biçimde kaydırdı.

## 8. PAMR ve ikinci bakış — 55 saniye
200 geliştirme görüntüsünde PAMR yaklaşık sıfır mIoU farkına karşı yüzde 41 ek süre getirdi. İkinci bakış, 221 adaydan 11'ini kabul etti; bunların 6'sında sınıf gerçekten vardı, 5'inde yoktu. 442 ek kırpıntı kodlaması yapıldı. Sorun yalnızca sınıfın varlığı değil: doğru kabul edilen sınıfın geri gelen haritası yanlış yere yayılabiliyor. Birleştirilmiş yöntem de çözüm olmadı. Sonradan bulunan yaklaşık 0,194 puanlık eşik iyileşmesi ise tek bir kuş görüntüsü çıkarılınca yaklaşık 0,008'e indi.

## 9. Y2 kuralı — 55 saniye
Her pencere için CLIP'in zaten hesapladığı sınıf skoruna, tüm sorgular üzerinden softmax uyguluyoruz. RP sınıfı atsa bile olasılığı en az 0,50 ise mevcut refined haritayı silmiyoruz ve sınıfı değişmeyen AP aşamasına veriyoruz. RP'nin zaten tuttuğu sınıflar korunuyor. Yeni görüntü kodlaması yok, ama AP yükü değişebilir. Eşik, geliştirme kümesinde mIoU'yu artıracak şekilde değil, sınıf kurtarma precision/recall ölçütleriyle seçildi. Her pencerede aynı sayıda sınıfı rastgele kurtaran kontrol de var.

## 10. Tanı bulguları — 40 saniye
CLIP'in genel varlık AUROC'u RP'den daha iyi değildi; birleşik genel skor da iyileşmedi. Buna rağmen yüksek olasılıklı küçük bir bölgede kurtarma hassasiyeti iyiydi: orijinal vocab için 49 doğru ve 7 yanlış, geniş vocab için 29 doğru ve 6 yanlış VOC sınıf olayı. Ek sınıfların varlığı bilinmiyor. Bu, Y2'yi denemeye değer kıldı; segmentasyonda kesin işe yarayacağını göstermedi.

## 11. Bağımsız doğrulama — 60 saniye
Birincil koşul geniş vocab'dı: Y2'nin farkı yaklaşık artı 0,289 puan, yüzde 95 paired bootstrap aralığı eksi 0,113 ile artı 0,844. Orijinal vocab için artı 0,158 ve aralık eksi 0,112 ile artı 0,663. İki aralık da sıfırı kapsıyor. Y2-random farklarının aralıkları da sıfırı kapsıyor. Her iki vocab'da yalnızca 16/300 maske değişti. Geniş vocab artışı, etkili tek bir bitki görüntüsünü çıkarınca yaklaşık eksi 0,005 oluyor. Dolayısıyla bağımsız ve dayanıklı bir üstünlük kanıtlandığını söylemiyorum.

## 12. Ek benchmark — 45 saniye
Resmi val1449 üzerinde baseline 65,763, Y2 66,024 çıktı. Fark artı 0,260; betimsel bootstrap aralığı artı 0,013 ile artı 0,507. Tek tek görüntüler çıkarıldığında fark pozitif kalıyor. Fakat bu benchmark daha önce görüldü; güven aralığı, adaptif araştırma sürecinin yanlılığını düzeltmez. Zaman ve bellek yakın. Ardışık tek ölçümden bir hızlanma iddiası çıkarmıyorum.

## 13. İyi ve kötü örnek — 50 saniye
Burada gerçek kaydedilmiş tahmin maskeleri var; RGB fotoğraf veya ground truth değil. Beyaz alanlar değişen pikselleri gösteriyor. Tekne görüntüsünde net 25.676 doğru piksel kazanılıyor. Kuş görüntüsünde kuş piksellerinde biraz kazanç olsa da arka planın yanlış boyanması nedeniyle net 38.858 doğru piksel kaybediliyor. Bu örnekler leave-one-out etkilerine göre seçildi. Doğru sınıfı kurtarmak ile doğru alanı kurtarmak arasındaki fark burada görünür.

## 14. Sonuç — 45 saniye
Temel sonuç başarılı bir reprodüksiyon ve hangi müdahalenin neden sorun çıkardığına ilişkin kayıtlı bir açıklama. Y2 ucuz ve seçici, ancak bağımsız doğrulamadaki belirsizlik devam ediyor. Y1 eşanlamlı gruplama, saf güven ve saf şekil ablation'ları, hiç kullanılmamış bir veri seti ve birden fazla random kontrol tohumu gelecek çalışma. Bunlar yapılmış deneyler değil. ChatGPT/Codex araştırma, kod, test, analiz ve yazımda destek verdi; Colab deneylerini ben çalıştırdım ve çalışmanın doğruluğundan ben sorumluyum.

# Muhtemel sorular ve kısa cevaplar

1. **Bu neden multimodal?** Görüntü özellikleri ile metin sınıf sorguları aynı tahmin hattında karşılaştırılıyor. Canlı bir LLM sohbeti gerekmiyor.
2. **Gerçekten yeni bir model eğittiniz mi?** Hayır. Frozen modele inference-time bir sınıf kurtarma kuralı ekledik.
3. **Y2 neden özgün?** İddiamız büyük bir yeni mimari değil; released FreeCP içindeki belirli eleme kararına, mevcut pencere tanıma sinyaliyle sınırlı bir müdahale ve kontrollü değerlendirme.
4. **0,50 eşiğini test sonucuna göre mi seçtiniz?** Hayır. Train'den ayrılmış 200 geliştirme görüntüsündeki varlık kurtarma ölçütleriyle, önceden tanımlanmış grid üzerinden seçildi.
5. **Yüzde 95 aralık sıfırı kapsıyorsa ne diyebilirsiniz?** Nokta tahmini olumlu, fakat bu örnekleme altında sıfır ya da küçük negatif farkla da uyumlu. Güçlü üstünlük iddiası yok.
6. **1449 görüntüde aralık pozitifken neden temkinlisiniz?** Bu benchmark önceki deneylerde incelendi. Bootstrap aralığı test setine adaptasyonu ortadan kaldırmaz.
7. **mIoU nasıl hesaplandı?** Her görüntünün confusion matrix sayıları toplanır; her sınıfın toplam kesişimi toplam birleşime bölünür, sınıflar ortalanır. Görüntü IoU'larının basit ortalaması alınmaz.
8. **Random kontrol adil mi?** Aynı pencerede Y2 ile aynı sayıda RP-rejected sınıf geri alınıyor. Bir sabit random seed kullanılması bir sınırlama.
9. **Ek ADE sınıflarına neden false positive demediniz?** VOC onları eksiksiz etiketlemiyor. Varlıkları bilinmiyor; segmentasyon değerlendirmesinde tam argmax sonrası ek etiketler arka plana eşleniyor.
10. **PAMR neden başarısız?** Bu backbone, veri ve ayarda kalite kazancı göstermedi ve maliyet ekledi. Genel olarak PAMR'nin faydasız olduğu sonucuna varmıyoruz.
11. **Cosine sonucu neyi kanıtlıyor?** Normalize edilmiş şekil benzerliği tek başına çok sayıda olmayan sınıfı tutuyor. Büyüklük bilgisinin yararlı olduğuna işaret ediyor; güven ve tutarlılığın nedensel ayrıştırması henüz yapılmadı.
12. **Hoca sonucu nasıl doğrulayabilir?** CPU mini kontrolü, paketlenmiş matrislerden skorları ve paired karşılaştırmaları yeniden hesaplar; tam Colab komutları model deneyini üretir. Büyük ham loglar yerine seçilmiş kanıtlar paketlenmiştir.
13. **Konferans makalesi hazır mı?** Hayır; daha güçlü bağımsız kanıt, farklı veri setleri ve daha kapsamlı kontroller gerekir. Mevcut kapsam, dikkatli bir ders projesi ve devam araştırması temeli.
