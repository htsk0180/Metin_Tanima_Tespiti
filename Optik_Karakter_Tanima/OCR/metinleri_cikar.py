# Bu kod, ICDAR_2015 verisinden ya da bu veriyle aynı biçimde gerçek referans değeri işaretlemesi
# yapılmış verilerden, her bir resim için metin parçalarının çıkarılmasını ve başka bir konuma
# hem metin resimlerinin, hem de bu metin resimlerinin içeriği olan metinlerin yazılmasını sağlar.
# Oluşturulacak olan klasör formatı, Metin Tanıma repository'si tarafından lmdb adı verilen bir,
# veriye hızlı erişmek için kullanılan yapının oluşturulması için kullanılır.

import os
import sys
import math

import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def gercek_referans_degerleri_oku(referans_dosyası):
    dortgenler = []
    metinler = []
    with open(referans_dosyası, 'r') as f_gt:
        for line in f_gt:
            parcalar = line.rstrip('\n').lstrip('\ufeff').split(',')
            metin = ','.join(parcalar[8:])
            if metin == '###':  # eger okunamayan bir metinse bu veri setinde ### ile isaretlenmis
                continue

            dortgen = np.array([float(koordinat) for koordinat in parcalar[:8]]).reshape((4, 2))
            # sol ve saga denk gelen x ve y koordinatlarını bul
            x_sol = np.min(dortgen[:, 0])
            x_sag = np.max(dortgen[:, 0])
            y_ust = np.min(dortgen[:, 1])
            y_alt = np.max(dortgen[:, 1])
            dortgenler.append((x_sol, y_ust, x_sag, y_alt))
            metinler.append(metin)
    return np.array(dortgenler), metinler


def metinleri_cikar_kaydet(veriseti_konumu, sonuc_klasoru, resim_klasor_ismi="train_img", referans_klasor_ismi="train_gt"):
    resim_klasoru_konumu = os.path.join(veriseti_konumu, resim_klasor_ismi)
    referans_klasoru_konumu = os.path.join(veriseti_konumu, referans_klasor_ismi)
    resim_dosyasi_isimleri = sorted(os.listdir(resim_klasoru_konumu))

    sonuc_referans_dosyasi = os.path.join(sonuc_klasoru, resim_klasor_ismi + '_gt.txt')
    sonuc_resimler_klasoru = os.path.join(sonuc_klasoru, resim_klasor_ismi)
    os.makedirs(sonuc_resimler_klasoru, exist_ok=True)

    toplam_resim = 0
    toplam_metin = 0
    for i, resim_dosyasi_ismi in enumerate(tqdm(resim_dosyasi_isimleri)):
        resim_dosyasi = os.path.join(resim_klasoru_konumu, resim_dosyasi_ismi)
        referans_dosyasi = os.path.join(referans_klasoru_konumu, "gt_" + resim_dosyasi_ismi[:-4] + '.txt')
        # resime ait metinlerin bulunduğu pixelleri ve metinleri oku
        dortgenler, metinler = gercek_referans_degerleri_oku(referans_dosyasi)
        if len(metinler) == 0:  # resimde okunabilir bir metin yoksa resmi gec
            continue
        resim = Image.open(resim_dosyasi).convert('RGB')
        toplam_resim += 1
        toplam_metin += len(metinler)
        # her bir gerçek referans metin değeri için, metnin bulunduğu alanı resimden kes ve ayrı bir resim olarak kaydet
        for j, ((x_sol, y_ust, x_sag, y_alt), metin) in enumerate(zip(dortgenler, metinler)):
            metin_parcasi = resim.crop((x_sol, y_ust, x_sag, y_alt))
            metin_resmi_kayit_ismi = '{}_{}_{}.jpg'.format(resim_klasor_ismi, resim_dosyasi_ismi[:-4], j)
            metin_parcasi.save(os.path.join(sonuc_resimler_klasoru, metin_resmi_kayit_ismi), 'JPEG')
            with open(sonuc_referans_dosyasi, 'a') as f:
                f.write('{}/{}\t{}\n'.format(resim_klasor_ismi, metin_resmi_kayit_ismi, metin))
    print('toplam resim:', toplam_resim, 'toplam okunabilir metin:', toplam_metin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metin Cikarma")

    parser.add_argument("--veriseti_klasoru", help="Metinleri cikarilacak verinin konumu)",
                        type=str, default="./ICDAR_2015/")
    parser.add_argument("--sonuc_klasoru", help="Cikarilan metinlerin kaydedilecegi klasor", type=str,
                        default="./ICDAR_2015_metinler")
    args = parser.parse_args()

    veriseti_konumu = args.veriseti_klasoru
    sonuc_klasoru = args.sonuc_klasoru

    pre_run = False
    if os.path.isdir(sonuc_klasoru):  # sonuc klasoru onceden olusturulmussa
        if len(os.listdir(sonuc_klasoru)) != 0:  # sonuc klasoru bos degilse
            print('Metin çıkarma islemi onceden tamamlanmış! Bir kez daha cikarmak icin sonuc klasorunu silin, ya da',
                'baska bir sonuc klasor konumu secin.')
            pre_run = True
    if not pre_run:
        veriseti_klasorler = os.listdir(veriseti_konumu)
        if "train_img" not in veriseti_klasorler or "train_gt" not in veriseti_klasorler:
            print("Hata.. Verisetinin klasorunde resimler, 'train_img' klasorunun icinde olmali, gercek referans degerler ise"
                  " 'train_gt' klasorunde olmali. Klasor yapisini gozden gecirin.")
            sys.exit(-2)
        # sonuc klasoru yoksa olustur
        if not os.path.isdir(sonuc_klasoru):
            os.mkdir(sonuc_klasoru)

        metinleri_cikar_kaydet(veriseti_konumu, sonuc_klasoru, resim_klasor_ismi="train_img", referans_klasor_ismi="train_gt")
        metinleri_cikar_kaydet(veriseti_konumu, sonuc_klasoru, resim_klasor_ismi="test_img", referans_klasor_ismi="test_gt")

