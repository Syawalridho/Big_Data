import time
import os
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from datetime import datetime

options = webdriver.ChromeOptions()
options.add_argument("--headless")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://www.worldometers.info/world-population/")

file_name = "data_fixed.csv"

# Cek apakah file sudah ada
file_exists = os.path.exists(file_name)

# Buka file dengan mode append ('a') untuk menambahkan data tanpa menghapus
with open(file_name, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Tulis header kalau file belum ada
    if not file_exists:
        writer.writerow(["timestamp", "populasi", "kelahiran", "kematian"])

    while True:
        try:
            population_str = driver.find_element(By.XPATH, "//span[@rel='current_population']").text
            born_str = driver.find_element(By.XPATH, "//span[@rel='births_today']").text
            death_str = driver.find_element(By.XPATH, "//span[@rel='dth1s_today']").text

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            writer.writerow([current_time, population_str, born_str, death_str])
            print(f"✅ Data berhasil disimpan pada {current_time}")
            print(f"   Populasi: {population_str}, Kelahiran: {born_str}, Kematian: {death_str}\n")

        except Exception as e:
            print("❌ Gagal mengambil data:", e)

        time.sleep(1)  # Tunggu 1 detik sebelum mengambil data lagi

driver.quit()
