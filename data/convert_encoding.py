import csv

in_path = r'C:\Users\USER\Desktop\2025 J-IGACS\受領データ\★eGFRありデータ\00 raw data.csv'
out_path = r'C:\Users\USER\Desktop\2025 J-IGACS\受領データ\★eGFRありデータ\00_raw_data_utf8.csv'

with open(in_path, 'r', encoding='shift_jis', errors='replace') as infile, \
     open(out_path, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        writer.writerow(row)

print("File converted to UTF-8")
