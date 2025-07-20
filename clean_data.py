# save as clean_data.py

def clean_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf8') as out:
        for line in lines:
            line = line.strip()
            if not line:
                out.write('\n')  # Sentence boundary
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) == 3:
                word, lang, pos = parts
                out.write(f"{word}\t{pos}\n")

print("Cleaning dataset...")
clean_dataset('FB_BN_EN_FN-POS.txt', 'cleaned_data.txt')
print("Done. File saved as cleaned_data.txt ✅")


