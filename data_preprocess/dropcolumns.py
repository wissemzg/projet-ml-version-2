import pandas as pd


file_path = r"C:\Users\Extra\Desktop\antispoof\projet-ML-2\merged_all.csv"
df = pd.read_csv(file_path)

columns_to_drop = [
    "IND_RES",
    "source_file",
    "source_folder",
    "C_GR_RLC",
    "CODE_VAL",
    "LIB_VAL",
    "COURS_REF",
    "COURS_VEILLE",
    "DERNIER_COURS",
    "NB_TRAN",
    "I"
]


df = df.drop(columns=columns_to_drop, errors='ignore')


df.to_csv("cleaned_file.csv", index=False)

print("Columns dropped successfully!")
print("Remaining columns:")
print(df.columns)
