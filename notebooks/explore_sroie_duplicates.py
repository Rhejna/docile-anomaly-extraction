import os
import json
import re
from src.data.loader import load_receipts

path = "data/raw/sroie/task2train"

all_files = os.listdir(path)

# Tous les noms qui contiennent une parenthèse
with_paren = [f for f in all_files if "(" in f]

problems = []

for f in with_paren:
    # Enlève l'extension
    name, ext = os.path.splitext(f)
    
    # Enlève le (1), (2), etc. pour retrouver le nom de base
    base_name = re.sub(r"\(\d+\)$", "", name)
    
    base_file = base_name + ext
    base_path = os.path.join(path, base_file)
    
    # 1. Le fichier de base existe-t-il ?
    if not os.path.exists(base_path):
        problems.append(f"PAS DE BASE : {f}  (base attendue : {base_file})")
        continue
    
    # 2. Si c'est un .txt, compare les 4 champs
    if ext.lower() == ".txt":
        try:
            with open(os.path.join(path, f), encoding="utf-8") as file1:
                data_dup = json.load(file1)
            with open(base_path, encoding="utf-8") as file2:
                data_base = json.load(file2)
            
            if data_dup != data_base:
                problems.append(f"CHAMPS DIFFÉRENTS : {f}")
        except Exception as e:
            problems.append(f"ERREUR LECTURE : {f} → {e}")

print(f"Nombre de fichiers avec parenthèse : {len(with_paren)}")
print(f"Cas problématiques trouvés : {len(problems)}")
print()

if problems:
    print("--- Problèmes ---")
    for p in problems:
        print(p)
else:
    print("Aucun problème. Tous les (1), (2)... sont de vrais doublons identiques.")


# Vérifie la forme des IDs sans parenthèse
ids_sans_paren = set()
for f in all_files:
    if "(" not in f:
        name = os.path.splitext(f)[0]
        ids_sans_paren.add(name)

print(f"\nNombre d'IDs sans parenthèse : {len(ids_sans_paren)}")
print("Exemples d'IDs normaux :")
for i in sorted(list(ids_sans_paren))[:10]:
    print(" ", i)


load_receipts(path)