LABELS = ["O", "B-COMPANY", "I-COMPANY", "B-ADDRESS", "I-ADDRESS", "B-DATE",  "I-DATE", "B-TOTAL", "I-TOTAL"]
label2id = {valeur: cle for cle, valeur in enumerate(LABELS)}
id2label = {cle: valeur for cle, valeur in enumerate(LABELS)}