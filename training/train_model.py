from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "StudentsPerformance.csv"
MODEL_PATH = ROOT / "models" / "students_model.joblib"

# On prédit « reussite » si la note de maths est supérieure ou égale à ce seuil
SEUIL_REUSSITE = 60

# 1. Variables NUMÉRIQUES : on les met à la même échelle (StandardScaler)
NUM_COLS = ["reading_score", "writing_score"]

# 2. Variables ORDINALES : l'ordre compte, on l'écrit explicitement
#    (du plus faible au plus fort)
ORDINAL = {
    "parental_level_of_education": [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ],
    "test_preparation_course": ["none", "completed"],
}
ORD_COLS = list(ORDINAL)

# 3. Variables NOMINALES : aucun ordre naturel, on utilise le One-Hot
NOM_COLS = ["gender", "race_ethnicity", "lunch"]


def load_data():
    """Lit le fichier CSV et renvoie les variables d'entrée X et la cible y (en texte)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}\n"
            "Placez StudentsPerformance.csv dans le dossier data/ du projet."
        )

    df = pd.read_csv(DATA_PATH)
    # « race/ethnicity » -> « race_ethnicity », « math score » -> « math_score », etc.
    df.columns = [
        c.strip().lower().replace("/", "_").replace(" ", "_") for c in df.columns
    ]

    X = df[NUM_COLS + ORD_COLS + NOM_COLS].copy()

    # Garde-fou : une modalité absente de notre liste serait encodée par -1
    # sans aucune erreur
    for col, order in ORDINAL.items():
        inconnues = set(X[col].unique()) - set(order)
        assert not inconnues, f"Modalités inattendues dans {col} : {inconnues}"

    # La note de maths sert uniquement à fabriquer la cible : elle n'est PAS dans X
    y_text = (df["math_score"] >= SEUIL_REUSSITE).map(
        {True: "reussite", False: "echec"}
    )
    return X, y_text


def main():
    # 1. Charger les données
    X, y_text = load_data()

    # 2. Encoder la cible : « echec » -> 0, « reussite » -> 1 (ordre alphabétique)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    codes = {
        c: int(i)
        for c, i in zip(
            label_encoder.classes_,
            label_encoder.transform(label_encoder.classes_),
        )
    }
    print(f"{len(X)} étudiants. Cible encodée : {codes}")

    # 3. Séparer 80 % pour l'entraînement, 20 % pour le test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Prétraitement : un traitement différent pour chaque groupe de colonnes
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            (
                "ord",
                Pipeline(
                    [
                        (
                            "encode",
                            OrdinalEncoder(
                                categories=[ORDINAL[c] for c in ORD_COLS],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                ORD_COLS,
            ),
            ("nom", OneHotEncoder(handle_unknown="ignore"), NOM_COLS),
        ]
    )

    # 5. Pipeline complet : prétraitement puis régression logistique
    model = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)

    # 6. Mesurer la précision sur des données jamais vues
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Précision sur le jeu de test : {accuracy:.3f}")

    # 7. Sauvegarder le pipeline ET l'encodeur de la cible dans un seul fichier
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "label_encoder": label_encoder}, MODEL_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    main()