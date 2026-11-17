from typing import Literal
from pydantic import BaseModel, Field
class StudentFeatures(BaseModel):
 """Données envoyées par le client : le profil d'un étudiant."""
 reading_score: int = Field(
 ..., ge=0, le=100, description="Note de lecture (0 à 100)", examples=[85]
 )
 writing_score: int = Field(
 ..., ge=0, le=100, description="Note d'écriture (0 à 100)", examples=[82]
 )
 parental_level_of_education: Literal[
 "some high school",
 "high school",
 "some college",
 "associate's degree",
 "bachelor's degree",
 "master's degree",
 ] = Field(..., description="Niveau d'études des parents", examples=["bachelor's degree"])
 test_preparation_course: Literal["none", "completed"] = Field(
 ..., description="Cours de préparation à l'examen suivi ?", examples=["completed"]
 )
 gender: Literal["female", "male"] = Field(..., examples=["female"])
 race_ethnicity: Literal["group A", "group B", "group C", "group D", "group E"] = Field(
 ..., description="Groupe (anonymisé) de l'étudiant", examples=["group D"]
 )
 lunch: Literal["standard", "free/reduced"] = Field(
 ..., description="Type de repas", examples=["standard"]
 )
class PredictionResponse(BaseModel):
 """Réponse renvoyée par l'API."""
 class_id: int
 class_name: str
 probabilities: dict[str, float]