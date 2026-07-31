"""Knowledge distillation pipeline: Teacher (ViT-1D) → Student (CNN micro) → STM32."""

from models.Distillation.Teacher import (
    MODEL_NAME as TEACHER_MODEL_NAME,
    DISTILL_TEMP,
    build_teacher,
    train_teacher_loso,
)
from models.Distillation.Student import MODEL_NAME as STUDENT_MODEL_NAME
from models.Distillation.run_pipeline import run_pipeline

__all__ = [
    "TEACHER_MODEL_NAME",
    "STUDENT_MODEL_NAME",
    "DISTILL_TEMP",
    "build_teacher",
    "train_teacher_loso",
    "run_pipeline",
]
