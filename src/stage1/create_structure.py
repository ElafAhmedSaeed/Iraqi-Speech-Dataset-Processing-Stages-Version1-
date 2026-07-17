# ماذا يفعل هذا الكود؟
# يحدد مجلد المشروع الرئيسي.
# يقرأ قائمة المجلدات المطلوبة.
# ينشئ أي مجلد غير موجود.
# لا يحذف أي شيء موجود.
# إذا شغلته مرة ثانية فلن يحدث أي خطأ (exist_ok=True).

from pathlib import Path

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# All folders required for the project
FOLDERS = [
    "config",

    "data/raw/audio",
    "data/raw/transcripts",
    "data/raw/vocabulary",
    "data/raw/metadata",

    "data/organized",
    "data/validated",
    "data/cleaned",
    "data/normalized",
    "data/processed_audio",
    "data/segmented",
    "data/aligned",
    "data/lexicon",

    "metadata",

    "reports",

    "logs",

    "tests"
]


def create_project_structure():
    """
    Create the complete project folder structure.
    """

    print("\nCreating project structure...\n")

    for folder in FOLDERS:

        path = PROJECT_ROOT / folder

        path.mkdir(parents=True, exist_ok=True)

        print(f"Created: {path}")

    print("\nProject structure created successfully.")


if __name__ == "__main__":
    create_project_structure()