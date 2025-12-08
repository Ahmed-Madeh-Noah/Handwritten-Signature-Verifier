Development Commands

```ps1
cd ./src/
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python the_file_to_run.py

pip freeze > requirements.txt # Please remove torch and torchvision from requirements.txt
```