# DAG-Analysis

# 1. Create the directory
mkdir -p treebanks

# 2. Clone only the necessary files for each language
git clone --depth 1 https://github.com/SurfacesyntacticUD/SUD_English-GUM.git
mv SUD_English-GUM/*.conllu treebanks/en_train.conllu

git clone --depth 1 https://github.com/SurfacesyntacticUD/SUD_German-GSD.git
mv SUD_German-GSD/*.conllu treebanks/de_train.conllu

git clone --depth 1 https://github.com/SurfacesyntacticUD/SUD_Spanish-AnCora.git
mv SUD_Spanish-AnCora/*.conllu treebanks/es_train.conllu

git clone --depth 1 https://github.com/SurfacesyntacticUD/SUD_Hindi-HDTB.git
mv SUD_Hindi-HDTB/*.conllu treebanks/hi_train.conllu

git clone --depth 1 https://github.com/SurfacesyntacticUD/SUD_Turkish-Kenet.git
mv SUD_Turkish-Kenet/*.conllu treebanks/tr_train.conllu

# 3. Clean up the cloned folders
rm -rf SUD_English-GUM SUD_German-GSD SUD_Spanish-AnCora SUD_Hindi-HDTB SUD_Turkish-Kenet

python3 dag_analysis.py
