# Image Captioning — README

Ce projet regroupe les différentes expériences menées autour de la génération automatique de légendes d’images (*image captioning*) utilisant diverses architectures encodeur–décodeur : CNN/(LSTM ou Transformer), VGG/(LSTM,Transformer), ViT et un modèle VLM.


### Fichiers

Les fichiers ci-dessous contiennent les entraînements, tests et visualisations des modèles :

| Notebook | Description |
|---------|-------------|
| `cnn_lstm_transformer.ipynb` | CNN léger (from scratch) comme encodeur + LSTM ou Transformer en décodeur. |
| `vgg_lstm.ipynb` | VGG-16 gelé → MLP → LSTM. |
| `ft_vgg_lstm.ipynb` | Version fine-tunée du modèle précédent (derniers blocs VGG entraînés). |
| `vgg_transformer.ipynb` | VGG-16 gelé → MLP → Transformer Decoder. |
| `ft_vgg_transformer.ipynb` | Version fine-tunée du modèle VGG → Transformer. |
| `vit.ipynb` | Encodeur Vision Transformer. (ajouté par rapport à la soutenance mais pas présent dans le rapport)|
| `script_eval.ipynb` | Évaluations BLEU, ROUGE-L, METEOR, CIDEr, BERTScore. |
| `vlm.py` |  Script d'inférence du vlm sur l'ensemble de test |
| `captions_gen/` | Contient les légendes générées automatiquement par toutes les architectures testées. Sert à comparer la qualité des générations. |
|`evaluations/`| Contient l'évaluation des différentes architctures sur l'ensemble de test.
---
