<!--- Banner -->
<br />
<p align="center">
<a href="#"><img src="https://www.tclf.org/sites/default/files/styles/crop_2000x700/public/thumbnails/image/CA_Stanford_StanfordUniversity_courtesyWikimediaCommons_2011_005_Hero.jpg?itok=B8YAapxD"></a>
<h3 align="center">Finetuning BERT for Natural Language Inference with PyTorch</h3>
<p align="center">A PyTorch project leveraging HuggingFace transformers to classify textual entailment on the Stanford NLI corpus.</p>

<!--- About --><br />
## About
This repo offers a PyTorch and HuggingFace-based toolkit for Natural Language Inference (NLI), a.k.a. Recognizing Textual Entailment (RTE).
The [Stanford NLI corpus](https://www.kaggle.com/datasets/stanfordu/stanford-natural-language-inference-corpus) is used, featuring 570k human-written English sentence pairs each labeled as 'entailment', 'contradiction', or 'neutral'.

<!--- Models --><br />
## Models
The main model is a BERT transformer finetuned on the task.
Other models are also available for experimentation:
* Pooled Logistic Regression
* Shallow Neural Network
* Deep Neural Network

Each model is fine-tuned to work with token embeddings and gives classification scores for the three NLI labels.

<!--- Usage --><br />
## Usage
Run the main script with various command-line arguments to specify the model, number of epochs, and other settings.

```python
python main.py --model=shallow --epochs=5 --device=cuda --batch_size=64 --embedding_dim=128
```

<!--- Built with... --><br />
## Built with...
* [PyTorch](https://pytorch.org/) — model building and training
* [scikit-learn](https://scikit-learn.org/stable/) — for additional machine learning utilities
