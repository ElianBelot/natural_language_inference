<!--- Banner -->
<br />
<p align="center">
<a href="#"><img src="https://www.tclf.org/sites/default/files/styles/crop_2000x700/public/thumbnails/image/CA_Stanford_StanfordUniversity_courtesyWikimediaCommons_2011_005_Hero.jpg?itok=B8YAapxD"></a>
<h3 align="center">Natural Language Inference with PyTorch</h3>
<p align="center">A PyTorch implementation for classifying textual entailment on the Stanford NLI corpus.</p>

<!--- About --><br />
## About
This repository provides a PyTorch-based solution for the task of Natural Language Inference (NLI), also known as Recognizing Textual Entailment (RTE). It uses the [Stanford Natural Language Inference corpus](https://www.kaggle.com/datasets/stanfordu/stanford-natural-language-inference-corpus) as the underlying dataset, comprising 570k human-written English sentence pairs that are manually labeled for balanced classification with labels: entailment, contradiction, and neutral.

<!--- Architecture --><br />
## Architecture
Three different models are available for experimentation:
* Pooled Logistic Regression
* Shallow Neural Network
* Deep Neural Network

Each model is designed to operate on token embeddings and provide classification scores for 'entailment', 'contradiction', and 'neutral' labels.
The models, data processing, and tokenizing are all built from scratch for educational purposes.

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
