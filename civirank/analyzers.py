import pandas as pd
from lexicalrichness import LexicalRichness
from sentence_transformers import SentenceTransformer, util
import torch
from huggingface_hub import hf_hub_download

import numpy as np
import os
import fasttext
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


class LexicalDensityAnalyzer():
    def __init__(self):
        pass

    def get_mtld(self, text):
         # copying approach from here https://github.com/notnews/unreadable_news
        assert type(text) in [str, pd.core.frame.DataFrame]
        if type(text) == str:
            lex = LexicalRichness(text)
            try:
                return lex.mtld()
            except ZeroDivisionError:
                return np.nan
        else:
            densities = []
            for i, row in text.iterrows():
                if row["lang"] != "en":
                    densities.append(np.nan)
                else:
                    lex = LexicalRichness(row["text"])
                    try:
                        densities.append(lex.mtld())
                    except ZeroDivisionError:
                        densities.append(np.nan)
            return densities

class TrustworthinessAnalyzer():
    '''
        Class that loads a trustworthiness analyzer using the domain scores from https://doi.org/10.1093/pnasnexus/pgad286, avaiable at https://github.com/hauselin/domain-quality-ratings. It exposes a function to calculate the trustworthiness of links contained in a post. It returns a single floating point value between 0 and 1 as trustworthiness score. A higher value means a more trustworthy link. If multiple links are contained in a post and indexed in the NewsGuard data base, the average trustworthiness rating is returned.
    '''
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        fname = "domain_pc1.csv"
        filepath = os.path.join(current_dir, 'data', 'domain_pc1.csv')

        # print current working directory
        self.scores = pd.read_csv(filepath, usecols=["domain", "pc1"])
        self.scores = self.scores.set_index("domain")

    def extract_scores(self, domains):
        # no domains contained in text? 
        if domains != domains:
            return np.nan
        else:
            ratings = []
            for domain in domains:
                if domain in self.scores.index:
                    rating = self.scores.loc[domain]["pc1"]
                    ratings.append(rating)

            # domains contained in text but they are not news rated by NG?
            if len(ratings) == 0:
                return np.nan
                
            # domain(s) rated by NG contained in text: return average rating of
            # all rated domains
            else:
                return np.mean(ratings)
            
    def get_trustworthiness_scores(self, domains):
        assert type(domains) in [list, pd.core.frame.DataFrame]
        if type(domains) == list:
            return self.extract_scores(domains)
        else:
            scores = []
            for d in domains["domain"]:
                scores.append(self.extract_scores(d))
            return scores

class ToxicityAnalyzer():
    def __init__(self, model_name="protectai/unbiased-toxic-roberta-onnx", file_name='model.onnx', gpu_id=0):
        # Initialize the ONNX model and tokenizer with the specified model name
        self.model = ORTModelForSequenceClassification.from_pretrained(model_name, file_name=file_name, provider="CUDAExecutionProvider", provider_options={'device_id': gpu_id})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Find the index of the 'toxicity' label
        self.toxicity_index = None
        for idx, label in self.model.config.id2label.items():
            if label.lower() == 'toxicity':
                self.toxicity_index = idx
                break
        if self.toxicity_index is None:
            raise ValueError("Toxicity label not found in model's id2label mapping.")

    def classify_texts(self, texts):
        """ Tokenize and classify a batch of texts """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)

        probabilities = torch.sigmoid(outputs.logits)

        batch_results = []
        for prob in probabilities:
            result = prob[self.toxicity_index].item()  # Get the probability for the 'toxicity' label
            batch_results.append(result)

        return batch_results

    def get_toxicity_scores(self, text, batch_size=8):
        """ Analyze the given text or DataFrame and return toxicity scores """
        assert isinstance(text, (str, pd.DataFrame)), "Input should be either a string or a DataFrame"

        if isinstance(text, str):
            results = self.classify_texts([text])
            return results[0]  # Return the score for the single input string
        else:
            results = []
            for start in range(0, len(text), batch_size):
                end = start + batch_size
                batch_texts = text["text"].iloc[start:end].tolist()
                batch_results = self.classify_texts(batch_texts)
                results.extend(batch_results)

            return results

class PolarizationAnalyzer():
    def __init__(self, model = 'joaopn/glove-model-reduced-stopwords', local_model = False, label_filter = 'issue'):
        # Initialize the model
        if local_model:
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, 'data', model)
        else:
            model_path = model
        self.model = SentenceTransformer(model_path, device="cuda")
        self.batch_size = 1024
        # Load the polarization terms and compute their embeddings
        self.label_filter = label_filter
        self.load_and_embed_terms()
        
    def load_and_embed_terms(self):
        # Load terms from CSV
        current_dir = os.path.dirname(__file__)
        fname = "polarization_dictionary.csv"
        filepath = os.path.join(current_dir, 'data', fname)
        df = pd.read_csv(filepath, header=0)
        if self.label_filter is not None:
            df = df[df['label'] == self.label_filter]
        unique_words = df['word'].unique()
        
        # Compute embeddings for the unique words
        self.dict_embeddings = self.model.encode(
            list(unique_words),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        
        # Average the embeddings to create a single dictionary embedding
        self.dict_embeddings = torch.mean(self.dict_embeddings, dim=0)

    def preprocess(self, df):
        # Regular expressions to clean up the text data
        df["text"] = df["text"].replace(
            to_replace=[r"(?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})"],
            value=[""], 
            regex=True,
        )
        df["text"] = df["text"].replace(to_replace=r"&.*;", value="", regex=True)
        df["text"] = df["text"].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True) 
        df["text"] = df["text"].replace(to_replace=r"\s+", value=" ", regex=True)
        df["text"] = df["text"].replace(to_replace=r"\@\w+", value="@user", regex=True)

    def get_embeddings(self, df):
        # Encode text in batches
        corpus_embeddings = self.model.encode(
            list(df["text"]),
            batch_size=self.batch_size,
            show_progress_bar=False, 
            convert_to_tensor=True
        ) 

        assert len(corpus_embeddings) == len(df)
        return corpus_embeddings
    
    def compute_similarity(self, text_embeddings):
        # Calculate cosine similarity between text embeddings and dictionary embeddings
        cos_sim = util.cos_sim(text_embeddings, self.dict_embeddings)
        return cos_sim
    
    def get_similarity(self, texts):
        df = texts.copy()
        self.preprocess(df)
        text_embeddings = self.get_embeddings(df)
        cos_sim = self.compute_similarity(text_embeddings)
        return cos_sim.cpu().numpy()

class ProsocialityAnalyzer():
    def __init__(self, model = 'joaopn/glove-model-reduced-stopwords', local_model = False):
        # Initialize the model
        if local_model:
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, 'data', model)
        else:
            model_path = model
        self.model = SentenceTransformer(model_path, device="cuda")
        self.batch_size = 1024
        # Load the polarization terms and compute their embeddings
        self.load_and_embed_terms()


    def load_and_embed_terms(self):
        # Load terms from CSV
        current_dir = os.path.dirname(__file__)
        fname = "prosocial_dictionary.csv"
        filepath = os.path.join(current_dir, 'data', fname)
        prosocial_dict = pd.read_csv(filepath, header=None, names = ['word'])
        prosocial_dict["word"] = prosocial_dict["word"].str.replace("*", "")
        prosocial_dict["word"] = prosocial_dict["word"].str.replace("nt", "not")
        prosocial_dict = list(prosocial_dict["word"].values)

        # Compute embeddings for the unique words
        self.dict_embeddings = self.model.encode(
            prosocial_dict,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        
        # Average the embeddings to create a single dictionary embedding
        self.dict_embeddings = torch.mean(self.dict_embeddings, dim=0)

    def preprocess(self, df):
        # Regular expressions to clean up the text data
        df["text"] = df["text"].replace(
            to_replace=[r"(?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})"],
            value=[""], 
            regex=True,
        )
        df["text"] = df["text"].replace(to_replace=r"&.*;", value="", regex=True)
        df["text"] = df["text"].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True) 
        df["text"] = df["text"].replace(to_replace=r"\s+", value=" ", regex=True)
        df["text"] = df["text"].replace(to_replace=r"\@\w+", value="@user", regex=True)

    def get_embeddings(self, df):
        # Encode text in batches
        corpus_embeddings = self.model.encode(
            list(df["text"]),
            batch_size=self.batch_size,
            show_progress_bar=False, 
            convert_to_tensor=True
        ) 

        assert len(corpus_embeddings) == len(df)
        return corpus_embeddings
    
    def compute_similarity(self, text_embeddings):
        # Calculate cosine similarity between text embeddings and dictionary embeddings
        cos_sim = util.cos_sim(text_embeddings, self.dict_embeddings)
        return cos_sim
    
    def get_similarity(self, texts):
        df = texts.copy()
        self.preprocess(df)
        text_embeddings = self.get_embeddings(df)
        cos_sim = self.compute_similarity(text_embeddings)
        return cos_sim.cpu().numpy()

class LanguageAnalyzer():
    def __init__(self):
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.model = fasttext.load_model(model_path)

    def detect_language(self, text):
        return self.model.predict(text)[0][0].replace("__label__", "").split("_")[0][0:2]

def winsorize(val, bottom_limit, top_limit):
    if val < bottom_limit:
        return bottom_limit
    elif val > top_limit:
        return top_limit
    else:
        return val

def normalize(posts):
    '''
        We rescale all scores to be in the range [-1, 1] with negative values
        being undesirable and positive values being desirable. We also rename
        the reverted columns to avoid confusion.
    '''
    # scale polarization to be in [0, 1] for easier handling
    posts["polarization"] = (posts["polarization"] + 1) / 2
    # scale prosociality to be in [0, 1] for easier handling
    posts["prosociality"] = (posts["prosociality"] + 1) / 2
    # scale mtld to be in [0, 1] for easier handling
    posts["mtld"] = posts["mtld"] / posts["mtld"].max()
    
    # winsorize scores
    bottom_limit = 0.1
    top_limit = 0.9

    for col in ["toxicity", "polarization", "mtld", "prosociality"]:
        posts[col] = posts[col].apply(
            winsorize, 
            args=(posts[col].quantile(q=bottom_limit),
                  posts[col].quantile(q=top_limit))
        )
    
        # rescale score to be in [0, 1] after removing outliers
        # this assumes that the score was in [0, 1] before winsorizing
        posts[col] = posts[col] - posts[col].min()
        posts[col] = posts[col] / posts[col].max()
        
    # revert score: high toxicity is good
    posts["toxicity"] = 1 - posts["toxicity"]
    # shift and rescale toxicity to be in [-1, 1]
    posts["toxicity"] = (posts["toxicity"] * 2) - 1
    
    # revert score: high polarization is good
    posts["polarization"] = 1 - posts["polarization"]
    # shift and rescale polarization to be in [-1, 1]
    posts["polarization"] = (posts["polarization"] * 2) - 1

    # shift and rescale prosociality to be in [-1, 1]
    posts["prosociality"] = (posts["prosociality"] * 2) - 1
    
    # shift and rescale mtld to be in [-1, 1]
    posts["mtld"] = (posts["mtld"] * 2) - 1

    # shift and rescale trustworthiness to be in [-1, 1]
    posts["trustworthiness"] = (posts["trustworthiness"] * 2) - 1
    
    posts = posts.rename(columns={"toxicity":"no_toxicity", "polarization":"no_polarization"})
    return posts

def calculate_compound_score(row, weights, min_scores):
    if len(row.dropna()) < min_scores:
        return np.nan
        
    norm = 0
    compound_score = 0
    for score in weights.keys():
        if row[score] == row[score]: # nan-check
            compound_score += row[score] * weights[score]
            norm += weights[score]
    if norm != 0:
        return compound_score / norm
    else:
        return np.nan