from civirank import analyzers, parsers
from ranking_challenge.request import RankingRequest
import numpy as np
import pandas as pd

class LocalRanker():
    def __init__(self, weights=None, lim=False, min_scores=0, debug=False, warning_urls=None):

        # Set the weights for the different scores
        if weights is None:
            self.weights = {
                "no_toxicity": 1,
                "no_polarization": 1,
                "mtld": 0.5,
                "trustworthiness": 2,
                "prosociality": 1
            }
        else:
            self.weights = weights

        # Set the warning urls for the scroll component
        if warning_urls is None:
            self.warning_urls = {
                'reddit': {'id': '1bvin9r', 'url': 'https://www.reddit.com/r/test/comments/1bvin9r/'},
                'twitter': {'id': '1776172261436727724', 'url': 'https://x.com/ScrollWarning/status/1776172261436727724'},
                'facebook': {'id': '61557764711849', 'url': 'https://www.facebook.com/permalink.php?story_fbid=pfbid0qFdDR2P2mZjSvintqSqWGgzLRi14tvPt5ccYMFKu7BcNvkxEX7ZmufENH9QQrnnKl&id=61557764711849'}
            }
        else:
            self.warning_urls = warning_urls

        # Initialize analyzers
        self.TrustworthinessAnalyzer = analyzers.TrustworthinessAnalyzer()
        self.ToxicityAnalyzer = analyzers.ToxicityAnalyzer()
        self.ProsocialityPolarizationAnalyzer = analyzers.ProsocialityPolarizationAnalyzer()
        self.LexicalDensityAnalyzer = analyzers.LexicalDensityAnalyzer()
        self.LanguageAnalyzer = analyzers.LanguageAnalyzer()

        # Scores that are considered in the compound score
        self.scores = ['no_toxicity', 'no_polarization', 'mtld', 'trustworthiness', 'prosociality']

        # Minimum number of scores a post needs to have to be considered in the compound score
        self.min_scores = min_scores

        # Limit the number of posts to be analyzed
        self.lim = lim

        # Debug flag
        self.debug = debug

    def rank(self, ranking_request, batch_size=16, scroll_warning_limit=-0.1):

        # Check if ranking_request is a RankingRequest object or a dictionary
        if isinstance(ranking_request, RankingRequest):
            dataset = ranking_request.dict()
        else:
            dataset = ranking_request

        platform = dataset["session"]["platform"]
        
        # Detect language of each post
        for i in range(len(dataset["items"])):
            dataset['items'][i]['lang'] = self.LanguageAnalyzer.detect_language(dataset['items'][i]['text'].replace('\n', ' '))
            if self.debug:
                dataset['items'][i]['original_rank'] = i
        
        if self.debug:
            print("{:d} posts not in English.".format(len(dataset["items"]) - len([item for item in dataset["items"] if item["lang"] == "en"])))

            #prints value_counts of languages
            print(pd.DataFrame([item["lang"] for item in dataset["items"]], columns=["lang"])["lang"].value_counts())

        # Parse posts
        if platform == "twitter":
            posts = parsers.parse_twitter_posts(dataset["items"], lim=self.lim, debug=self.debug)
        elif platform == "reddit":
            posts = parsers.parse_reddit_posts(dataset["items"], lim=self.lim, debug=self.debug)
        elif platform == "facebook":
            posts = parsers.parse_facebook_posts(dataset["items"], lim=self.lim, debug=self.debug)

        # Splits the posts into ones that get reranked and ones that don't
        parse_posts = posts[(posts["lang"] == "en") & (posts["text"].str.len() > 0)].copy()
        non_parse_posts = posts[(posts["lang"] != "en") | (posts["text"].str.len() == 0)].copy()

        # Process posts
        parse_posts.loc[:, "trustworthiness"] = self.TrustworthinessAnalyzer.get_trustworthiness_scores(parse_posts)
        parse_posts.loc[:, "toxicity"] = self.ToxicityAnalyzer.get_toxicity_scores(parse_posts, batch_size=batch_size)
        parse_posts.loc[:, "polarization"] = self.ProsocialityPolarizationAnalyzer.get_similarity_polarization(parse_posts)
        parse_posts.loc[:, "prosociality"] = self.ProsocialityPolarizationAnalyzer.get_similarity_prosocial(parse_posts)
        parse_posts.loc[:, "mtld"] = self.LexicalDensityAnalyzer.get_mtld(parse_posts)

        parse_posts = analyzers.normalize(parse_posts)

        # Calculate the compound score
        parse_posts["compound_score"] = parse_posts[self.scores].apply(analyzers.calculate_compound_score, args=(self.weights, self.min_scores), axis=1)

        # Sort posts in descending order based on compound score
        parse_posts = parse_posts.sort_values(by="compound_score", ascending=False)

        # Create a list to store final posts in the correct order
        final_posts = []
        en_index = 0
        non_en_index = 0

        # Reinsert posts to their original positions
        for idx in range(len(posts)):
            if posts.iloc[idx]["lang"] == "en" and posts.iloc[idx]["text"].strip() != "":
                final_posts.append(parse_posts.iloc[en_index])
                en_index += 1
            else:
                final_posts.append(non_parse_posts.iloc[non_en_index])
                non_en_index += 1

        # Reset index for the final_posts list
        final_posts_df = pd.DataFrame(final_posts).reset_index(drop=True)

        # Inserts a warning message for the scroll component
        insert_index = final_posts_df[final_posts_df['compound_score'] < scroll_warning_limit].first_valid_index()
        if insert_index is not None:
            id_platform = self.warning_urls[platform]['id']
            new_row = pd.DataFrame({'id': id_platform, 'compound_score': scroll_warning_limit}, index=[insert_index - 0.5])
            final_posts_df = pd.concat([final_posts_df.iloc[:insert_index], new_row, final_posts_df.iloc[insert_index:]]).reset_index(drop=True)


        # Return full dataframe with original dataset and scores if DEBUG is True
        if self.debug:
            return final_posts_df

        # Otherwise, return list of ids
        if insert_index is not None:
            return list(final_posts_df["id"]), [self.warning_urls[platform]]
        else:
            return list(final_posts_df["id"]), []
