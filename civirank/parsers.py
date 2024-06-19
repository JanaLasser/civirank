from urllib.parse import urlparse
import re
from pathlib import Path
import json

import pandas as pd
import numpy as np


def extract_urls(text):
    if text != text:
        return np.nan
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    if len(urls) > 0:
        return urls
    else:
        return np.nan


def extract_domains(url_list):
    if url_list != url_list:
        return np.nan
    else:
        return [urlparse(url).netloc.replace("www.", "") for url in url_list if urlparse(url).scheme]

def parse_twitter_posts(posts_json, lim=False, debug=False):
    if lim:
        posts_json = posts_json[0:lim]
    IDs = [post.get("id") for post in posts_json]
    texts = [post.get("text") for post in posts_json]
    lang = [post.get("lang") for post in posts_json]

    # the current data format seems to include a maximum of one "expanded_url",
    # therefore no need to deal with lists of urls/domains in posts
    # we still keep the list format though to be consistent with the data from
    # the other platforms
    urls = [[post.get("expanded_url")] if post.get("expanded_url") != None else np.nan for post in posts_json]
    domains = [[urlparse(url[0]).netloc.replace("www.", "")] if url == url else np.nan for url in urls]
    
    posts = pd.DataFrame({
        "id":IDs,
        "text":texts,
        "url":urls,
        "domain":domains,
        "lang":lang
    })
    
    if debug == True:
        posts["original_rank"] = [post.get("original_rank") for post in posts_json]

    return posts.reset_index(drop=True)


def combine_reddit_text(title, text):
    if title != None:
        title = title.replace("[deleted by user]", "")
        title = title.replace("[deleted]", "")
        title = title.replace("[removed]", "")
    if text != None:
        text = text.replace("[deleted by user]", "")
        text = text.replace("[deleted]", "")
        text = text.replace("[removed]", "")
        
    if (title != None) and (len(title) > 0):
        if (text != None) and (len(text) > 0):
            return title + " " + text
        else:
            return title
    elif (text != None) and (len(text) > 0):
        return text
    else:
        return ''


def parse_reddit_posts(posts_json, lim=False, debug=False):
    if lim:
        posts_json = posts_json[0:lim]
        
    IDs = [post.get("id") for post in posts_json]
    texts = [combine_reddit_text(post.get("title"), post.get("text")) for post in posts_json]
    url_lists = [extract_urls(text) for text in texts]
    domain_lists = [extract_domains(url_list) for url_list in url_lists]
    lang = [post.get("lang") for post in posts_json]

    posts = pd.DataFrame({
        "id":IDs,
        "text":texts,
        "url":url_lists,
        "domain":domain_lists,
        "lang":lang
    })

    if debug == True:
        posts["original_rank"] = [post.get("original_rank") for post in posts_json]
    
    return posts.reset_index(drop=True)


def parse_facebook_posts(posts_json, lim=False, debug=False):
    if lim:
        posts_json = posts_json[0:lim]
        
    IDs = [post.get("id") for post in posts_json]
    texts = [post.get("text") for post in posts_json]
    url_lists = [extract_urls(text) for text in texts]
    domain_lists = [extract_domains(url_list) for url_list in url_lists]
    lang = [post.get("lang") for post in posts_json]
    
    posts = pd.DataFrame({
        "id":IDs,
        "text":texts,
        "url":url_lists,
        "domain":domain_lists,
        "lang":lang
    })

    if debug == True:
        posts["original_rank"] = [post.get("original_rank") for post in posts_json]
    
    return posts.reset_index(drop=True)
