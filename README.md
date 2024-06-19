# Civirank
Repo for the Civirank ranker.

# Submission form

* **Describe your algorithm. How will you reorder, add, and/or remove content?**  (Up to 250 words.)
The overall goal of my algorithm is to create a balanced ranking that surfaces informative and friendly content whereas suppressing toxic and polarizing content or untrustworthy news items. To achieve this, I reorder content based on a compound score which is composed of five sub-scores for different content dimensions: toxicity, polarization, prosociality, trustworthiness and informativeness. Toxicity is currently measured using the Perspective API (to be changed to a locally run classifier). Polarization and prosociality are only measured for English language posts and measured as similarity between embeddings of English language polarization and prosociality dictionaries from the literature, and the post text. Trustworthiness is only assessed if a post contains a link to a news site and is measured on the domain level using NewsGuard ratings [might switch to Lin et al.]. Informativeness is measured using a standard measure for lexical diversity.

The compound score is calculated as [(1 - toxicity) + (1 - polarization) + prosociality + 2 * trustworthiness + 0.5 * informativeness] / 5.5. A high compound score therefore means that toxicity and polarization are low, whereas trustworthiness, prosociality and lexical diversity are high. The weight of informativeness in the compound score is lower whereas the weight of trustworthiness is higher, meaning that posts that contain a news item will be on average ranked higher. The weight of informativeness is lower to not overly penalize very short posts.

If the compound score in a given feed drops below a threshold value, a post with a "scroll warning" is inserted, encouraging the user to stop scrolling until new high-quality content becomes available.

* **What is the point of doing this? What is the intended impact of your ranking algorithm on the lives of people or communities?** (Up to 250 words.)
The intended impact on communities is to increase participation of community members in productive civic discourse. This includes both an increase in discourse between polarized communities (e.g. Democrats and Republicans) as well as increased participation by members of minority groups. This is driven by an increase in exposure to high-quality news and informative & prosocial content that serves as basis for discussion, less exposure to toxic content that could discourage people from participating in discussions, and less polarizing content that could incite cross-partisan fights.

The intended impact on individuals is that their overall well-being is increased by reducing exposure to toxic, polarizing or misleading content by ranking this content lower and only serving it after showing a "scroll warning". In addition, I also intend to slightly reduce overall social media usage by showing the "scroll warnings" and encouraging users to reflect their social media usage and log off if scrolling more would cause them to see low-quality content or content that could cause negative emotions or stress. Furthermore, I intend to increase user's knowledge of current events by ranking posts containing (high-quality) news and generally more informative posts higher. Lastly, by ranking prosocial content higher, I expect users have an overall better time on the platform, ensuring retention.


* **Please consult the [list of variables](https://rankingchallenge.substack.com/p/dependent-variables-the-outcomes) we will measure.  Among those, what specific outcome variables do you hope to move, and by how much (in standard deviations)?**
It's hard to find good priors for effect sizes, therefore I tried to estimate basted on the few effect sizes I know and the weight of the outcome in the ranker.

    * reduce affective polarization 0.2 std
    * increase wellbeing 0.2 std
    * reduce negative/bad for the world content 0.2 std
    * increase news knowledge, "learned something useful" & "meaningful connection" 0.2 std
    * reduce total time spent & total posts seen on platform 0.2 std
    * increase engagements with news 0.2 std
    * decrease engagements with toxic content 0.2 std
    * decrease engagements with misinformation 0.2 std
    * decrease scroll depth 0.2 std
    * decrease exposure to misinformation 0.8 std
    * decrease exposure to toxic content 0.4 std
    * increase exposure to content with high information quality by 0.8 std
    * increase exposure to news content by 0.8 std


| outcome | level | instrument | relevant intervention | prediction direction | prediction magnitude [std dev] |
| --- | --- | --- | --- | --- | --- |
| conflict | primary | [affective polarization](https://electionstudies.org/data-tools/anes-guide/anes-guide.html?chart=affective_polarization_parties) | affective polarization similarity | down | 0.2 - 0.3 |
| conflict | primary | [meta-perceptions](https://www.pnas.org/doi/epdf/10.1073/pnas.2116851119) | affective polarization similarity | down | 0.1 - 0.2 |
| conflict | primary | [index](https://www.journals.uchicago.edu/doi/abs/10.1086/685735?journalCode=jop) | affective polarization similarity | down | 0.1 - 0.2 |
| conflict | secondary | social trust | prosociality similarity | up | 0.1 - 0.2 |
| conflict | secondary | [outparty friends](https://dl.acm.org/doi/10.1145/3610190) | prosociality similarity | up | 0.1 - 0.2 |
| well-being | primary | [wellbeing](https://www.psykiatri-regionh.dk/who-5/Documents/WHO-5%20questionaire%20-%20English.pdf) | toxicity, prosociality similarity | down | 0.1 - 0.2 |
| well-being | primary | [Neely index](https://psychoftech.substack.com/p/unveiling-the-neely-ethics-and-technology) | toxicity, prosociality similarity, trustworthiness, affective polarisation similarity, scroll warnings | down | 0.2 - 0.3 |
| information | primary | [news knowledge quiz](https://assets.aeaweb.org/asset-server/files/11625.pdf) | news prioritisation, trustworthiness | up | 0.2 - 0.3 |
| information | primary | [Neely index](https://psychoftech.substack.com/p/unveiling-the-neely-ethics-and-technology) | lexical density | up | 0.2 - 0.3 |
| engagement & usage | primary | total number of posts seen on each platform | scroll warnings | down | 0.2 - 0.3 |
| engagement & usage | primary | total time spent on each platform | scroll warnings | down | 0.2 - 0.3 |
| engagement & usage | primary | rate of engagements [overall, politics/civic, news] | lexical density | up | 0.1 - 0.2 |
| engagement & usage | primary | engagement with toxicity [Jigsaw, Dignity Index] |  |  |  |
| engagement & usage | primary | Average toxicity of posts created toward ingroup vs. outgroup members |  |  |  |
| engagement & usage | secondary | retention  | scroll warnings | down | 0.1 - 0.2 |
| engagement & usage | secondary | Distribution of scroll depth (posts seen per session)  | scroll warnings | down | 0.1 - 0.2 |
| engagement & usage | secondary | engagements politics/civic, news | news prioritisation | down | 0.1 - 0.2 |
| engagement & usage | secondary | engagements misinformation | trustworthiness | down | 0.3 - 0.4 |
| engagement & usage | secondary | # of Ads seen |  |  | |
| engagement & usage | secondary | time spent on other social media platforms |  |  | |
| engagement & usage | secondary | % of mobile social media usage |  |  | |
| feed changes | primary | per-session number of items added, deleted |  |  |  |
| feed changes | primary | measures of difference between lists input to and output from ranker | trustworthiness | up | 0.3 - 0.4 |
| feed changes | primary | fraction of political content/civic content | news prioritisaion | up | 0.2 - 0.3 |
| feed changes | primary | fraction of news content | news prioritisation | up | 0.3 - 0.4 |
| feed changes | primary | fraction of toxic content | |  |  |
| feed changes | primary | fraction of misinformation content |  |  |  |
| feed changes | primary | distribution over posts seen and posts served information quality |  |  |  |
| feed changes | primary | distribution over posts seen and posts served toxicity |  |  |  |

* **(Optional) Are there other things you would like to measure, including survey measures that would appear as part of the study's panel surveys or as in-feed survey questions? Please describe.**
I would like to add back two items from the addiction scale, replacing "phone" with "social media" in the second question. In addition, I would like to test if users have noticed the "scroll warning" post and why they continued scrolling. Therefore I would like to add the following three questions as in-feed survey questions:
1. *In the past 24 hours, did you feel like you had an easy time controlling your screen time?* [Never, Rarely, Sometimes, Often, Always]
2. *In the past 24 hours, would you ideally have used social media less?* [Never, Rarely, Sometimes, Often, Always]
3. *A few posts ago we showed you a warning, encouraging you to stop scrolling in your feed. Why did you keep scrolling?* [I didn't notice the warning, I didn't want to stop, I was curious about the posts after the warning, None of the above, I don't know]

* **What is new about your approach? Why do you believe that it will work, and what evidence do you have? Please discuss your theory of change and include citations to previous work.** (Up to 250 words.)
I believe my ranker has two new components: 
(1) A balanced ranking that considers various quality dimensions with the overall aim of improving civic discourse.
(2) "Scroll warnings" reducing the social media usage if it would mainly cause users to see low-quality content. 

Below I discuss why I believe it will work:
 
- Reducing toxic content might prevent members of minority groups from leaving or self-censoring (hateful content directed at minorities has these effects https://doi.org/10.1177/1043986213507403).
- Reducing content high in affective polarization might directly reduce affective polarization. Facebook & Instagram election study (https://doi.org/10.1126/science.abp9364) didn't find a change in affective polarizatio, but they only tested curated vs. chronological feed, not re-ranking based on affective polarization. I am interested to see if this makes a difference.
- Prosocial features in language predict a prosocial trajectory of conversations (see https://doi.org/10.1145/3442381.3450122). I expect people to have an overall better time using the platform if they see more prosocial content, keeping up retention. I also expect them to see less content "bad for the world" that is not caught by the toxicity ranking.
- My ranker ranks posts with news from trustworthy sources very high. Therefore I expect people's news knowledge to increase but their exposure to misinformation decrease.
- Increasing well-being through reducing stress caused by seeing toxic content (exposure to hateful content increases stress https://doi.org/10.1145%2F3292522.3326032). Reducing low-quality news content inciting negative emotions might also contribute.
- Injecting content can change user behaviour (https://doi.org/10.31234/osf.io/u8anb). I expect that reminding people to stop scrolling if there is no good content might cause some to log off earlier than they would have otherwise.



* **How was your prototype built? (What language was used to code it, what dependencies are required, and what external services does it depend on?)**  (Up to 250 words.)
The prototype was build using Python. Dependencies are the libraries numpy, pandas, sentence_transformers, PyTorch, googleapiclient, lexicalrichness and langdetect (I can provide a requirements.txt file). In addition, the prototype depends on pre-calculated embeddings of the polarization and prosociality dictionaries, and the NewsGuard trustworthiness scores, provided in the form of a .csv file. In addition, the prototype currently uses the Google Perspective API to calculate toxicity scores.

* **Where can we find your submission? This needs to be a URL to a live HTTP endpoint implementing our first-round specifications.**
Ranker can be found at http://5.75.245.130:5001/rank. Note that this is currently really slow (takes about 1 min to rank ~600 posts). The main bottleneck is the current sequential toxicitiy inference with the perspective API, which will be switched to a locally hosted alternative. 

* **Have any members of your team previously published related work? Please provide up to five links to previous projects, code, papers, etc.**
- Social media sharing of low quality news sources by political elites https://doi.org/10.1093/pnasnexus/pgac186. In this paper we use the same approach of assessing trustworthiness of news pieces on the domain level with NewsGuard scores.
- High level of agreement across different news domain quality ratings https://doi.org/10.1093/pnasnexus/pgad286. In this paper we show that NewsGuard scores are consistent with other data bases that assess domain trustworthiness.
- From Alternative conceptions of honesty to alternative facts in communications by U.S. politicians https://doi.org/10.1038/s41562-023-01691-w. In this paper we use a similar approach to measure social media post similarity to constructs such as "polarization" and "prosociality" by embedding dictionaries.
