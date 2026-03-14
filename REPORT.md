# How AI Sees Your Name: Measuring Name-Outcome Associations in Word Embedding Spaces

**Authors:** [Anonymous for review]

---

## Abstract

Personal names pass through word embedding layers millions of times daily in deployed AI systems — from resume screeners to credit models to large language models. Do names that occupy "favorable" positions in embedding space correlate with real-world socioeconomic outcomes? We apply the Word Embedding Association Test (WEAT) across six semantic dimensions (wealth, wisdom, happiness, health, leadership, beauty) to measure name-concept associations in both Chinese (Tencent AI Lab, 200d) and English (GloVe 6B, 300d) embedding spaces. In a Chinese-language experiment, a three-layer fusion model (Word2Vec character-average + BERT + whole-word matching) distinguishes 99,729 government grant recipients from 100,000 general-population names with AUC = 0.748. In English, name-level WEAT composites correlate with Chicago city employee salaries at r = 0.345, with top-quartile names earning $9,702/year more than bottom-quartile names. Five categories of global elites (senators, Nobel laureates, Oscar winners, Olympic gold medalists, billionaires) all score significantly higher than population baselines. Crucially, however, a state-level analysis of U.S. baby names reveals a *negative* correlation (r = −0.127) between family income and WEAT score, driven by culturally non-mainstream naming traditions in high-income communities. We argue that WEAT measures proximity to the cultural mainstream of the training corpus — which is precisely the bias vector that affects downstream AI applications.

**Keywords:** word embeddings, WEAT, name bias, AI fairness, cultural bias, NLP

---

## 1. Introduction

Names are among the most frequently embedded tokens in natural language processing pipelines. Every time a resume screening system processes an applicant, a content recommendation engine profiles a user, or a large language model generates a response mentioning a person, names are mapped to dense vector representations that encode — among other things — the social and cultural associations the training corpus reflects.

We call this phenomenon **algorithmic numerology**: the assignment of implicit valence to names through the mathematics of embedding spaces. Unlike traditional numerology, this process is neither mystical nor benign. It is linear algebra applied at industrial scale, and it carries measurable consequences.

The question we investigate is straightforward: *Do names that occupy semantically "favorable" positions in word embedding spaces — closer to concepts like wealth, wisdom, and leadership — correlate with favorable real-world outcomes?* If so, what does this correlation reveal about the nature of embedding bias?

We find that the answer is yes, with important caveats. Across five experiments spanning Chinese and English, covering nearly 250,000 individuals and multiple embedding architectures, names associated with positive semantic concepts in embedding space are consistently overrepresented among high achievers. Grant-funded scientists have names closer to "wealth" and "wisdom" in Chinese embedding space. Higher-WEAT English names correlate with higher salaries. Senators, Nobel laureates, and billionaires all carry names that score above population baselines.

But our most important finding is not the positive correlations — it is the discovery of what WEAT actually measures. A state-level analysis of U.S. baby names reveals that the wealthiest families often give their children names that score *low* on English WEAT, because those families belong to cultural communities (e.g., Orthodox Jewish communities in New York and New Jersey) whose naming conventions diverge from the English-language mainstream. This finding reframes the entire enterprise: what WEAT captures is not "inherent name quality" but *proximity to the cultural mainstream of the training data*. This is both a limitation of our method and, we argue, the most policy-relevant finding of the paper, because it is exactly this cultural-mainstream bias that propagates through downstream AI systems.

The narrative arc of this paper is thus: we set out to test whether name embeddings predict success; we found that they do; and then we discovered *why* — and the why reveals fundamental issues with embedding-based decision systems.

---

## 2. Related Work

### 2.1 Bias in Word Embeddings

Caliskan, Bryson, and Narayanan (2017) introduced the Word Embedding Association Test (WEAT), adapting the Implicit Association Test (IAT) from psychology to measure bias in embedding spaces. They demonstrated that embeddings trained on internet corpora reproduce human-like implicit biases — European-American names are associated with pleasantness, flowers with positivity, and so on — with effect sizes comparable to those observed in human subjects. Garg et al. (2018) extended this work diachronically, showing that word embeddings trained on text from different decades track the historical evolution of gender and ethnic stereotypes over 100 years, with embedding-derived bias measures correlating with contemporaneous social attitude surveys. Bolukbasi et al. (2016) demonstrated structural gender bias in Word2Vec ("man is to computer programmer as woman is to homemaker") and proposed geometric debiasing methods, establishing embedding fairness as a research program.

### 2.2 Names and Socioeconomic Outcomes

The economics literature on name effects is substantial. Levitt and Fryer (2004) analyzed millions of California birth records and found significant statistical associations between names and outcomes such as education and income. Their key identification strategy — sibling comparisons within families — showed that most of the association is driven by family background rather than the name itself. Clark (2014) traced surnames across 800 years of records in multiple countries, finding that elite surnames persist at the top of socioeconomic distributions far longer than standard models of social mobility predict, suggesting that names are powerful markers — though not necessarily causes — of social position.

### 2.3 Name Embeddings

Ye and Skiena (2019) demonstrated that name embeddings learned from social media can predict demographic attributes (gender, ethnicity, nationality) with high accuracy, revealing both the power and the privacy risks of name representations in vector spaces.

### 2.4 Chinese Naming Culture

Chinese names (名, *míng*) differ structurally from English names in a crucial respect: each character carries explicit semantic content. A name like 志远 (*zhìyuǎn*, "ambition-far") is semantically transparent in a way that "James" is not. This means Chinese name embeddings encode both the inherent lexical semantics of individual characters and the social associations of the name as a whole — a duality that makes Chinese names particularly interesting for WEAT analysis.

---

## 3. Method

### 3.1 WEAT Scoring

Following Caliskan et al. (2017), we define the association score of a target word *w* with respect to two attribute sets *A* (positive) and *B* (negative) as:

$$s(w, A, B) = \frac{1}{|A|} \sum_{a \in A} \cos(\vec{w}, \vec{a}) - \frac{1}{|B|} \sum_{b \in B} \cos(\vec{w}, \vec{b})$$

where $\cos(\cdot, \cdot)$ denotes cosine similarity between embedding vectors. A positive score indicates that *w* is, on average, closer to the positive attribute set than to the negative attribute set in embedding space.

### 3.2 Semantic Dimensions

We evaluate names along six dimensions, each defined by a set of 8 positive and 8 negative attribute words:

| Dimension | Positive examples | Negative examples |
|-----------|------------------|-------------------|
| **Wealth** (财富) | wealthy, prosperous, affluent... | poor, impoverished, destitute... |
| **Wisdom** (智慧) | intelligent, wise, brilliant... | foolish, ignorant, stupid... |
| **Happiness** (幸福) | joyful, happy, content... | miserable, sorrowful, wretched... |
| **Health** (健康) | healthy, vigorous, strong... | sick, frail, weak... |
| **Leadership** (领导力) | leader, distinguished, elite... | mediocre, ordinary, insignificant... |
| **Beauty** (美感) | beautiful, elegant, graceful... | ugly, vulgar, grotesque... |

Chinese-language attribute sets use corresponding Chinese terms (e.g., 富裕/贫穷 for wealth, 智慧/愚蠢 for wisdom). The composite WEAT score for each name is the average across all six dimensions.

### 3.3 Embedding Models

**Chinese:** Tencent AI Lab Chinese Word Vectors (Song et al., 2018), 200 dimensions, approximately 143,000-token vocabulary (lite version). A total of 2,899 unique Chinese characters received WEAT scores via this model. Additionally, we employ chinese-macbert-base (Cui et al., 2020) as a BERT-based contextual embedding layer, scoring 3,000 characters on an RTX 5080 GPU.

**English:** GloVe 6B 300d (Pennington et al., 2014), trained on 6 billion tokens from Wikipedia and Gigaword. A total of 29,842 English given names from SSA records were scored.

### 3.4 Three-Layer Fusion for Chinese Names

Chinese names require special handling because most two-character given names do not appear as whole tokens in Word2Vec vocabularies. We implement a three-layer scoring architecture:

1. **Word2Vec character-average:** Compute the WEAT score for each character independently using the Tencent embeddings, then average across the characters of the name.
2. **BERT contextual embedding:** Pass the full name through chinese-macbert-base and extract the pooled representation, then compute WEAT against attribute words embedded in the same space.
3. **Whole-word matching:** For names that do appear as whole tokens in the vocabulary, compute WEAT directly on the whole-word vector.

The three layers are fused via a weighted average. We note that the cosine similarity between character-average vectors and whole-word vectors is only 0.52 on average, indicating that these two representations capture substantially different information — justifying the multi-layer approach.

### 3.5 Data Sources

**Chinese grant recipients (成功者, "achievers"):** 99,729 scientists and scholars who received government research funding, drawn from the Chinese Gender Dataset (Shi & Tong, 2025, *Nature Scientific Data*). Receiving a competitive national grant serves as our operational definition of career achievement.

**Chinese general population (普通人, "general population"):** 100,000 names randomly sampled (seed = 42) from the CCNC (Chinese Corpus of Name and Character) dataset, comprising approximately 3.65 million name records. The top-10,000 names by frequency were used for frequency-based analyses.

**Chicago city employees:** 32,069 employees with publicly available names and annual salaries, obtained from the City of Chicago open data portal.

**Global elites:** Compiled from Wikidata and Forbes: U.S. Senators (n = 965), Nobel Laureates (n = 860), Oscar Winners (n = 125), Olympic Gold Medalists (n = 950), and Forbes Billionaires (n = 1,699).

**SSA baby names:** U.S. Social Security Administration baby name records, including state-level data for 2015–2024, linked to state median household income from the U.S. Census Bureau.

### 3.6 Statistical Methods

Group comparisons use Welch's t-test with Cohen's d as the effect size measure. Continuous associations are reported as Pearson correlations. Classification performance is reported as area under the ROC curve (AUC) with standard deviation from 5-fold cross-validation. All reported p-values are two-tailed unless otherwise noted.

---

## 4. Experiments and Results

### 4.1 Experiment 1: Chinese Name-Achievement Association

We compare the WEAT profiles of 99,729 grant recipients against 100,000 general-population names in the Tencent embedding space.

**Table 1.** WEAT dimension scores: grant recipients vs. general population (差值 = grantee mean − general mean).

| Dimension | Δ (difference) | Direction | p-value |
|-----------|----------------|-----------|---------|
| Wealth (财富) | +0.009 | Grantees > General | ≈ 0 |
| Wisdom (智慧) | +0.005 | Grantees > General | ≈ 0 |
| Leadership (领导力) | +0.005 | Grantees > General | ≈ 0 |
| Health (健康) | +0.001 | Grantees > General | < 0.001 |
| Happiness (幸福) | −0.002 | Grantees < General | < 0.001 |
| Beauty (美感) | −0.005 | Grantees < General | ≈ 0 |

The three-layer fusion model (Word2Vec character-average + BERT + whole-word) achieves an AUC of **0.748 ± 0.031** in distinguishing grantees from general-population names.

The dimension-level results reveal a striking pattern: grantees' names score higher on "achievement-oriented" dimensions (wealth, wisdom, leadership) but *lower* on "affective" dimensions (happiness, beauty). This is consistent with Chinese naming conventions (命名文化): scholarly families tend to choose characters connoting ambition and intellect (志 *zhì* "ambition," 哲 *zhé* "philosophy," 明 *míng* "brightness"), while characters associated with happiness (欢 *huān*, 乐 *lè*) and beauty (秀 *xiù*, 美 *měi*) are considered more colloquial and are disproportionately used in female names — relevant given the male skew among funded researchers.

### 4.2 Experiment 2: U.S. Salary Validation

To test whether English-language WEAT scores correlate with a concrete economic outcome, we analyze 32,069 Chicago city employees with publicly available names and annual salaries, scoring first names using GloVe 6B 300d.

**Table 2.** WEAT–salary associations (Chicago city employees).

| Metric | Value |
|--------|-------|
| Individual-level: all 6 dimensions vs. salary | p < 0.001 (all significant) |
| Name-level Pearson r (names with ≥ 10 holders) | **0.345** (p ≈ 0) |
| Top 25% WEAT vs. Bottom 25% WEAT salary gap | **+$9,702/year** |

At the individual level, all six WEAT dimensions are significantly correlated with salary (p < 0.001). Aggregating to the name level (averaging salary across all employees sharing a given name, restricting to names held by at least 10 employees) yields a Pearson correlation of r = 0.345 between composite WEAT score and mean salary. The salary gap between the top and bottom WEAT quartiles — $9,702 per year — is economically meaningful, though we emphasize that this is an observational association, not a causal estimate.

### 4.3 Experiment 3: Global Elite Name Analysis

We score the first names of individuals in five elite categories using GloVe 6B 300d and compare against a baseline of 5,000 randomly sampled SSA names.

**Table 3.** Mean WEAT composite scores by elite category.

| Category | n | Mean WEAT | Δ vs. baseline | p-value |
|----------|---|-----------|----------------|---------|
| U.S. Senators | 965 | 0.064 | **+0.086** | ≈ 0 |
| Oscar Winners | 125 | 0.057 | +0.079 | ≈ 0 |
| Olympic Gold Medalists | 950 | 0.046 | +0.068 | ≈ 0 |
| Nobel Laureates | 860 | 0.046 | +0.068 | ≈ 0 |
| Forbes Billionaires | 1,699 | 0.042 | +0.063 | ≈ 0 |
| SSA Baseline | 5,000 | −0.022 | — | — |

All five elite categories score significantly above the population baseline, with U.S. Senators showing the largest gap (+0.086) and Forbes Billionaires the smallest (+0.063). The ordering is itself informative: senators — whose names are most deeply embedded in English-language political discourse — show the strongest association, while billionaires — a more internationally diverse group — show the weakest.

### 4.4 Experiment 4: Self-Made vs. Inherited Billionaires

The most common objection to name-outcome associations is that names merely proxy for family socioeconomic status (SES). To probe this, we exploit the `wealth.type` field in the Forbes billionaire dataset, which distinguishes self-made wealth from inherited wealth.

**Table 4.** WEAT scores by wealth origin.

| Group | n | Mean WEAT |
|-------|---|-----------|
| Self-made | 828 | **0.047** |
| Inherited | 595 | **0.037** |
| Difference | — | +0.010 (p = 5 × 10⁻⁶) |

If WEAT scores were purely an SES proxy, we would expect inherited billionaires — who come from wealthier families by definition — to score *higher*. Instead, the self-made group scores significantly higher (Δ = +0.010, p = 5 × 10⁻⁶). This finding rules out the simplest version of the SES-proxy hypothesis and suggests that WEAT captures something beyond parental wealth — plausibly, proximity to the English-language cultural mainstream from which self-made billionaires disproportionately emerge.

### 4.5 Experiment 5: The Cultural Bias Discovery

This experiment constitutes our most important finding. We link SSA state-level baby name data (2015–2024) to state median household income from the U.S. Census Bureau, computing the average WEAT score of names given in each state-year cell and correlating it with state-level income.

**Expected result:** Wealthier states → higher-SES families → names with higher WEAT scores (positive correlation).

**Actual result:**

$$r = -0.127 \quad (p = 8.9 \times 10^{-16})$$

The correlation is *negative*. Wealthier states and communities tend to produce names with *lower* WEAT scores.

Inspection of the data reveals why. The names associated with the highest household incomes include: **Dovid, Yaakov, Shmuel, Yehuda** — names from Orthodox Jewish communities concentrated in New York and New Jersey, among the highest-income demographics in the United States. These names score *low* on English WEAT because they are distant from the English-language cultural mainstream on which GloVe was trained. The embedding model has not encountered "Yaakov" in the same linguistic contexts as "wealth" and "leadership"; instead, "Yaakov" occupies a region of embedding space associated with a specific cultural-religious community.

This result fundamentally reframes what WEAT measures. It does not measure "inherent name quality" or even straightforward SES. It measures **proximity to the cultural mainstream of the training corpus**. Names that are common in English-language text — names borne by senators, CEOs, and protagonists of English novels — score high. Names from communities that are wealthy but culturally distinct from the English-language mainstream score low.

This is simultaneously:

- **A limitation of our method:** WEAT cannot distinguish "good names" from "mainstream names."
- **The most important finding for AI fairness:** It is exactly this cultural-mainstream bias that propagates through downstream NLP systems. A resume screening model trained on English text will implicitly favor names that are proximate to the English cultural mainstream — not because those names are "better," but because the embedding space is structured around that mainstream.

---

## 5. Discussion

### 5.1 What WEAT Actually Measures

Our five experiments, taken together, paint a coherent picture. WEAT name scores correlate with real-world success (Experiments 1–4), but this correlation is driven by a specific mechanism: names that are proximate to the cultural mainstream of the training corpus tend to belong to individuals who are also proximate to that mainstream — and who, for reasons related to cultural capital, institutional access, and historical advantage, tend to achieve favorable outcomes.

The self-made vs. inherited finding (Experiment 4) strengthens this interpretation. Self-made billionaires — who are more likely to emerge from within the English-language cultural mainstream — carry names that score higher than inherited billionaires, who include a larger share of individuals from non-Anglophone cultural backgrounds. WEAT is measuring cultural proximity, not parental investment in name selection.

The cultural bias discovery (Experiment 5) makes this explicit. The wealthiest naming communities in America — Orthodox Jewish families in the New York metropolitan area — produce names that score low on English WEAT because their naming conventions are rooted in Hebrew and Yiddish traditions rather than the English-language mainstream. The embedding model's "opinion" of these names reveals more about the model than about the names.

### 5.2 Implications for AI Fairness

This finding has direct implications for deployed AI systems. Consider a resume screening model that processes applicant names through a GloVe or similar embedding layer. Our results suggest that such a system would:

1. **Favor names from the cultural mainstream** — names like Elizabeth, Victoria, and James — not because these names signal competence, but because the training corpus associates them with positive contexts.
2. **Penalize names from culturally distinct communities** — names like Yaakov, Shmuel, or Lakisha — not because of anything inherent to those names, but because the training corpus underrepresents or misrepresents those communities.
3. **Do so invisibly** — because the bias is encoded in the geometry of embedding space, it is difficult to detect without targeted auditing of the kind we perform here.

This pattern extends beyond resume screening to credit scoring, content recommendation, and any NLP application where names pass through embedding layers. The bias is structural: it is not in any single parameter but in the relative positions of name vectors in high-dimensional space.

### 5.3 The Chinese Dimension Pattern

The Chinese results add nuance. Unlike English, where WEAT scores show a "rising tide" pattern (higher-scoring names are higher on all dimensions), Chinese names show a *selective* pattern: grantees' names score higher on achievement dimensions but lower on affective dimensions. This likely reflects the semantic transparency of Chinese characters. When parents choose 志远 (*zhìyuǎn*, "far-reaching ambition") over 欢乐 (*huānlè*, "joyful happiness"), they are making an explicit semantic choice that is preserved in the embedding. The embedding is, in some sense, merely reading the label that the parents wrote.

This cross-linguistic contrast suggests that the mechanism linking names to WEAT scores differs across languages. In Chinese, the dominant channel is *lexical semantics* (what the characters mean). In English, the dominant channel is *social association* (who has historically borne the name). Both channels produce measurable name-outcome correlations, but through different causal pathways.

### 5.4 Scale Amplification

The effect sizes we report are modest. Cohen's d for the Chinese grantee analysis is 0.092 — small by conventional standards. The name-level salary correlation of r = 0.345 is moderate but far from deterministic. At the individual level, name WEAT is a weak predictor of any outcome.

But AI systems operate at scale. A resume screening platform processes millions of applications per year. A credit scoring model evaluates millions of applicants. When a small bias is applied millions of times, the cumulative effect on the distribution of opportunities becomes substantial. This is the well-documented problem of *scale amplification* in algorithmic systems (Barocas & Selbst, 2016), and our results suggest that name-embedded bias is one channel through which it operates.

---

## 6. Limitations

**Correlation, not causation.** All of our findings are observational associations. We cannot determine whether names *cause* outcomes, whether outcomes *cause* naming patterns (through intergenerational cultural transmission), or whether both are driven by unobserved confounders. The most parsimonious interpretation is that names and outcomes share common causes — family SES, cultural milieu, historical period — and that embeddings faithfully encode the resulting statistical patterns.

**Cultural specificity of embeddings.** Our English results are specific to GloVe trained on English-language web text. Different embedding models, trained on different corpora, would produce different WEAT scores. This is not a bug — it is the central finding — but it means our specific numerical results do not generalize to other embedding models without replication.

**Temporal instability.** Embedding models are trained on fixed corpora and do not update in real time. The cultural associations captured by GloVe 6B (trained primarily on text from the 2000s and early 2010s) may not reflect current naming norms. Names that are gaining cultural prominence (e.g., through popular media) may be undervalued by older models.

**Attribute word selection.** The choice of positive and negative attribute words for each WEAT dimension involves researcher judgment. Different attribute sets could yield different results. We mitigate this by using six dimensions and reporting results across all of them, but the specific magnitudes are sensitive to attribute word choice.

**Sample composition.** The Chinese grantee sample is skewed male and concentrated in STEM fields. The Chicago salary dataset covers government employees, not the full labor market. The elite samples are subject to survivorship bias and historical overrepresentation of certain demographics.

---

## 7. Conclusion

We set out to answer a simple question: do names in "favorable" embedding positions correlate with favorable real-world outcomes? Across five experiments, two languages, and nearly 250,000 individuals, the answer is yes. Chinese grant recipients carry names closer to "wealth" and "wisdom" in embedding space. English names with higher WEAT scores correspond to higher salaries, and the names of senators, Nobel laureates, and billionaires all score above population baselines. Self-made billionaires score higher than inherited billionaires, ruling out simple SES-proxy explanations.

But our most important contribution is the discovery of *what WEAT actually measures*. The negative correlation between family income and WEAT score at the state level — driven by high-income communities with culturally non-mainstream naming traditions — reveals that WEAT captures proximity to the cultural mainstream of the training corpus, not inherent name quality. Names like Yaakov and Shmuel score low not because they are "bad names" but because GloVe's training data does not associate them with the positive concepts that define the English-language cultural mainstream.

This finding has direct implications for AI fairness. Any system that processes names through embedding layers — resume screeners, credit models, recommendation engines, large language models — inherits this cultural-mainstream bias. The bias is structural, invisible without targeted auditing, and amplified by the scale at which AI systems operate. Our work provides both a method for detecting this bias (WEAT applied to name embeddings) and a framework for understanding what it represents (cultural proximity, not quality).

Names do occupy systematically different positions in embedding space, and those positions correlate with real-world outcomes. But what we are measuring is not the destiny encoded in a name — it is the cultural geography of the training corpus, projected onto the space of human identities. In an era of pervasive AI systems, that projection has consequences.

---

## 8. Technical Appendix

**Table A1.** Embedding model specifications and coverage.

| Model | Dimensions | Names/Characters Scored | Application |
|-------|-----------|------------------------|-------------|
| Tencent AI Lab Word Vectors | 200 | 2,899 Chinese characters | Experiment 1 |
| chinese-macbert-base (BERT) | 768 | 3,000 Chinese characters | Experiment 1 |
| GloVe 6B | 300 | 29,842 English names | Experiments 2–5 |

**Table A2.** Chinese three-layer fusion performance.

| Layer | Description | Standalone AUC |
|-------|-------------|---------------|
| Word2Vec char-avg | Character-level WEAT averaged over name | — |
| BERT | Contextual embedding from macbert-base | — |
| Whole-word | Direct lookup of full name in vocabulary | — |
| **Three-layer fusion** | **Weighted combination** | **0.748 ± 0.031** |

Note: Whole-word vs. character-average cosine similarity = 0.52 on average, confirming that these representations capture substantially different information.

**Table A3.** Experiment 5 detail: SSA state-level WEAT × income.

| Statistic | Value |
|-----------|-------|
| Correlation (r) | −0.127 |
| p-value | 8.9 × 10⁻¹⁶ |
| Period | 2015–2024 |
| Highest-income names (low WEAT) | Dovid, Yaakov, Shmuel, Yehuda |
| Interpretation | WEAT measures cultural-mainstream proximity |

---

## References

Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. *California Law Review*, 104(3), 671–732.

Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *Advances in Neural Information Processing Systems (NeurIPS)*, 29.

Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183–186.

Clark, G. (2014). *The Son Also Rises: Surnames and the History of Social Mobility*. Princeton University Press.

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

Cui, Y., Che, W., Liu, T., Qin, B., Wang, S., & Hu, G. (2020). Revisiting pre-trained models for Chinese natural language processing. *Findings of EMNLP 2020*, 657–668.

Garg, N., Schiebinger, L., Jurafsky, D., & Zou, J. (2018). Word embeddings quantify 100 years of gender and ethnic stereotypes. *Proceedings of the National Academy of Sciences*, 115(16), E3635–E3644.

Greenwald, A. G., McGhee, D. E., & Schwartz, J. L. K. (1998). Measuring individual differences in implicit cognition: The Implicit Association Test. *Journal of Personality and Social Psychology*, 74(6), 1464–1480.

Levitt, S. D., & Fryer, R. G., Jr. (2004). The causes and consequences of distinctively Black names. *The Quarterly Journal of Economics*, 119(3), 767–805.

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of EMNLP*, 1532–1543.

Shi, L., & Tong, C. (2025). An open dataset of Chinese name-to-gender associations. *Nature Scientific Data*.

Song, Y., Shi, S., Li, J., & Zhang, H. (2018). Directional skip-gram: Explicitly distinguishing left and right context for word embeddings. *Proceedings of NAACL*, 175–180.

Ye, J., & Skiena, S. (2019). MediaRank: Computational ranking of online news sources. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.
