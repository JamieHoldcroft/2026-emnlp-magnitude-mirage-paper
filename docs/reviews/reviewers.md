The Magnitude Mirage: Rethinking Confidence for Reasoning-Intensive Retrieval
Download PDF
Jamie Holdcroft, Abdelrahman Abdallah, Adam Jatowt
17 Mar 2026 (modified: 10 Apr 2026)
ACL ARR 2026 March Submission
March, Senior Area Chairs, Area Chairs, Reviewers, Authors
Revisions
CC BY 4.0
Keywords: Query Performance Prediction, Retrieval-Augmented Generation, Retrieval Calibration, Abstention, Score Distribution, Reasoning-Intensive Retrieval
Abstract:
Reliable Retrieval-Augmented Generation (RAG) requires detecting when retrieval has failed. Many production RAG systems implement simple similarity thresholding, treating raw retrieval scores as calibrated confidence signals. We expose the \textit{Magnitude Mirage}: as queries transition from simple semantic matching to complex reasoning, neural retrievers often assign high similarity scores to semantically related but constraint-violating documents, causing magnitude thresholding to degrade to near-random abstention performance. To address this without relying on computationally expensive LLM-based evaluators, we conduct a large-scale empirical study evaluating six zero-cost Query Performance Prediction (QPP) metrics across 11 retrieval architectures and 28 datasets. Benchmarking across three cognitive tiers -- semantic matching, logical reasoning, and temporal reasoning -- we show that score-distribution signals consistently outperform raw magnitude thresholding. In particular, Score Gap (
) and Linearized Score Magnitude and Variance (LSMV) improve abstention AUROC by up to 0.16 across models and reasoning tiers. Because these signals operate solely on the score distribution already produced during retrieval, they do not require additional model inference, retraining, or latency, providing a practical zero-cost replacement for magnitude thresholding in deployed RAG systems.

Paper Type: Long
Research Area: Language Modeling
Research Area Keywords: passage retrieval, dense retrieval, re-ranking, retrieval-augmented generation, automatic evaluation, retrieval-augmented generation
Contribution Types: Model analysis & interpretability, NLP engineering experiment
Languages Studied: English
Reassignment Request Area Chair: This is not a resubmission
Reassignment Request Reviewers: This is not a resubmission
A1 Limitations Section: This paper has a limitations section.
A2 Potential Risks: N/A
B Use Or Create Scientific Artifacts: Yes
B4 Data Contains Personally Identifying Info Or Offensive Content: N/A
B6 Statistics For Data: Yes
B6 Elaboration: section 3,4,5
C Computational Experiments: Yes
C2 Experimental Setup And Hyperparameters: Yes
C2 Elaboration: section 3,4,5
C3 Descriptive Statistics: Yes
C3 Elaboration: section 3,4,5
D Human Subjects Including Annotators: No
D1 Instructions Given To Participants: N/A
D2 Recruitment And Payment: N/A
D3 Data Consent: N/A
D4 Ethics Review Board Approval: N/A
E Ai Assistants In Research Or Writing: No
E1 Information About Use Of Ai Assistants: N/A
Author Submission Checklist: yes
Preprint: no
Preprint Status: We plan to release a non-anonymous preprint in the next two months (i.e., during the reviewing process).
Preferred Venue: EMNLP
Consent To Share Data: no
Consent To Share Submission Details: On behalf of all authors, we agree to the terms above to share our submission details.
Association For Computational Linguistics - Blind Submission License Agreement: On behalf of all authors, I agree
Submission Number: 1471
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
8 / 8 replies shown
Add:
Official Review of Submission1471 by Reviewer coAZ
Official Reviewby Reviewer coAZ21 Apr 2026, 17:57 (modified: 03 May 2026, 13:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer coAZRevisions
Paper Summary:
The paper benchmarks six zero-cost QPP metrics (MaxScore, Score Gap, Top-k Std, NQC, Max Iterative Std, LSMV) across 11 retrievers and 28 datasets, grouped into semantic (BEIR), logical (BRIGHT), and temporal (TEMPO) tiers. The main claim is that raw similarity thresholding ("the Magnitude Mirage") is unreliable for reasoning-intensive retrieval, while variance-based signals like Score Gap and LSMV recover useful abstention AUROC at zero extra cost.

Summary Of Strengths:
The empirical scope is large and the BRIGHT and TEMPO coverage is timely. The headline takeaway is operationally useful: if you are thresholding on s1 in production, switching to Score Gap or LSMV is a free and usually positive change. The observation that the gap between MaxScore and any variance-based metric is much larger than the spread among variance-based metrics is a clean and actionable result. Limitations are candid, particularly the acknowledgement that the evaluation stops at the retriever.

Summary Of Weaknesses:
Related work on QPP for neural IR is significantly underspecified. Faggioli's ICTIR 2023 follow-up (Dense-Centroid), Arabzadeh's DenseQPP, Singh et al. 2023 (QPP-PRP), Datta et al. PDQPP, and Meng et al. 2024 "Uncovering the Limitations of QPP" all address QPP for dense retrieval and are not cited. The Cosine Adapter paper (arxiv:2408.04887) directly tackles thresholding on embedding scores and is also missing. This weakens the "it remains unclear whether classical QPP signals are informative" framing. The "Magnitude Mirage" is partially established prior art. Faggioli 2023 already showed magnitude-type QPPs degrade on neural IR. The contribution here is really "extending this to reasoning benchmarks with a broader model suite," which is valuable but should be framed honestly. LSMV is presented as one of two headline metrics, but it is just s1 times σ(Sk), a minor modification of SMV (the log was dropped because neural scores are bounded). Positioning it as a novel contribution is a stretch. NDCG@k > 0 as the success criterion is very lenient, especially at k=25 or 50. The paper should show sensitivity to this, since it likely inflates AUROC on BEIR. Absolute AUROC on BRIGHT tops out around 0.65. The paper calls this "restored reliable abstention," but 0.65 is meaningful signal, not strong signal. The framing should match the numbers. No end-to-end RAG evaluation. The entire motivation is deployed abstention, but we never see that variance-based thresholds actually reduce hallucinations or improve answer quality downstream. Even a small experiment on one benchmark would make the practical claim much stronger.

Comments Suggestions And Typos:
Please expand Related Work to cover the dense-IR QPP line of work. A brief note on how to set thresholds in practice (percentile calibration, etc.) would make "zero-cost replacement" more actionable. Consider one alternative success criterion (e.g., NDCG@k above a threshold, or relevant-in-top-5) in an appendix to show robustness.

Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 3 = Findings: I think this paper could be accepted to the Findings of the ACL.
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review
Add:
Official Comment by Authors
Official Commentby Authors (Jamie Holdcroft, Adam Jatowt, Abdelrahman Abdallah)04 May 2026, 21:51Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer coAZ
Comment:
Thank you for the thorough and constructive review. We address your main concerns below.

Missing related work. We thank the reviewer for these pointers. We note that Faggioli et al. (2023) is already discussed in Section 2.2 (lines 200–208); we will make this more prominent and expand the related work to include DenseQPP, QPP-PRP, PDQPP, Meng et al. (2024), and the Cosine Adapter work.

"Magnitude Mirage" framing. We agree the framing should be sharpened. Our intention was to position the contribution as demonstrating that this failure becomes substantially more pronounced in reasoning-intensive settings, where MaxScore AUROC collapses to 0.520 - 0.611 universally across all models on BRIGHT, a regime prior work did not study. We will make this distinction more explicit.

LSMV contribution. We agree. We will reframe LSMV as a practical adaptation of SMV for bounded neural scores rather than a large methodological contribution.

NDCG@k > 0 criterion. We agree this criterion is relatively lenient. Our intention was to reflect the minimal requirement in RAG settings, where the presence of at least one relevant document can enable correct generation (Lewis et al., 2020; Karpukhin et al., 2020), as discussed in Section 4.3. We will add sensitivity analysis with stricter thresholds in the appendix to demonstrate robustness.

Strength of claims. We agree. We will revise phrasing such as "reliable abstention" to better reflect absolute performance levels (e.g. AUROC ≈ 0.65 on BRIGHT).

End-to-end validation. We agree this is important. We will add a downstream RAG experiment evaluating whether variance-based thresholds improve answer quality and reduce hallucination relative to magnitude thresholding.

Add:
Official Review of Submission1471 by Reviewer bGgL
Official Reviewby Reviewer bGgL20 Apr 2026, 13:08 (modified: 03 May 2026, 13:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer bGgLRevisions
Paper Summary:
This paper identifies the Magnitude Mirage: raw retrieval score thresholding (s₁ > τ), widely used in production RAG pipelines, collapses to near-random performance on reasoning-intensive queries because neural retrievers confidently assign high scores to semantically related but constraint-violating documents. Through a large-scale empirical study across 11 retrievers, 28 datasets, and three cognitive tiers (semantic, logical, temporal), the authors show that classical variance-based QPP metrics—particularly Score Gap (s₁ − sₖ) and LSMV—consistently restore calibration, improving abstention AUROC by up to 0.16. Notably, the gain from abandoning magnitude far exceeds the differences among distributional alternatives, suggesting the paradigm shift itself is what matters, and these zero-cost signals can drop-in replace magnitude thresholding without extra inference or latency.

Summary Of Strengths:
The paper identifies a widely used practice—raw score thresholding in production RAG systems—and convincingly articulates why it breaks down under reasoning-intensive retrieval, giving the work clear practical relevance.
The evaluation spans 11 retrievers, 28 datasets, and three cognitive tiers with multiple cutoffs and three complementary metrics (AUROC, Pearson, Spearman), providing strong breadth and making the observed patterns hard to attribute to a specific model or dataset.
The proposed variance-based signals operate purely on already-available top-k retrieval scores, requiring no additional inference, retraining, or latency, which makes the recommendation easy to adopt in existing RAG pipelines.
Summary Of Weaknesses:
The paper shows that distribution-based metrics improve AUROC, but it does not evaluate how this translates into actual abstention behavior at concrete operating points. Since real RAG systems care about answered-query accuracy, abstention rate, and precision–recall trade-offs under a chosen threshold, the current evaluation stops short of demonstrating the full practical benefit of the proposed confidence signals.
Reasoning difficulty is not fully disentangled from benchmark-family effects. The paper interprets the semantic → temporal/logical performance drop as evidence that reasoning-intensive retrieval breaks magnitude-based confidence. Section 6.2 partially addresses this by showing that relevant document density is comparable across benchmarks, but other potential confounders such as corpus size, annotation protocols, and query distributions are not discussed. This makes it difficult to fully isolate whether the reported degradation is caused by reasoning complexity itself or by broader dataset differences.
The main tables report benchmark-level AUROC averages, and while Figure 3 illustrates results on one BRIGHT subset (TheoremQA), systematic per-subset breakdowns across the 12 BRIGHT and 12 TEMPO datasets are not provided. Without such breakdowns, uncertainty estimates, or paired significance tests, it is difficult to assess whether the gains of Score Gap and LSMV are broadly consistent or concentrated in a smaller number of favorable subsets.
Comments Suggestions And Typos:
For weakness 3: Adding AUROC tables for each of the 12 BRIGHT and 12 TEMPO subsets, along with bootstrap confidence intervals, would make it easier to assess whether the observed gains are broadly consistent across reasoning types.
Confidence: 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Excitement: 2.5
Overall Assessment: 2.5 = Borderline Findings
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits
Add:
Official Comment by Authors
Official Commentby Authors (Jamie Holdcroft, Adam Jatowt, Abdelrahman Abdallah)04 May 2026, 21:52Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer bGgL
Comment:
Thank you for the careful and balanced review. We address each weakness below.

W1: Operating-point evaluation. We agree that evaluating concrete thresholds would strengthen the practical case. We will add a calibration analysis showing precision-recall trade-offs at representative operating points, making the impact on abstention behavior more explicit for system designers.

W2: Reasoning difficulty vs. dataset effects. We agree that additional confounders beyond document density may exist. Section 6.2 addresses corpus sparsity (showing comparable relevant document density across tiers), but we will expand the discussion to better separate reasoning complexity from broader dataset differences and clarify remaining limitations.

W3: Per-subset analysis and uncertainty. We agree that more granular reporting would strengthen the paper. We will add per-subset AUROC tables for all 12 BRIGHT and 12 TEMPO datasets along with bootstrap confidence intervals in the appendix, making the consistency of gains across reasoning types directly assessable.

Add:
 Replying to Official Comment by Authors
Official Comment by Reviewer bGgL
Official Commentby Reviewer bGgL05 May 2026, 09:53Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer bGgL
Comment:
Thank you for the authors’ response. After reviewing it, my scores remain unchanged.

Add:
Official Review of Submission1471 by Reviewer LuRs
Official Reviewby Reviewer LuRs09 Apr 2026, 19:49 (modified: 03 May 2026, 13:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer LuRsRevisions
Paper Summary:
This paper investigates the problem of retrieval confidence estimation in RAG systems. The authors point out a flaw in the currently widely used magnitude thresholding: in reasoning-intensive retrieval scenarios, a high similarity score does not necessarily indicate a correct retrieval. Through large-scale experiments on 11 retrieval models and 28 datasets, the paper evaluates various Query Performance Prediction methods. The results show that variance-based metrics based on score distribution, such as Score Gap and LSMV, outperform the traditional MaxScore.

Summary Of Strengths:
The proposed Magnitude Mirage phenomenon is clearly defined. Experiments with different cognitive tiers demonstrate the systematic failure of MaxScore in complex tasks.
The experiments are large-scale, covering 11 retrieval architectures and 28 datasets, and distinguishing between semantic, logical, and temporal tasks, demonstrating a comprehensive design.
The methods are simple and practical. Score Gap and LSMV do not require additional inference or retraining, making them valuable for engineering applications.
Summary Of Weaknesses:
The evidence for the core conclusions remains insufficient. The paper primarily relies on retrieval-level metrics such as AUROC and correlation, but fails to validate whether these metrics truly reduce hallucination or improve final answer quality in end-to-end RAG tasks.
The definition of retrieval success is weak. The paper uses NDCG@k > 0 as the success criterion, requiring only the existence of one relevant document. This criterion is too lenient and fails to reflect the real needs in complex reasoning tasks, potentially overestimating the method's effectiveness.
The method's innovation is limited. Score Gap and variance-based QPP methods are derived from existing information retrieval research. The paper's main contribution is revalidating their effectiveness in neural retrieval scenarios, but it lacks new modeling methods or theoretical analysis.
The analysis of the failure mechanism is not in-depth enough. The paper attributes the problem to a mismatch between semantic similarity and constraint satisfaction, but lacks empirical analysis of the retriever training objective or embedding space, leaving the explanation at the phenomenological level.
Comments Suggestions And Typos:
Suggested Additional References
Query Performance Prediction for Neural IR: Are We There Yet? This paper provides a systematic evaluation of QPP methods in neural retrieval settings and is directly relevant to the evaluation framework adopted in this work.
Disco-RAG: Discourse-Aware Retrieval-Augmented Generation This work studies structural signals in RAG and provides a complementary perspective on improving retrieval reliability beyond score-based confidence estimation.
NoMIRACL: Knowing When You Don’t Know for Robust Multilingual Retrieval-Augmented Generation This work studies abstention and uncertainty in RAG systems, which is closely related to the confidence estimation problem addressed in this paper.
The Power of Noise: Redefining Retrieval for RAG Systems This paper analyzes how retrieval noise affects RAG performance and provides a broader system-level perspective on retrieval reliability.
Confidence: 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 2.5
Excitement: 2.5
Overall Assessment: 2.5 = Borderline Findings
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 2 = They would be hard pressed to reproduce the results: The contribution depends on data that are simply not available outside the author's institution or consortium and/or not enough details are provided.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits
Add:
Official Comment by Authors
Official Commentby Authors (Jamie Holdcroft, Adam Jatowt, Abdelrahman Abdallah)04 May 2026, 21:52Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer LuRs
Comment:
Thank you for the constructive feedback. We address each concern below.

End-to-end RAG validation. We agree this is the most significant limitation, and we acknowledge it explicitly in the Limitations section (lines 634-644). We will add a downstream QA experiment evaluating whether variance-based abstention improves answer quality and reduces hallucination relative to magnitude thresholding.

Success criterion. We agree NDCG@k > 0 is relatively lenient. Our intention was to reflect the minimal RAG requirement (Section 4.3), but we will add stricter criteria and sensitivity analysis in the appendix to demonstrate robustness.

Methodological novelty. We agree the individual signals are not new, this is intentional. We will revise the framing to more clearly emphasise the contribution as large-scale empirical validation in reasoning-intensive retrieval, and the demonstration that the paradigm shift from magnitude to distributional estimation matters more than the specific metric chosen.

Mechanistic analysis. We agree the explanation can be deepened. We will extend the analysis with additional examination of score distribution behavior across tiers to better support the underlying intuition.

Suggested references. We thank the reviewer for these suggestions. We note that Faggioli et al. ("Are We There Yet?") is already cited in Section 2.2, NoMIRACL (Thakur et al., 2023) is already cited at line 824, and "The Power of Noise" (Cuconasu et al., 2024) is already cited at line 691, however, we will make these connections more explicit. We will review Disco-RAG and include it if the fit is appropriate.

Add:
 Replying to Official Comment by Authors
Official Comment by Reviewer LuRs
Official Commentby Reviewer LuRs04 May 2026, 22:18Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer LuRs
Comment:
Thanks for the response. I appreciate the authors' clarification and the planned additional analyses. I have no further comments at this stage.

Add:
