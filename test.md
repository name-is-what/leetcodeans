We sincerely thank you for your constructive and helpful comments. We initially address all your concerns below.

**W1:** Some explanations are insufficient. For example, in the introduction about stage-wise NCD, I think terms like 'incremental' or 'continuous' learning are more commonly used, instead of 'stage-wise.' Please elaborate on the meaning of 'stage-wise' clearly before stating the need for a stage-wise NCD setting. Similarly, the meaning of 'joint' training is also vague. From my understanding, it means that unlabeled and labeled data are jointly trained simultaneously. Before discussing the limitations of joint training, please define its meaning first.

**Answer:** Thank you for pointing out this issue. We have chosen the terms "stage-wise" and "joint training" as they are commonly used as fixed expressions in the NCD setting [1],[2],[3] to distinguish between the usage of labeled old classes and unlabeled new classes at different stages. 
As the NCD setting assumes only the old classes to be labeled, "stage-wise" aims to emphasize the separate utilization of labeled and unlabeled data at different stages. We opted not to use 'incremental' or 'continuous' to differentiate from the concept of incremental learning, while "stage-wise" and "joint training" are concepts in the NCD setting.
The disparity between incremental learning and NCD is elaborately contrasted in Table 1 and illustrated in Figure 1.

Regarding the meaning of joint training, as you mentioned, it means that unlabeled and labeled data are jointly trained simultaneously. In line 82-83 of the manuscript, we briefly discussed it as "clustering unlabeled data and jointly training with labeled data." Since the term "joint training" is widely used in the relevant literature of NCD, we did not fully define it. Once again, we thank you for your suggestion, and we will enhance the explanations of these concepts in the new revision..

[1] Han, Kai, et al. "Automatically Discovering and Learning New Visual Categories with Ranking Statistics." _International Conference on Learning Representations_. 2019.
[2] Han, Kai, Andrea Vedaldi, and Andrew Zisserman. "Learning to discover novel visual categories via deep transfer clustering." _Proceedings of the IEEE/CVF International Conference on Computer Vision_. 2019.
[3] Zhong, Zhun, et al. "Neighborhood contrastive learning for novel class discovery." _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_. 2021.



**W2:** Please clearly state the abbreviation NC-NCD in line 98. It seems like it stands for Node Classification-Novel Class Discovery.

**Answer:** Thank you for pointing this out. We appreciate your attention to detail. It would be more appropriate to write, "In light of the above analysis, we propose a novel class discovery task for node classification on graphs called Node Classification-Novel Class Discovery (NC-NCD)", which will be included in the new revision.


**W3:** There are missing notations: N^u in line 283, and S and P in line 424.

**Answer:** Thank you for pointing out these omissions. The notation $N^u$ in line 283 refers to the number of samples of unlabeled new classes. In line 424, $P$ represents the entire set of elements of the feature dimensions, while $S$ denotes a subset of indices {1, . . . , d} with a specific order. We will address these parts during the revision process.


**Q1:** In Figures 1(c) and (d), why is the red class not included? Is it because the task or class incremental learning has limitations in terms of the number of classes?

**Answer:** Thank you for your inquiry. As noted, traditional incremental learning settings often impose restrictions that the number of new classes learned at each incremental stage should be the same. This limitation is one of the reasons for our NC-NCD setting being more practical. The depiction in Figure 1 is designed to better illustrate that under our setting, there is no need to impose such constraints on the number of new and old classes.



**Q2:** In the NCD setting, do all the newly arrived data belong to novel classes? The current setting, a) not utilizing old samples in the next phase, and b) testing all classes, are realistic. However, the assumption that samples from old classes might not arrive in the new phase is unrealistic. For example, in [1], this is the most crucial question from my side. Previous NCD literature assumes that the classes of labeled and unlabeled sets are disjoint, but starting from GCD [2], it's better to assume that the unlabeled set contains samples from old classes.

**Answer:** Thank you for your valuable comment. Our current work primarily focuses on the NCD setting, emphasizing that only unlabeled new class data are available for learning new classes, while old class data are not available for training. As you mentioned, the scenario where unlabeled data contains both known and unknown classes aligns with the Generalized Category Discovery (GCD) setting, which indeed corresponds to real-world scenarios. However, considering the storage constraints and data privacy issues associated with old class data, our research in the NCD setting remains practical. Moreover, in scalable graph-structured data such as citation networks, forums, and social communities, the emergence of new articles, topics, or users often involves only unlabeled new class data, signifying the novelty of our work. We will discuss GCD as future work in our Conclusion section during the revision. Thank you again for your insightful comment.


**Q3:** Does the current framework consider only one stage of Phase 2, or can it be applied to multiple stages in Phase 2?

**Answer:** Thank you for bringing up this point. In this work, we discuss NCD on graph-structured data, employing a two-stage approach, following the classic NCD setting with 2 stages training. As you mentioned, in the future, we also plan to extend the NC-NCD framework to multiple stages NCD.


**Q4:** The authors mention that one of the limitations in joint training was the memory constraint. However, is the replaying strategy in SWORD free from the memory constraint?

**Answer:** Thank you for this inquiry. There is a slight bias in the expression. In the manuscript, line 86 states "practical issues such as memory constraints or privacy protection," where we actually mean storage constraint. Joint training with data from both old and new classes may raise privacy concerns regarding old class data, as well as additional storage consumption for storing old class data separately. To further clarify the memory constraint of the replaying strategy, we also conducted statistics and examination of the GPU memory usage with the command `print(torch.cuda.memory_summary(device=None, abbreviated=False))`. For instance, in the Cora dataset, the memory consumption during the NCD-training phase is as shown in the table below, indicating that the memory overhead required by our replaying strategy is actually very minimal.

| Allocated Memory | Cur Usage | Peak Usage | Tot Alloc |
| ---------------- | --------- | ---------- | --------- |
| w/o replay       | 18669 KB  | 20155 KB   | 2874 MB   |
| w/ replay        | 18672 KB  | 20159 KB   | 3793 MB   |

**Q5:** In the adjacency matrix, A_i, how can the adjacency matrix be defined for an individual node? Moreover, are the adjacency matrices in the labeled set (Phase 1) and the unlabeled set (Phase 2) the same?

**Answer:** Thank you for your question. The notation for A_i​ is a result of our writing oversight. We will correct it during the revision. In graph deep learning, we don't represent an individual node by an adjacency matrix A_i, but rather use a single adjacency matrix to represent the connectivity of all nodes in the graph. Specifically, we divide the entire graph structure data into two subgraphs based on old and new classes, used for the labeled set (Phase 1) and unlabeled set (Phase 2) respectively. Therefore, their adjacency matrices are different. In this paper, for the sake of clarity and simplicity in representation, we denote the adjacency matrix as A.



**Q6:** Which k is used for pairwise pseudo labeling? The selection of k is important due to error drift. A small k might allow too many 1s in the pseudo label, and a large k might allow for only a few labels. Is this resolved by Section 2.2.3? Please elaborate on the impact of k.

**Answer:** Thank you for pointing that out. In Section 2.2.1 of the manuscript, k is used for calculating the top-k dimensions' similarity and pseudo labels. However, in other parts of our paper, including Section 2.2.3, k is merely used as an index. We apologize for any confusion this may have caused.

Regarding the impact of k, we considered your suggestion and conducted experiments to provide additional insights into the evolution of our method's performance with respect to k, as shown in the table below.

| k   | 1     | 2     | 3     | 10    | 20    | 30    | 50    |
| --- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Old | 64.04 | 62.57 | 60.67 | 65.35 | 66.37 | 64.86 | 66.37 |
| New | 24.05 | 31.65 | 37.97 | 20.89 | 21.56 | 22.16 | 21.56 |
| All | 51.40 | 52.80 | 53.50 | 51.30 | 52.10 | 51.20 | 52.10 |
We observed that when k is small, such as k=3, the performance is relatively high on Old, New, and All categories. As k increases, the performance tends to decrease and stabilize. In this experiment, we chose k=3, and we will elaborate on the impact of k. We will clarify this in the Experimental Setups section during the revision.


**Q7:** Can you explain in detail how L_perturb works? What is the significance of minimizing the distance between the original and perturbed features?

**Answer:** Thank you for your question. $L_{Perturb}$ introduces perturbations primarily to enhance the model's ability to unsupervisedly learn features of new classes. Specifically, like the data augmentation in contrastive learning [4], this allows the model to identify common, as well as crucial and essential features, while filtering out noise interference. Results from ablation experiments in Table 5 demonstrate that under the influence of  $L_{Perturb}$, the model's performance in learning unlabeled new classes and inference on all classes has improved.

[4] Xia, Jun, et al. "Simgrace: A simple framework for graph contrastive learning without data augmentation." _Proceedings of the ACM Web Conference 2022_. 2022.

**Q8:** Is $\Vert C^u \Vert$ assumed to be known?

**Answer:** Thank you for your inquiry. In the NCD setting, the number of novel classes ($C_u$) is generally treated as a prior knowledge. This is because it can often be estimated reliably through various off-the-shelf clustering methods. Most recent works in the field assume the number of novel classes ($C_u$) to be known a priori, as it can be effectively estimated through methods such as semi-supervised k-means.

**Q9:** Is it a common assumption that the class distribution follows a Gaussian distribution, as discussed in line 496?

**Answer:** Your question is appreciated. In line 496 of the manuscript, we discussed "replaying features from the class-specific Gaussian distribution." This indeed suggests the assumption that the **feature distribution** follows a Gaussian distribution [5].

[5] Wu, Tailin, et al. "Graph information bottleneck." _Advances in Neural Information Processing Systems_ 33 (2020): 20437-20448.

We hope we were able to answer your questions. If there are further concerns that keep you from recommending acceptance, please do let us know.
