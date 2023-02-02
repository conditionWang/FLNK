# Dynamic Federated Learning
This is all you need for a brand new but interesting topic -- dynamic federated learning (DFL), including research papers, datasets, tools, and you name it. Standard federated learning usually assumes that the data distribution, model structure, and participating clients of the overall framework are fixed to unchanged over time. But DFL explores research problems of training better models (global or personalized) when the aforementioned important elements of FL are dynamically changing. Relevant topics include heterogeneous federated learning, personalized federated learning, incremental learning, continual learning, domain adaptation, domain generalization and self-supervised learning. We believe dynamic federated learning will be a practical mechanism that can really enable federated learning to be applied in the real world.

## Research Papers

### Survey
+ Non-IID data and Continual Learning processes in Federated Learning: A long road ahead [[paper]](https://www.sciencedirect.com/science/article/pii/S1566253522000884)

### Papers
#### Computer Vision
+ Partitioned Variational Inference: A unified framework encompassing federated and continual learning (Arxiv 2018) [[paper]](https://arxiv.org/abs/1811.11206)
+ Federated and continual learning for classification tasks in a society of devices (Arxiv 2020) [[paper]](https://arxiv.org/abs/2006.07129)
+ **Federated Continual Learning with Weighted Inter-client Transfer (ICML 2021)** [[paper]](https://proceedings.mlr.press/v139/yoon21b.html?ref=https://githubhelp.com) [[code]](https://github.com/wyjeong/FedWeIT)
+ **Federated Class-Incremental Learning (CVPR 2022)** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Federated_Class-Incremental_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/conditionWang/FCIL)
+ **Learn From Others and Be Yourself in Heterogeneous Federated Learning (CVPR 2022)** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.html) [[code]](https://github.com/WenkeHuang/FCCL)
+ Concept drift detection and adaptation for federated and continual learning (Multimedia Tools and Applications 2021) [[paper]](https://link.springer.com/article/10.1007/s11042-021-11219-x)
+ Federated Reconnaissance: Efficient, Distributed, Class-Incremental Learning (NeurIPS 2021 Workshop) [[paper]](https://arxiv.org/abs/2109.00150) [[code]](https://github.com/ml4ai/fed-recon)
+ ODE: A Data Sampling Method for Practical Federated Learning with Streaming Data and Limited Buffer (Arxiv 2022) [[paper]](https://arxiv.org/abs/2209.00195)
+ Diurnal or Nocturnal? Federated Learning from Periodically Shifting Distributions (NeurIPS 2022 Workshop) [[paper]](https://openreview.net/forum?id=WRmTnEOk0E)
+ Knowledge Lock: Overcoming Catastrophic Forgetting in Federated Learning (PAKDD 2022) [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-05933-9_47)
+ **Continual Federated Learning Based on Knowledge Distillation (IJCAI 2022)** [[paper]](https://www.ijcai.org/proceedings/2022/0303.pdf) [[code]](https://github.com/lianziqt/CFeD)
+ A Federated Incremental Learning Algorithm Based on Dual Attention Mechanism (Applied Science 2022) [[paper]](https://www.mdpi.com/2076-3417/12/19/10025)
+ Tackling Dynamics in Federated Incremental Learning with Variational Embedding Rehearsal (Arxiv 2021) [[paper]](https://arxiv.org/abs/2110.09695)
+ Federated Continuous Learning With Broad Network Architecture (IEEE Transactions on Cybernetics 2022) [[paper]](https://ieeexplore.ieee.org/abstract/document/9477571?casa_token=m7gcGPrMbPsAAAAA:5GGy8hdewYfNFYj6UMTFGgiyzIa9g9VkyTts8CoeCnfikxULR0kML733vV-K6InUQZ1_CDPPefw)
+ Addressing Client Drift in Federated Continual Learning with Adaptive Optimization (Preprint 2022) [[paper]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4188586)
+ Towards Federated Learning on Time-Evolving Heterogeneous Data (Arxiv 2022) [[paper]](https://arxiv.org/abs/2112.13246)
+ Continual Horizontal Federated Learning for Heterogeneous Data (Arxiv 2022) [[paper]](https://arxiv.org/abs/2203.02108)
+ Better Generative Replay for Continual Federated Learning (ICLR 2023 under review) [[paper]](https://openreview.net/forum?id=cRxYWKiTan)
+ Federated Learning for Data Streams (Arxiv 2023) [[paper]](https://arxiv.org/abs/2301.01542)
+ FedKNOW: Federated Continual Learning with Signature Task Knowledge Integration at Edge (Arxiv 2022) [[paper]](https://arxiv.org/pdf/2212.01738.pdf) [[code]](https://github.com/LINC-BIT/FedKNOW)


#### Natural Language Processing
+ **Federated Continual Learning for Text Classification via Selective Inter-client Transfer (EMNLP Findings 2022)** [[paper]](https://arxiv.org/abs/2210.06101)

#### Audio and IoT
+ A distillation-based approach integrating continual learning and federated learning for pervasive services (Arxiv 2021) [[paper]](https://arxiv.org/abs/2109.04197)
+ **FedSpeech: Federated Text-to-Speech with Continual Learning (IJCAI 2021)** [[paper]](https://arxiv.org/abs/2110.07216)
+ Spatial-Temporal Federated Learning for Lifelong Person Re-identification on Distributed Edges (Arxiv 2022) [[paper]](https://arxiv.org/abs/2207.11759)
+ **Learnings from Federated Learning in The Real World (ICASSP 2022)** [[paper]](https://ieeexplore.ieee.org/abstract/document/9747113?casa_token=-JC76TB47JIAAAAA:03kp3BFvulzlDEFq5UZ1pJUHKz35zmww2hZXmzLk1YHIKW75ec1wkSH5WDtTkOfM6gjLSd_Bq-U)
+ New Generation Federated Learning (Sensors 2022) [[paper]](https://www.mdpi.com/1424-8220/22/21/8475)
+ Attention-based federated incremental learning for traffic classification in the Internet of Things (Computer Communications 2022) [[paper]](https://www.sciencedirect.com/science/article/pii/S0140366422000123?casa_token=lB1i8C4Mud0AAAAA:LJNPcUuOpesrSeQsD6BHlwVs4orzTgLmuxXbTDBes3HFdFat1w78hfyrWUVYiJK4QpExu-wFZZN2)
+ Federated Learning and catastrophic forgetting in pervasive computing: demonstration in HAR domain (IEEE Workshop 2022) [[paper]](https://ieeexplore.ieee.org/abstract/document/9767246)
+ Federated Continual Learning through distillation in pervasive computing (SmartComp 2022) [[paper]](https://ieeexplore.ieee.org/abstract/document/9821057?casa_token=qAwx-o_ga4gAAAAA:MFqUbWBzqkfJ79QWU5yMvPmvFoG_T-pzAVdPEABcEiciymal0kAy5Ie1BCowtHyELCCNOtGbSyk)
+ DILoCC: An approach for Distributed Incremental Learning across the Computing Continuum (SmartComp 2022) [[paper]](https://ieeexplore.ieee.org/abstract/document/9556258?casa_token=uIv0gtgLWhEAAAAA:j592VhM8vYz0R__phIyBvnx5YQEtwrPAJiiZ16qu9nu2wu3jYr8xIfodzm5OpUn2NwwaPbYx8co)
+ **Cross-FCL: Toward a Cross-edge Federated Continual Learning Framework in Mobile Edge Computing Systems (TMC 2022)** [[paper]](https://ieeexplore.ieee.org/abstract/document/9960821?casa_token=1Ovr4510alIAAAAA:53TSbfLNHX8M5eh-2p65eXO2F7vbB4rAXIFAudCG92EAkPlFhecA5e0emL2r0gUBb5tvT9ePpoE)

#### Security Relevant
+ GFCL: A GRU-based Federated Continual Learning Framework against Data Poisoning Attacks in IoV (Arxiv 2022) [[paper]](https://arxiv.org/abs/2204.11010)
+ Towards a Defense against Backdoor Attacks in Continual Federated Learning (Arxiv 2022) [[paper]](https://arxiv.org/abs/2205.11736)
+ Federated Continual Learning with Differentially Private Data Sharing (NeurIPS 2022 Workshop) [[paper]](https://openreview.net/forum?id=b7vu9ukdpdL)

#### Other Topics
+ Federated Continual Learning to Detect Accounting Anomalies in Financial Auditing (Arxiv 2022) [[paper]](https://arxiv.org/abs/2210.15051)
+ Continual Learning of Dynamical Systems with Competitive Federated Reservoir Computing (Arxiv 2022) [[paper]](https://arxiv.org/abs/2206.13336)
+ Towards Lifelong Federated Learning in Autonomous Mobile Robots with Continuous Sim-to-Real Transfer (Arxiv 2022) [[paper]](https://arxiv.org/abs/2205.15496)

## Datasets

## Tools






