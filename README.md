# Federated Learning with New Knowledge
This is all you need for a brand new but interesting topic -- dynamic federated learning (DFL), including research papers, datasets, tools, and you name it. Standard federated learning usually assumes that the data distribution, model structure, and participating clients of the overall framework are fixed to unchanged over time. However, DFL explores research problems of training better models (global or personalized) when the aforementioned important elements of FL are dynamically changing. Relevant topics include heterogeneous federated learning, personalized federated learning, incremental learning, continual learning, domain adaptation, and out-of-distribution generalization. We believe dynamic federated learning will be a practical mechanism that can really enable federated learning to be applied in the real world.

- [Computer Vision](#computer-vision)
  - [Pure Classification](#pure-classification)
  - [Advanced CV Tasks (object detection, semantic segmentation)](#advanced-cv-tasks)
  - [Out-of-Distribution Learning (domain adaptation, domain generalization, out-of-distribution detection)](#ood-learning)
- [Natural Language Processing](#nlp)
- [Audio and IoT](#iot)
- [Security Relevant](#security)
- [Other Topics](#other)


<a name="computer-vision" />

## Computer Vision
+ Non-IID data and Continual Learning processes in Federated Learning: A long road ahead 
  + (Survey, Information Fusion 2022) [[paper]](https://www.sciencedirect.com/science/article/pii/S1566253522000884)
 
<a name="pure-classification" />

### _Pure Classification_
+ Partitioned Variational Inference: A unified framework encompassing federated and continual learning 
  + (Arxiv 2018) [[paper]](https://arxiv.org/abs/1811.11206)
+ Federated and continual learning for classification tasks in a society of devices 
  + (Arxiv 2020) [[paper]](https://arxiv.org/abs/2006.07129)
+ A New Analysis Framework for Federated Learning on Time-Evolving Heterogeneous Data 
  + (FL-ICML2021) [[paper]](https://fl-icml.github.io/2021/papers/FL-ICML21_paper_47.pdf)
+ **Federated Continual Learning with Weighted Inter-client Transfer**  
  + **(ICML 2021)** [[paper]](https://proceedings.mlr.press/v139/yoon21b.html?ref=https://githubhelp.com) [[code]](https://github.com/wyjeong/FedWeIT)
+ **Federated Class-Incremental Learning** 
  + **(CVPR 2022)** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Federated_Class-Incremental_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/conditionWang/FCIL)
+ **Learn From Others and Be Yourself in Heterogeneous Federated Learning** 
  + **(CVPR 2022)** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.html) [[code]](https://github.com/WenkeHuang/FCCL)
+ Towards Federated Learning on Time-Evolving Heterogeneous Data 
  + (Arxiv 2021) [[paper]](https://arxiv.org/pdf/2112.13246.pdf)
+ Concept drift detection and adaptation for federated and continual learning 
  + (Multimedia Tools and Applications 2021) [[paper]](https://link.springer.com/article/10.1007/s11042-021-11219-x)
+ Federated Reconnaissance: Efficient, Distributed, Class-Incremental Learning 
  + (NeurIPS 2021 Workshop) [[paper]](https://arxiv.org/abs/2109.00150) [[code]](https://github.com/ml4ai/fed-recon)
+ ODE: A Data Sampling Method for Practical Federated Learning with Streaming Data and Limited Buffer 
  + (Arxiv 2022) [[paper]](https://arxiv.org/abs/2209.00195)
+ Diurnal or Nocturnal? Federated Learning from Periodically Shifting Distributions 
  + (NeurIPS 2022 Workshop) [[paper]](https://openreview.net/forum?id=WRmTnEOk0E)
+ Knowledge Lock: Overcoming Catastrophic Forgetting in Federated Learning 
  + (PAKDD 2022) [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-05933-9_47)
+ **Continual Federated Learning Based on Knowledge Distillation** 
  + **(IJCAI 2022)** [[paper]](https://www.ijcai.org/proceedings/2022/0303.pdf) [[code]](https://github.com/lianziqt/CFeD)
+ A Federated Incremental Learning Algorithm Based on Dual Attention Mechanism 
  + (Applied Science 2022) [[paper]](https://www.mdpi.com/2076-3417/12/19/10025)
+ Tackling Dynamics in Federated Incremental Learning with Variational Embedding Rehearsal 
  + (Arxiv 2021) [[paper]](https://arxiv.org/abs/2110.09695)
+ Federated Continuous Learning With Broad Network Architecture 
  + (IEEE Transactions on Cybernetics 2022) [[paper]](https://ieeexplore.ieee.org/abstract/document/9477571?casa_token=m7gcGPrMbPsAAAAA:5GGy8hdewYfNFYj6UMTFGgiyzIa9g9VkyTts8CoeCnfikxULR0kML733vV-K6InUQZ1_CDPPefw)
+ Addressing Client Drift in Federated Continual Learning with Adaptive Optimization 
  + (Preprint 2022) [[paper]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4188586)
+ Continual Horizontal Federated Learning for Heterogeneous Data 
  + (Arxiv 2022) [[paper]](https://arxiv.org/abs/2203.02108)
+ Online Federated Learning via Non-Stationary Detection and Adaptation amidst Concept Drift 
  + (Arxiv 2022) [[paper]](https://arxiv.org/pdf/2211.12578.pdf)
+ **Better Generative Replay for Continual Federated Learning** 
  + **(ICLR 2023)** [[paper]](https://openreview.net/forum?id=cRxYWKiTan)
+ Federated Learning for Data Streams 
  + (Arxiv 2023) [[paper]](https://arxiv.org/abs/2301.01542)
+ FedKNOW: Federated Continual Learning with Signature Task Knowledge Integration at Edge 
  + (Arxiv 2022) [[paper]](https://arxiv.org/pdf/2212.01738.pdf)
+ No One Left Behind: Real-World Federated Class-Incremental Learning 
  + (Arxiv 2023) [[paper]](https://arxiv.org/abs/2302.00903) [[code]](https://github.com/JiahuaDong/LGA)
+ Federated probability memory recall for federated continual learning 
  + (Info Science 2023) [[paper]](https://www.sciencedirect.com/science/article/pii/S0020025523001883?casa_token=Srn81YlRjF4AAAAA:Jw28ekpauxEeC4-kxJrzhRPpHV0dTJeInJ-s3mRxOi77YbXShvkg43119RHHjnO2qQ9wOSlRVyUx)
+ GradMA: A Gradient-Memory-based Accelerated Federated Learning with Alleviated Catastrophic Forgetting 
  + (Arxiv 2023) [[paper]](https://arxiv.org/abs/2302.14307)
+ **Addressing Catastrophic Forgetting in Federated Class-Continual Learning**
  + **(ICCV 2023)** [[paper]](https://arxiv.org/abs/2303.06937)
+ **Federated Learning under Distributed Concept Drift** 
  + **(AISTATS 2023)** [[paper]](https://arxiv.org/pdf/2206.00799.pdf)
+ Asynchronous Federated Continual Learning 
  + (CVPR FedVision Workshop 2023) [[paper]](https://arxiv.org/abs/2304.03626)
+ Re-Weighted Softmax Cross-Entropy to Control Forgetting in Federated Learning 
  + (Arxiv 2023) [[paper]](https://arxiv.org/abs/2304.05260)
+ **Distributed Offline Policy Optimization Over Batch Data** 
  + **(AISTATS 2023)** [[paper]](https://proceedings.mlr.press/v206/shen23b.html)
+ CoDeC: Communication-Efficient Decentralized Continual Learning 
  + (Arxiv 2023) [[paper]](https://arxiv.org/abs/2303.15378)
+ Ensemble and continual federated learning for classification tasks 
  + (Machine Learning 2023) [[paper]](https://link.springer.com/article/10.1007/s10994-023-06330-z)
+ **To Store or Not? Online Data Selection for Federated Learning with Limited Storage** 
  + **(WWW 2023)** [[paper]](https://dl.acm.org/doi/abs/10.1145/3543507.3583426)
+ Masked Autoencoders are Efficient Continual Federated Learners
  + (Arxiv 2023) [[paper]](https://arxiv.org/pdf/2306.03542.pdf)
+ Semi-supervised federated learning on evolving data streams
  + (Information Sciences 2023) [[paper]](https://www.sciencedirect.com/science/article/pii/S0020025523008204)
+ A federated learning-based approach to recognize subjects at a high risk of hypertension in a non-stationary scenario
  + (Information Sciences 2023) [[paper]](https://www.sciencedirect.com/science/article/pii/S0020025522014384)
+ Fed-CPrompt: Contrastive Prompt for Rehearsal-Free Federated Continual Learning
  + (FL-ICML 2023) [[paper]](https://arxiv.org/abs/2307.04869)
+ Don't Memorize; Mimic The Past: Federated Class Incremental Learning Without Episodic Memory
  + (FL-ICML 2023) [[paper]](https://arxiv.org/abs/2307.00497)
+ **FedET: A Communication-Efficient Federated Class-Incremental Learning Framework Based on Enhanced Transformer**
  + **(IJCAI 2023)** [[paper]](https://arxiv.org/abs/2306.15347)
+ Federated Multi-Phase Curriculum Learning to Synchronously Correlate User Heterogeneity
  + (IEEE Transactions on Artificial Intelligence 2023) [[paper]](https://ieeexplore.ieee.org/abstract/document/10227551)
+ Federated Class-Incremental Learning with Prompting
  + (Arxiv 2023) [[paper]](https://arxiv.org/abs/2310.08948)
+ FedRCIL: Federated Knowledge Distillation for Representation based Contrastive Incremental Learning
  + (ICCV Workshop 2023) [[paper]](https://openaccess.thecvf.com/content/ICCV2023W/VCL/html/Psaltis_FedRCIL_Federated_Knowledge_Distillation_for_Representation_based_Contrastive_Incremental_Learning_ICCVW_2023_paper.html) [[code]](https://github.com/chatzikon/FedRCIL/tree/main)
+ Distributed Continual Learning with CoCoA in High-dimensional Linear Regression
  + (Arxiv 2023) [[paper]](https://arxiv.org/abs/2312.01795)
+ Concept Matching: Clustering-based Federated Continual Learning
  + (Arxiv 2023) [[paper]](https://arxiv.org/abs/2311.06921)
+ Spatial-Temporal Federated Learning for Lifelong Person Re-identification on Distributed Edges
  + (TCSVT 2023) [[paper]](https://ieeexplore.ieee.org/abstract/document/10142016?casa_token=Ii8P8kcPlY4AAAAA:MY2XgwufZ-bXHTyAv1X8uPjamUxtVYSRkDi80NIBVizQRhgu80UFdUfznGvvReZFpRRFdYfB-zo)
+ HePCo: Data-Free Heterogeneous Prompt Consolidation for Continual Federated Learning
  + (Arxiv 2023) [[paper]](https://arxiv.org/abs/2306.09970)
+ Decentralized Deep Learning under Distributed Concept Drift: A Novel Approach to Dealing with Changes in Data Distributions Over Clients and Over Time
  + (MS Thesis) [[paper]](https://odr.chalmers.se/items/3c39a17d-9d4f-45d9-8745-157711bcb100)
+ **A Data-Free Approach to Mitigate Catastrophic Forgetting in Federated Class Incremental Learning for Vision Tasks**
  + **(NeurIPS 2023)** [[paper]](https://arxiv.org/pdf/2311.07784.pdf)
+ Accurate Forgetting for Heterogeneous Federated Continual Learning
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=ShQrnAsbPI)
+ Variational Federated Continual Learning
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=lzt60v45V4)
+ Towards Out-of-federation Generalization in Federated Learning
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=70PPJo3DwI)
+ FedGP: Buffer-based Gradient Projection for Continual Federated Learning
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=Xi7UoErFRt)
+ Traceable Federated Continual Learning
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=OkZ5UrVpo6)
+ Prototypes-Injected Prompt for Federated Class Incremental Learning
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=cwN69teRIW)

<a name="advanced-cv-tasks" />

### _Advanced CV Tasks (object detection, semantic segmentation)_
+ **FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space** 
  + **(CVPR 2021)** [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.html) [[code]](https://github.com/liuquande/FedDG-ELCFS)
+ **Federated Incremental Semantic Segmentation** 
  + **(CVPR 2023)** [[paper]](https://arxiv.org/abs/2304.04620) [[code]](https://github.com/JiahuaDong/FISS)

<a name="ood-learning" />

### _Out-of-Distribution Learning (domain adaptation, domain generalization, out-of-distribution detection)_
+ Uncertainty-Aware Aggregation for Federated Open Set Domain Adaptation 
  + (TNNLS 2022) [[paper]](https://ieeexplore.ieee.org/document/9931728)
+ **FedSR: A Simple and Effective Domain Generalization Method for Federated Learning** 
  + **(NeurIPS 2022)** [[paper]](https://openreview.net/forum?id=mrt90D00aQX) [[code]](https://github.com/atuannguyen/FedSR)
+ Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer 
  + (WACV 2023) [[paper]](https://openaccess.thecvf.com/content/WACV2023/html/Chen_Federated_Domain_Generalization_for_Image_Recognition_via_Cross-Client_Style_Transfer_WACV_2023_paper.html) [[code]](https://github.com/JeremyCJM/CCST)
+ **Federated Domain Generalization with Generalization Adjustment** 
  + **(CVPR 2023)** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Federated_Domain_Generalization_With_Generalization_Adjustment_CVPR_2023_paper.pdf) [[code]](https://github.com/MediaBrain-SJTU/FedDG-GA)
+ **Rethinking Federated Learning with Domain Shift: A Prototype View**
  + **(CVPR 2023)** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf) [[code]](https://github.com/WenkeHuang/RethinkFL)
+ **Turning the Curse of Heterogeneity in Federated Learning into a Blessing for Out-of-Distribution Detection** 
  + **(ICLR 2023)** [[paper]](https://openreview.net/forum?id=mMNimwRb7Gr) [[code]](https://github.com/illidanlab/FOSTER)
+ **Test-Time Robust Personalization for Federated Learning** 
  + **(ICLR 2023)** [[paper]](https://arxiv.org/pdf/2205.10920.pdf) [[code]](https://github.com/LINs-lab/FedTHE)
+ PerAda: Parameter-Efficient and Generalizable Federated Learning Personalization with Guarantees 
  + (Arxiv 2023) [[paper]](https://arxiv.org/abs/2302.06637)
+ FedConceptEM: Robust Federated Learning Under Diverse Distribution Shifts 
  + (Arxiv 2023) [[paper]](https://arxiv.org/pdf/2301.12379.pdf)
+ MEC-DA: Memory-Efficient Collaborative Domain Adaptation for Mobile Edge Devices
  + (IEEE Transactions on Mobile Computing 2023) [[paper]](https://ieeexplore.ieee.org/abstract/document/10144399?casa_token=Maob5d_6oe4AAAAA:DIFvUdy6hD5HL6KicYdwMt1jMBoo-NJ_J1gpYstTodVDKucbk5XousyqqXBukB9LL_qt3Y_R2W8)
+ FedRC: Tackling Diverse Distribution Shifts Challenge in Federated Learning by Robust Clustering
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=6FAH0SgQzO)
+ A Lightweight Method for Tackling Unknown Participation Statistics in Federated Averaging
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=ZKEuFKfCKA)
+ Benchmarking Algorithms for Federated Domain Generalization
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=wprSv7ichW)
+ Principled Federated Domain Adaptation: Gradient Projection and Auto-Weighting
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=6J3ehSUrMU)
+ FedNovel: Federated Novel Class Learning
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=Unz9zYdjTt)
+ Federated Generalization via Information-Theoretic Distribution Diversification
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=VRCh74Liu9)
+ FedOD: Federated Outlier Detection via Neural Approximation
  + (ICLR 2024 submission) [[paper]](https://openreview.net/forum?id=BhxsjonZ0z)

<a name="nlp" />

## Natural Language Processing
+ Federated Learning Of Out-Of-Vocabulary Words 
  + (Arxiv 2019) [[paper]](https://arxiv.org/abs/1903.10635)
+ **Federated Continual Learning for Text Classification via Selective Inter-client Transfer** 
  + **(EMNLP Findings 2022)** [[paper]](https://arxiv.org/abs/2210.06101)
+ **Quantifying Catastrophic Forgetting in Continual Federated Learning**
  + **(ICASSP 2023)** [[paper]](https://ieeexplore.ieee.org/abstract/document/10097140?casa_token=w10yvdklDzEAAAAA:Bn2W7xONHMDoOKccJxJ2jtYe0yphLEvWpupQg2fLOJjw8I0x6y_uD5JUO8o6bW_bsggroWOA6IU)
+ FedET: A Communication-Efficient Federated Class-Incremental Learning Framework Based on Enhanced Transformer
  + (Arxiv 2023) [[paper]](https://arxiv.org/pdf/2306.15347.pdf)
+ **Coordinated Replay Sample Selection for Continual Federated Learning**
  + **(EMNLP 2023)** [[paper]](https://arxiv.org/pdf/2310.15054.pdf)

<a name="iot" />

## Audio and IoT
+ A distillation-based approach integrating continual learning and federated learning for pervasive services 
  + (Arxiv 2021) [[paper]](https://arxiv.org/abs/2109.04197)
+ **FedSpeech: Federated Text-to-Speech with Continual Learning** 
  + **(IJCAI 2021)** [[paper]](https://arxiv.org/abs/2110.07216)
+ Spatial-Temporal Federated Learning for Lifelong Person Re-identification on Distributed Edges 
  + (Arxiv 2022) [[paper]](https://arxiv.org/abs/2207.11759)
+ **Learnings from Federated Learning in The Real World** 
  + **(ICASSP 2022)** [[paper]](https://ieeexplore.ieee.org/abstract/document/9747113?casa_token=-JC76TB47JIAAAAA:03kp3BFvulzlDEFq5UZ1pJUHKz35zmww2hZXmzLk1YHIKW75ec1wkSH5WDtTkOfM6gjLSd_Bq-U)
+ New Generation Federated Learning 
  + (Sensors 2022) [[paper]](https://www.mdpi.com/1424-8220/22/21/8475)
+ Attention-based federated incremental learning for traffic classification in the Internet of Things 
  + (Computer Communications 2022) [[paper]](https://www.sciencedirect.com/science/article/pii/S0140366422000123?casa_token=lB1i8C4Mud0AAAAA:LJNPcUuOpesrSeQsD6BHlwVs4orzTgLmuxXbTDBes3HFdFat1w78hfyrWUVYiJK4QpExu-wFZZN2)
+ Federated Learning and catastrophic forgetting in pervasive computing: demonstration in HAR domain 
  + (IEEE Workshop 2022) [[paper]](https://ieeexplore.ieee.org/abstract/document/9767246)
+ Federated Continual Learning through distillation in pervasive computing 
  + (SmartComp 2022) [[paper]](https://ieeexplore.ieee.org/abstract/document/9821057?casa_token=qAwx-o_ga4gAAAAA:MFqUbWBzqkfJ79QWU5yMvPmvFoG_T-pzAVdPEABcEiciymal0kAy5Ie1BCowtHyELCCNOtGbSyk)
+ DILoCC: An approach for Distributed Incremental Learning across the Computing Continuum 
  + (SmartComp 2022) [[paper]](https://ieeexplore.ieee.org/abstract/document/9556258?casa_token=uIv0gtgLWhEAAAAA:j592VhM8vYz0R__phIyBvnx5YQEtwrPAJiiZ16qu9nu2wu3jYr8xIfodzm5OpUn2NwwaPbYx8co)
+ Cross-FCL: Toward a Cross-edge Federated Continual Learning Framework in Mobile Edge Computing Systems 
  + (TMC 2022) [[paper]](https://ieeexplore.ieee.org/abstract/document/9960821?casa_token=1Ovr4510alIAAAAA:53TSbfLNHX8M5eh-2p65eXO2F7vbB4rAXIFAudCG92EAkPlFhecA5e0emL2r0gUBb5tvT9ePpoE)
+ Urban Traffic Forecasting using Federated and Continual Learning 
  + (CIoT 2023) [[paper]](https://ieeexplore.ieee.org/abstract/document/10084875?casa_token=hXBpivr18bYAAAAA:dqqI7ezE5h_anlHEB5VSDsWwZy3bzPKLNm9QRHFSY7LF7_ep9HbbCpTpw7GZ4dkED9WjjRbc0Js)
+ ICMFed: An Incremental and Cost-Efficient Mechanism of Federated Meta-Learning for Driver Distraction Detection 
  + (Mathematics 2023) [[paper]](https://www.mdpi.com/2227-7390/11/8/1867)
+ Personalized Federated Continual Learning for Task-incremental Biometrics
  + (IEEE Internet of Things Journal 2023) [[paper]](https://ieeexplore.ieee.org/abstract/document/10148063)
+ Continual adaptation of federated reservoirs in pervasive environments
  + (Neurocomputing 2023) [[paper]](https://www.sciencedirect.com/science/article/pii/S0925231223007610)
+ Continual Federated Learning For Network Anomaly Detection in 5G Open-RAN
  + (2023) [[paper]](http://jultika.oulu.fi/files/nbnfioulu-202307042837.pdf)
+ Age-Aware Data Selection and Aggregator Placement for Timely Federated Continual Learning in Mobile Edge Computing
  + (IEEE Transactions on Computers 2023) [[paper]](https://www.computer.org/csdl/journal/tc/5555/01/10324368/1SgbRaDs67C)

<a name="security" />

## Security Relevant
+ GFCL: A GRU-based Federated Continual Learning Framework against Data Poisoning Attacks in IoV 
  + (Arxiv 2022) [[paper]](https://arxiv.org/abs/2204.11010)
+ Towards a Defense against Backdoor Attacks in Continual Federated Learning 
  + (Arxiv 2022) [[paper]](https://arxiv.org/abs/2205.11736)
+ Federated Continual Learning with Differentially Private Data Sharing 
  + (NeurIPS 2022 Workshop) [[paper]](https://openreview.net/forum?id=b7vu9ukdpdL)
+ FL-IIDS: A novel federated learning-based incremental intrusion detection system
  + (Future Generation Computer Systems 2023) [[paper]](https://www.sciencedirect.com/science/article/pii/S0167739X23003503?casa_token=0sLsyxT8Vy0AAAAA:9dZIKeJlexvlOFK5aBQ8ym4se3xO6FdK9mP0COFennIxvcWs909vntH1rHAwa5_ePe0WS_Rl4aA)
+ POSTER: Advancing Federated Edge Computing with Continual Learning for Secure and Efficient Performance
  + (ANCS 2023) [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-41181-6_40)

<a name="other" />

## Other Topics
+ Federated Continual Learning to Detect Accounting Anomalies in Financial Auditing 
  + (Arxiv 2022) [[paper]](https://arxiv.org/abs/2210.15051)
+ Continual Learning of Dynamical Systems with Competitive Federated Reservoir Computing 
  + (Arxiv 2022) [[paper]](https://arxiv.org/abs/2206.13336)
+ Towards Lifelong Federated Learning in Autonomous Mobile Robots with Continuous Sim-to-Real Transfer 
  + (Arxiv 2022) [[paper]](https://arxiv.org/abs/2205.15496)
+ Performative Federated Learning: A Solution to Model-Dependent and Heterogeneous Distribution Shifts
  + (Arxiv 2023) [[paper]](https://arxiv.org/pdf/2305.05090.pdf)
+ Concept Drift Detection and Adaptation for Robotics and Mobile Devices in Federated and Continual Settings
  + (Some Workshop 2023) [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-62579-5_6)
+ Incremental learning and federated learning for heterogeneous medical image analysis
  + (Master Thesis 2023) [[paper]](https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0437226)
+ Continual adaptation of federated reservoirs in pervasive environments
  + (Neurocomputing 2023) [[paper]](https://www.sciencedirect.com/science/article/pii/S0925231223007610?casa_token=Lv139KCqmjoAAAAA:cOSOaY88N8aO9Cv2KP3FTIgIung7A3hyXh-VuwZLKOjwDZmOf2x721ITsQD-77n53K3BljBvM5k)
+ Peer-to-Peer Federated Continual Learning for Naturalistic Driving Action Recognition
  + (CVPR workshop 2023) [[paper]](https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Yuan_Peer-to-Peer_Federated_Continual_Learning_for_Naturalistic_Driving_Action_Recognition_CVPRW_2023_paper.pdf)







