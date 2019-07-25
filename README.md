# DBank: Predictive Behavioral Analysis of Recent Android Banking Trojans (IEEE Transactions on Dependable and Secure Computing, 2019)

Authors: Chongyang Bai*, Qian Han*, Ghita Mezzour, Fabio Pierazzi, V.S. Subrahmanian (* equal contribution, authors listed in alphabetic order)

Link to the paper: https://ieeexplore.ieee.org/document/8684321

Introduction

DBank is a system to predict whether a given Android APK is a banking trojan or not using a novel dataset of Android banking trojans (ABTs), other Android malware, and goodware.

We introduce the novel concept of a Triadic Suspicion Graph (TSG for short) which contains three kinds of nodes: goodware, banking trojans, and API packages. We develop a novel feature space based on two classes of scores derived from TSGs: suspicion scores (SUS) and suspicion ranks (SR)—the latter yields a family of features that generalize PageRank. While TSG features (based on SUS/SR scores) provide very high predictive accuracy on their own in predicting recent ABTs.

Moreover, we have already reported two unlabeled APKs from VirusTotal (which DBank has detected as ABTs) to the Google Android Security Team—in one case, we discovered it before any of the 63 anti-virus products on VirusTotal did, and in the other case, we beat 62 of 63 anti-viruses on VirusTotal.

We also show that our novel TSG features have some interesting defensive properties as they are robust to knowledge of the training set by an adversary: even if the adversary uses 90% of our training set and uses the exact TSG features that we use, it is difficult for him to infer DBank’s predictions on APKs. 

We additionally identify the features that best separate and characterize ABTs from goodware as well as from other Android malware. Finally, we develop a detailed data-driven analysis of five major recent ABT families: FakeToken, Svpeng, Asacub, BankBot, and Marcher, and identify the features that best separate them from goodware and other malware.

# TRIADIC SUSPICION GRAPH (TSG) FEATURES

