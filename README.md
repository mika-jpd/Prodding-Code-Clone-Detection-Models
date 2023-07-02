# What do code-based Transformer models learn?

*I will be reorganizing the folder structure, refactoring the code and adding comments so that the project is easier to understand once I have the time !*

## Project Organization
This is the final project for the course Comp 599: Natural Language Understanding with Deep Learning (more info here [link](https://mcgill-nlp.github.io/teaching/comp599-ling782-484-f22/)) which I completed with Adam Weiss and Hector Leos. We received the maximum grade of 85% (A) for the project.

## Abstract
With the advent of state-of-the-art large-scale pre-trained Transformer-based models in various domains of Natural Language Processing, recent work has focused on understanding what about natural language these models learn. While some studies show they learn grammatical structures, others cast doubt over such claims. In addition to natural language tasks, transformer models, such as CodeBERT have also recently been used in many tasks involving code. However, little work has been done to understand whether Transformers are able to understand key structures related to code. In this paper, we provide novel evidence that state-of-the-art Code-Clone detection models are largely invariant to random word-order permutations (ie. they assign the same labels to code pairs that have been permuted and those which haven’t). We provide preliminary empirical evaluation of this phenomenon. Furthermore, we also find evidence that Transformers are capable of capturing important syntactic structures, as previously shown in ML models. Syntax structures such as corefence are captured to a great extent. We discuss the implications of these puzzling findings in the effort of understanding what these models are learning at large.

Our work falls within a general push to better understand how deep learning models function so they can be interpretable. 
