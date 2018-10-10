# Paper in Natural Laguage Processing


자연어처리를 공부하며 정리한 내용들을 정리하고 있습니다. 계속해서 update될 예정이며 향후 구현도 추가될 예정입니다. 논문들은 Task 별로 분류해두었으며 github에 Task들의 간단한 소개와 각 논문들에 대해 간단하게 설명해두었습니다. 논문에 대한 자세한 내용과 리뷰는 링크를 통해 블로그 글을 참고하시면 됩니다!  

</br>
## Table of Contents


* [Word Representation](#word-representation)
  * [Efficient Estimation of Word Representations in Vector Space](https://reniew.github.io/21/)
  * [Distributed Representations of Words and Phrases and their Compositionality](https://reniew.github.io/22/)
  * [Global Vectors for Word Representation](https://reniew.github.io/23/)
* [Text Classification](#text-classification)
  * [Convolutional Neural Network Sentence Classification](https://reniew.github.io/26/)
  * [A Convolutional Neural Network for Modelling Sentences](https://reniew.github.io/27/)
  * [A Sensitivity Analysis of Convolutional Neural Networks for Sentence Classification](https://reniew.github.io/28/)
  * [Character-level Convolutional Network for Text Classification](https://reniew.github.io/29/)
* [Machine Translation](#machine-translation)
  * [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://reniew.github.io/31/)
  * [Sequence to Sequence Learning with Neural Network](https://reniew.github.io/35/)
  * [Neural Machine Translation by Jointly Learning to Align and Translate](https://reniew.github.io/37/)
  * [Convolutional Seqeunce to Sequence Learning](https://reniew.github.io/44/)
  * [Attention Is All You Need](https://reniew.github.io/43/)
* [Question Answering](#question-answering)
  * [Memory Network](https://reniew.github.io/45/)
  * [End to End Memory Network](https://reniew.github.io/46/)

---
</br>

# Word Representation

Word Representation은 단어 표현 task로 흔히 word embedding이라고 불리는 task입니다. 자연어 처리에서 가장 기본적인 내용으로 단어를 dense한 vector로 표현하기 위한 모델을 학습합니다. 대표적인 방법으로 word2vec과 glove, fasttext가 있으며, 그 외에도 다양한 모델들이 있습니다. 자연어 처리의 모든 task에서 처음에 사용되는 기법들로 다른 task보다 먼저 숙지할 것을 추천합니다. :clap:

정리된 Word Representation 논문들의 목록은 다음과 같습니다.

* Efficient Estimation of Word Representations in Vector Space
* Distributed Representations of Words and Phrases and their Compositionality
* Global Vectors for Word Representation


## Efficient Estimation of Word Representations in Vector Space

Paper|Blog|author
--|--|--
[Paper](https://arxiv.org/pdf/1301.3781.pdf) | [Blog](https://reniew.github.io/21/)|T Mikolov, K Chen, G Corrado, J Dean

Word2Vec으로 유명한 모델입니다. 총 2개의 논문을 통해 word2vec을 소개를 하고 있습니다. 그 중 첫 번째 논문으로 논문에서는 CBOW 모델과 Skip-Gram 모델을 소개하고 있습니다. 기존의 one-hot encoding 방식에서 단어들간의 similarity를 잘 잡아내도록 학습된 dense vector로 표현하는 방식으로 성능이 좋아 많은 모델에서 embedding 방식으로 선택하는 모델입니다.   


## Distributed Representations of Words and Phrases and their Compositionality

Paper|Blog|author
--|--|--
[Paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) | [Blog](https://reniew.github.io/22/)|T Mikolov, I Sutskever, K Chen, GS Corrado

Word2Vec을 소개한 두 번째 논문으로, 기존 모델에서 학습시 연산량이 많아서 시간이 많이 소모되는 단점을 보완하기 위해 몇 가지 기법들에 대해서 소개하고 있습니다.

## Global Vectors for Word Representation

Paper|Blog|author
--|--|--
[Paper](https://nlp.stanford.edu/pubs/glove.pdf)|[Blog](https://reniew.github.io/23/)|J Pennington, R Socher, C Manning

Word2Vec 이후에 Stanford에서 만든 word embedding 기법으로 Glove로 불리는 모델입니다. 기존의 Word2Vec 모델이 Shallow Predict 모델이라 할 수 있다면, 해당 모델은 좀 더 Statistical한 정보를 담아서 학습할 수 있도록 만든 모델입니다. NLP 모델들이 Pre-trained 된 word vector를 사용한는 경우가 많은데 보통 Glove로 pre-trained 된 vector를 사용합니다. Word2vec과 학습 과정의 차이를 이해하며 알아보는 것을 추천합니다.

</br>
---
</br>

# Text Classification

Text Classification은 자연어처리 task들 중 가장 기본적인 task로 상대적으로 다른 task에 비해 직관적이고 간단한 편입니다. 따라서 만약 자연어 처리를 처음 접하는 사람이라면 Word Representation을 공부한 뒤에 다른 task들 보다 Text Classifcation을 보는 것을 추천합니다. Classification은 주로 Sentiment Analysis(감정 분석)이 주를 이루고 있으며 이 task는 해당 글의 감정이 긍정 인지 부정인지 판단하는 문제입니다. 영화 리뷰 데이터를 데이터로 모델을 학습하는 경우가 많습니다. :kissing_smiling_eyes:

정리된 Text Classification의 논문들의 목록은 다음과 같습니다.

* Convolutional Neural Network Sentence Classification
* A Convolutional Neural Network for Modelling Sentences
* A Sensitivity Analysis of Convolutional Neural Networks for Sentence Classification
* Character-level Convolutional Network for Text Classification


## Convolutional Neural Network Sentence Classification

Paper|Blog|Author
--|--|--
[Link](http://www.aclweb.org/anthology/D14-1181)|[Link](https://reniew.github.io/26/)|Y Kim

Convolutional Nerural Network 를 사용해서 Text Classification을 하는 모델로 모델 자체가 매우 간단한데도 불구하고 높은 성능으로 다양하게 활용되는 논문입니다.

## A Convolutional Neural Network for Modelling Sentences

Paper|Blog|Author
--|--|--
[Link](https://arxiv.org/abs/1404.2188)|[Link](https://reniew.github.io/27/)|N Kalchbrenner, E Grefenstette, P Blunsom

위의 논문과 비슷하게 Convolutional Neural Network를 통해 text classification을 하는 모델입니다. 여기서는 추가적인 기법으로 k-max pooling 기법을 사용해서 성능을 올리고 있습니다.

## A Sensitivity Analysis of Convolutional Neural Networks for Sentence Classification

Paper|Blog|Author
--|--|--
[Link](https://arxiv.org/abs/1510.03820)|[Link](https://reniew.github.io/28/)|Y Zhang, B Wallace

Convolutional Neural Network를 통해 Text Classification을 하는 많은 논문들이 나왔는데 그런 모델에서 중요한 것이 Kernel의 개수, Kernel의 크기 등 Hyper Parameter를 설정함에 따라 성능이 크게 달라진 다는 점입니다. 해당 논문에서는 각 Hyper Parameter를 조정함에 따라서 여러 데이터셋에 실험해 성능들을 비교하며 어떤 값을 Hyper Parameter 값으로 설정해야 하는지에 대한 Guide를 해주고 있습니다.

## Character-level Convolutional Network for Text Classification

Paper|Blog|Author
--|--|--
[Link](https://arxiv.org/pdf/1509.01626.pdf)|[Link](https://reniew.github.io/29/)|X Zhang, J Zhao, Y LeCun

기존의 Convolutional Neural Network를 통한 Text Classification 모델들은 주로 input의 단위를 하나의 단어(word)로 설정했습니다. 이 논문에서는 input의 기본 단위를 문자(character)로 설정하고 좋은 성능을 보여줬습니다. NLP 모델의 경우 최소 단위에 대한 연구가 계속해서 이뤄지는 만큼 Input을 Word로 하는 것과 Character로 하는 것에 대해 차이를 숙지하는 것을 추천합니다.

</br>

---

</br>

# Machine Translation

Machine Translation이란 기계 번역 분야로 NLP의 핵심 Task 중 하나 입니다. 기본적으로 input 과 output 모두 Sequence로 구성되어 있고, 기계 번역을 위한 모델들은 다른 sequence를 다루는 task에 적용할 수 있어 범용적으로 대부분 범용적으로 활용 할 수 있습니다. 아래 논문의 순서의 흐름에 따라 보시는 것을 추천합니다. :+1:  

Machine Translation 분야의 논문들의 목록은 다음과 같습니다.

* Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation(https://reniew.github.io/31/)
* Sequence to Sequence Learning with Neural Network](https://reniew.github.io/35/)
* Neural Machine Translation by Jointly Learning to Align and Translate](https://reniew.github.io/37/)
* Convolutional Seqeunce to Sequence Learning](https://reniew.github.io/44/)
* Attention Is All You Need](https://reniew.github.io/43/)

## Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

Paper|Blog|Author
--|--|--
[Link](https://arxiv.org/abs/1406.1078)|[Link](https://reniew.github.io/31/)|K Cho, B Van Merriënboer, C Gulcehre

뉴욕 대학의 조경현 교수님께서 만드신 모델로 Sequence to Sequence로 더욱 잘 알려진 모델입니다. Neural Machine Translation의 시작이라고도 불릴만큼 의미있는 논문입니다. 기본적으로 Encoder-Decoder 구조를 가지고 있으며, 이후 sequence를 다루는 많은 모델에서 사용하고 있는 구조입니다.

## Sequence to Sequence Learning with Neural Network

Paper|Blog|Author
--|--|--
[Link](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)|[Link](https://reniew.github.io/35/)|I Sutskever, O Vinyals, QV Le

Google에서 나온 Sequence to sequence 모델입니다. 위의 모델과 비슷한 시기에 나왔지만 조경현 교수님의 모델이 조금 더 먼저 나와 seq2seq의 첫 모델은 위의 모델으로 알려져 있습니다. 위 모델과 거의 유사한 구조를 가지고 있지만, Encoder에서 input을 넣는 방향을 반대로 넣었다는 점이 위의 모델과 차이점입니다.

## Neural Machine Translation by Jointly Learning to Align and Translate

Paper|Blog|Author
--|--|--
[Link](https://arxiv.org/pdf/1409.0473.pdf)|[Link](https://reniew.github.io/37/)|D Bahdanau, K Cho, Y Bengio

Attention의 개념을 처음으로 도입한 논문입니다. 기존의 Sequence to sequence 모델이 sequence의 align 정보를 담지 못한다는 문제점을 해결하기 위해 attention 기법을 사용했습니다. 이후 다양한 attention 기법들이 나왔는데 해당 논문에서 사용한 기법은 현재는 Additive attention 기법으로 분류하고 있습니다.

## Convolutional Seqeunce to Sequence Learning

Paper|Blog|Author
--|--|--
[Link](https://arxiv.org/pdf/1705.03122.pdf)|[Link](https://reniew.github.io/44/)|J Gehring, M Auli, D Grangier, D Yarats

Facebook에서 만든 모델입니다. 기존의 Sequence to sequence 모델은 대부분 Recurrent Neural Network를 기반으로 만들어졌는데, 해당 논문에서는 Convolutional Neural Network를 사용해서 Sequence to sequence 모델을 만들었습니다. CNN을 사용함으로써 병렬화가 쉽다는 장점이 있어 학습 속도가 매우 빠릅니다. 그리고 추가적으로 attention 기법을 사용했습니다.

## Attention is All You Need

Paper|Blog|Author
--|--|--
[Link](https://arxiv.org/pdf/1706.03762)|[Link](https://reniew.github.io/43/)|A Vaswani, N Shazeer, N Parmar

Google 에서 만든 모델로 기존의 모델들은 RNN 혹은 CNN을 사용한 모델을 기반으로 부가적인 기법으로 Attention을 사용했는데 해당 모델에서는 Attention 만을 사용해서 모델을 만들었습니다. 기본적인 Feed Forward Network와 Attention 기법 그리고 추가적으로 Residual connection을 사용했습니다. 그리고 해당 모델에서 두 가지의 attention 기법이 소개 되었는데, 하나는 Scaled dot-product attention이고 하나는 Multi-head attention입니다. 다양한 Attention 기법들이 있는데, attention 만을 사용해서 높은 성능을 보인만큼 의미있는 논문이라 생각됩니다.

</br>
---
</br>

# Question Answering

Question Answering이란 주어진 질문에 대해서 답을 할 수 있는 모델을 만드는 task입니다. 기본적으로 input인 질문과 output인 답이 모두 sequence 형태라 Machine Translation의 모델과 혼용해서 사용하기도 합니다. Qeustion Answering을 하는 대표적인 competition인 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)를 통해 많은 모델들이 나오고 있습니다.

Qeustion Answering을 위한 논문들의 목록은 다음과 같습니다.

* Memory Network
* End to End Memory Network


## Memory Network

Paper|Blog|Author
--|--|--
[Link](https://arxiv.org/pdf/1410.3916.pdf)|[Link](https://reniew.github.io/45/)|J Weston, S Chopra, A Bordes

memory의 개념을 도입해서 Long-term 에 대한 정보를 반영하기 위한 모델입니다. Memory와 4개의 Components를 통해서 모델을 구성했으며, 모든 학습이 supervised한 성격이 강하기 때문에 여러 task에 적용하기 어렵다는 단점이 있습니다.

## End to End Memory Networks

Paper|Blog|Author
--|--|--
[Link](https://arxiv.org/pdf/1503.08895.pdf)|[Link](https://reniew.github.io/46/)|S Sukhbaatar, J Weston, R Fergus

기존의 Memory Network는 모든 과정이 supervise해서 제약이 있다는 문제점을 End to End 한 모델을 통해 해결하고 있습니다. 모델 자체가 end-to-end를 목적으로 만들어 졌기 때문에 기존의 모델보다 조금 더 간단한 형태입니다. 그리고 기존의 모델처럼 제약을 받지 않으므로 범용적으로 사용할 수 있다는 점이 해당 모델의 장점입니다.

</br>

---

</br>


혹여나 오타나 review에서의 잘못된 정보가 있다면 issue 혹은 contact point를 통해서 알려주세요! 향후 계속해서 업데이트 할 예정으로 star를 해주시면 감사합니다! :pray:
