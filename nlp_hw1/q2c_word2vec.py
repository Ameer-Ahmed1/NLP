#!/usr/bin/env python
import random

import numpy as np
import math

from helpers.utils import normalize_rows, sigmoid, get_negative_samples
from q2a_softmax import softmax
from q2b_gradcheck import gradcheck_naive


def naive_softmax_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    center_word_vec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outside_word_idx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outside_vectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    grad_center_vec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    grad_outside_vecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    center_embedding = center_word_vec
    target_embedding = outside_vectors[outside_word_idx]
    scores = np.dot(outside_vectors, center_embedding)
    probabilities = softmax(scores)
    loss = -1 * math.log(probabilities[outside_word_idx]) 

    weighted_sum = np.sum(probabilities[:, np.newaxis] * outside_vectors, axis=0)
    grad_center_vec = -1 * (target_embedding - weighted_sum)

    grad_outside_vecs = probabilities[:, np.newaxis] * center_embedding
    grad_outside_vecs[outside_word_idx] = (probabilities[outside_word_idx] - 1) * center_embedding

    return loss, grad_center_vec, grad_outside_vecs


def neg_sampling_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset,
        K=10
):
    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)
    indices = [outside_word_idx] + neg_sample_word_indices

    center_vec = center_word_vec
    context_vec = outside_vectors[outside_word_idx]
    negative_vectors = outside_vectors[neg_sample_word_indices]

    loss = -math.log(sigmoid(np.dot(context_vec, center_vec))) - np.sum(np.log(sigmoid(-np.dot(negative_vectors, center_vec))))

    grad_center_vec = -(1 - sigmoid(np.dot(context_vec, center_vec))) * context_vec + np.sum((negative_vectors.T * (1 - sigmoid(-np.dot(negative_vectors, center_vec)))).T, axis=0)

    grad_outside_vecs = np.zeros(outside_vectors.shape)
    for k in neg_sample_word_indices:
        grad_outside_vecs[k] += (1 - sigmoid(-np.dot(outside_vectors[k], center_vec))) * center_vec
    grad_outside_vecs[outside_word_idx] = -(1 - sigmoid(np.dot(context_vec, center_vec))) * center_vec

    return loss, grad_center_vec, grad_outside_vecs

def skipgram(current_center_word, outside_words, word2ind,
             center_word_vectors, outside_vectors, dataset,
             word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    loss = 0.0
    grad_center_vecs = np.zeros(center_word_vectors.shape)
    grad_outside_vectors = np.zeros(outside_vectors.shape)

    center_word_index = word2ind[current_center_word]
    center_word_vec = center_word_vectors[center_word_index]

    for outside_word in outside_words:
        outside_word_index = word2ind[outside_word]
        loss_value, grad_center, grad_outside = word2vec_loss_and_gradient(center_word_vec, outside_word_index, outside_vectors, dataset)
        loss += loss_value
        grad_center_vecs[center_word_index, :] += grad_center
        grad_outside_vectors += grad_outside

    return loss, grad_center_vecs, grad_outside_vectors

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################


def word2vec_sgd_wrapper(word2vec_model, word2ind, word_vectors, dataset,
                         window_size, word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(word_vectors.shape)
    N = word_vectors.shape[0]
    center_word_vectors = word_vectors[:int(N / 2), :]
    outside_vectors = word_vectors[int(N / 2):, :]
    for i in range(batchsize):
        window_size1 = random.randint(1, window_size)
        center_word, context = dataset.getRandomContext(window_size1)

        c, gin, gout = word2vec_model(
            center_word, context, word2ind, center_word_vectors,
            outside_vectors, dataset, word2vec_loss_and_gradient
        )
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad


def test_word2vec_basic():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()

    def dummy_sample_token_idx():
        return random.randint(0, 4)

    def get_random_context(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
            [tokens[random.randint(0, 4)] for _ in range(2*C)]
    dataset.sampleTokenIdx = dummy_sample_token_idx
    dataset.getRandomContext = get_random_context

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalize_rows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print("==== Gradient check for skip-gram with naive_softmax_loss_and_gradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naive_softmax_loss_and_gradient),
        dummy_vectors, "naive_softmax_loss_and_gradient Gradient")

    print("==== Gradient check for skip-gram with neg_sampling_loss_and_gradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, neg_sampling_loss_and_gradient),
                    dummy_vectors, "neg_sampling_loss_and_gradient Gradient")

    print("\n=== Results ===")
    print("Skip-Gram with naive_softmax_loss_and_gradient")

    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", ["a", "b", "e", "d", "b", "c"], dummy_tokens,
                      dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
        )
    )

    print("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print("Skip-Gram with neg_sampling_loss_and_gradient")
    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", ["a", "b"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :],
                  dataset, neg_sampling_loss_and_gradient)
        )
    )
    print("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)


if __name__ == "__main__":
    test_word2vec_basic()


