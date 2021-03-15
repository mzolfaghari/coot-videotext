"""
Utility code for doing retrieval.
"""

from timeit import default_timer as timer
from typing import Callable, Dict, Tuple

import numpy as np
import torch as th


VALKEYS = ["r1", "r5", "r10", "r50", "medr", "meanr", "sum"]
VALHEADER = "Retriev | R@1   | R@5   | R@10  | R@50  | MeanR |  MedR |    Sum"
mean_score = {}
mean_score[0] = []
mean_score[1] = []
def retrieval_results_to_str(results: Dict[str, float], name: str) -> str:
    """
    Convert single dictionary of retrieval results to string.

    Args:
        results: Results dictionary.
        name: Type of retrieval to print.

    Returns:
        String results.
    """
    return ("{:7s} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:5.1f} | "
            "{:5.1f} | {:6.3f}").format(name, *[results[key] for key in VALKEYS])

def calculate_threshold_adjacency(similarity_measure, threshold):
    """
    from https://github.com/pik-copan/pyunicorn/blob/master/pyunicorn/climate/climate_network.py
    Extract the network's adjacency matrix by thresholding.
    The resulting network is a simple graph, i.e., self-loops and
    multiple links are not allowed.
    **Example** (Threshold zero should yield a fully connected network
    given the test similarity matrix):
    >>> net = ClimateNetwork.SmallTestNetwork()
    >>> net._calculate_threshold_adjacency(
    ...     similarity_measure=net.similarity_measure(), threshold=0.0)
    array([[0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0]], dtype=int8)
    :type similarity_measure: 2D Numpy array [index, index]
    :arg  similarity_measure: The similarity measure for all pairs of
                                nodes.
    :type threshold: number (float)
    :arg  threshold: The threshold of similarity measure, above which
                        two nodes are linked in the network.
    :rtype:  2D Numpy array (int8) [index, index]
    :return: the network's adjacency matrix.
    """

    N = similarity_measure.shape[0]
    A = np.zeros((N, N), dtype="int8")
    A[similarity_measure > threshold] = 1

    #  Set the diagonal of the adjacency matrix to zero -> no self loops
    #  allowed.
    A.flat[::N+1] = 0

    return A

def compute_retrieval2(data_collector_org, data_collector: Dict[str, th.Tensor], key1: str, key2: str, print_fn: Callable = print) -> (
        Tuple[Dict[str, float], Dict[str, float], float, str]):
    """
    Get embeddings from data collector given by keys, compute retrieval and return results.

    Args:
        data_collector: Collected validation data (output embeddings of the model).
        key1: Name of source embedding.
        key2: Name of target embedding.
        print_fn: Function to print the results with.

    Returns:
        Tuple of:
            Metrics for retrieval from key1 to key2.
            Metrics for retrieval from key2 to key1.
            Sum of R@1 metrics.
            Additional info string to print later (number of datapoints, time performance).
    """
    start_time = timer()
    emb1 = data_collector[key1]
    emb2 = data_collector[key2]
    if isinstance(emb1, th.Tensor):
        emb1 = emb1.numpy()
    if isinstance(emb2, th.Tensor):
        emb2 = emb2.numpy()

    d = np.dot(emb1, emb2.T)
    # import pdb; pdb.set_trace()
    # print("=="*20)
    ## ===========
    # diag = np.eye(emb1.shape[0])
    # mask = th.from_numpy((diag))
    # mask_neg = 1 - mask
    # print(d[mask.type(th.bool)].mean(), d[mask_neg.type(th.bool)].mean())
    # import pickle
    # neg_name = d[mask_neg.type(th.bool)]
    # pos_name = d[mask.type(th.bool)]
    # with open('name_pos1.pickle', 'wb') as handle: pickle.dump(pos_name, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('name_neg1.pickle', 'wb') as handle: pickle.dump(neg_name, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # emb1_org = data_collector_org["sent_feat"][:50]
    # sentences_org = data_collector_org["sentences"][:64]
    # d_org = np.dot(emb1_org, emb1_org.T)
    # a_org = calculate_threshold_adjacency(d_org, 120)

    # i = 0
    # with open('somefile6.txt', 'a') as the_file:
    #     for item1 in range(d_org.shape[0]):
    #         for item2 in range(item1+1, d_org.shape[0]):
    #             if a_org[item1, item2] == 1:
    #                 # import pdb; pdb.set_trace()
    #                 # print(item1, len(sentences_org[0]))
    #                 # print(sentences_org[0][item1][0].replace("[CLS]","").replace("[SEP]",""))
    #                 the_file.write('{} {} {}\n'.format(i, item1, item2))
    #                 i += 1
    # import pdb; pdb.set_trace()

    num_points = len(d)
    res1, _, _ = compute_retrieval_cosine(d)
    res2, _, _ = compute_retrieval_cosine(d.T)
    sum_at_1 = (res1["r1"] + res2["r1"]) / 2
    # sum_at_1 = (res1["r1"] + res2["r1"] + res1["r5"] + res2["r5"] + res1["r10"] + res2["r10"]) / 6

    print_fn(retrieval_results_to_str(res1, key1[:3]))
    print_fn(retrieval_results_to_str(res2, key2[:3]))
    result_str = f"{key1[:3]}{key2[:3]} ({num_points}) in {timer() - start_time:.3f}s, "
    return res1, res2, sum_at_1, result_str

def compute_retrieval(data_collector: Dict[str, th.Tensor], key1: str, key2: str, print_fn: Callable = print) -> (
        Tuple[Dict[str, float], Dict[str, float], float, str]):
    """
    Get embeddings from data collector given by keys, compute retrieval and return results.

    Args:
        data_collector: Collected validation data (output embeddings of the model).
        key1: Name of source embedding.
        key2: Name of target embedding.
        print_fn: Function to print the results with.

    Returns:
        Tuple of:
            Metrics for retrieval from key1 to key2.
            Metrics for retrieval from key2 to key1.
            Sum of R@1 metrics.
            Additional info string to print later (number of datapoints, time performance).
    """
    start_time = timer()
    emb1 = data_collector[key1]
    emb2 = data_collector[key2]
    if isinstance(emb1, th.Tensor):
        emb1 = emb1.numpy()
    if isinstance(emb2, th.Tensor):
        emb2 = emb2.numpy()

    d = np.dot(emb1, emb2.T)

    #=====
    # # print("=="*20)
    # diag = np.eye(emb1.shape[0])
    # mask = th.from_numpy((diag))
    # mask_neg = 1 - mask
    
    # print(d[mask.type(th.bool)].mean(), d[mask_neg.type(th.bool)].mean())
    # mean_score[0].append(d[mask.type(th.bool)].mean())
    # mean_score[1].append(d[mask_neg.type(th.bool)].mean())
    # print(mean_score)
    # import pdb; pdb.set_trace()

    num_points = len(d)
    res1, _, _ = compute_retrieval_cosine(d)
    res2, _, _ = compute_retrieval_cosine(d.T)
    sum_at_1 = (res1["r1"] + res2["r1"]) / 2
    # sum_at_1 = (res1["r1"] + res2["r1"] + res1["r5"] + res2["r5"] + res1["r10"] + res2["r10"]) / 6

    print_fn(retrieval_results_to_str(res1, key1[:3]))
    print_fn(retrieval_results_to_str(res2, key2[:3]))
    result_str = f"{key1[:3]}{key2[:3]} ({num_points}) in {timer() - start_time:.3f}s, "
    return res1, res2, sum_at_1, result_str


def compute_failures(data_collector: Dict[str, th.Tensor], key1: str, key2: str, sentences, print_fn: Callable = print) -> (
        Tuple[Dict[str, float], Dict[str, float], float, str]):
    """
    Get embeddings from data collector given by keys, compute retrieval and return results.

    Args:
        data_collector: Collected validation data (output embeddings of the model).
        key1: Name of source embedding.
        key2: Name of target embedding.
        print_fn: Function to print the results with.

    Returns:
        Tuple of:
            Metrics for retrieval from key1 to key2.
            Metrics for retrieval from key2 to key1.
            Sum of R@1 metrics.
            Additional info string to print later (number of datapoints, time performance).
    """
    start_time = timer()
    emb1 = data_collector[key1]
    emb2 = data_collector[key2]
    if isinstance(emb1, th.Tensor):
        emb1 = emb1.numpy()
    if isinstance(emb2, th.Tensor):
        emb2 = emb2.numpy()

    d = np.dot(emb1, emb2.T)
    num_points = len(d)
    res1, _, ranks1 = compute_retrieval_cosine(d)
    res2, _, ranks2 = compute_retrieval_cosine(d.T)
    x1 = []
    for b in sentences: 
        for xs in b:
            x1.extend(xs)

    id_fails = np.where(ranks1>0)[0]
    id_ok = np.where(ranks1==0)[0]
    
    file_object1 = open('text_all.txt', 'w')
    file_object2 = open('text_fails.txt', 'w')
    file_object3 = open('text_ok.txt', 'w')

    cnt_words = {}
    for idf in range(len(ranks1)):
        # for s_x1 in x1[idf]:
            sf = x1[idf].replace('[SEP]','').replace('[CLS]','')
            file_object1.write("%s\n" % sf)
            # sf_words = sf.split()
            # for sfw in sf_words:
            #     if cnt_words.get(sfw) is None:
            #         cnt_words[sfw] = 1
            #     else:
            #         cnt_words[sfw] += 1
    file_object1.close()

    cnt_words_fail = {}
    for idf in id_fails:
        # for s_x1 in x1[idf]:
            sf = x1[idf].replace('[SEP]',' ').replace('[CLS]','')
            file_object2.write("%s\n" % sf)
            # sf_words = sf.split()
            # for sfw in sf_words:
            #     if cnt_words_fail.get(sfw) is None:
            #         cnt_words_fail[sfw] = 1
            #     else:
            #         cnt_words_fail[sfw] += 1

    file_object2.close()

    for idf in id_ok:
        # for s_x1 in x1[idf]:
            sf = x1[idf].replace('[SEP]',' ').replace('[CLS]','')
            file_object3.write("%s\n" % sf)
            # sf_words = sf.split()
            # for sfw in sf_words:
            #     if cnt_words_fail.get(sfw) is None:
            #         cnt_words_fail[sfw] = 1
            #     else:
            #         cnt_words_fail[sfw] += 1
    file_object3.close()

    sorted_w = {k: v for k, v in sorted(cnt_words.items(), key=lambda item:item[1])}
    sorted_w_fails = {k: v for k, v in sorted(cnt_words_fail.items(), key=lambda item:item[1])}

    import pdb; pdb.set_trace()
    sum_at_1 = (res1["r1"] + res2["r1"]) / 2
    print_fn(retrieval_results_to_str(res1, key1[:3]))
    print_fn(retrieval_results_to_str(res2, key2[:3]))
    result_str = f"{key1[:3]}{key2[:3]} ({num_points}) in {timer() - start_time:.3f}s, "
    return res1, res2, sum_at_1, result_str

def compute_retrieval_cosine(dot_product: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Args:
        dot_product: Result of computing cosine similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    len_dot_product = len(dot_product)
    ranks = np.empty(len_dot_product)
    top1 = np.empty(len_dot_product)
    # loop source embedding indices
    for index in range(len_dot_product):
        # get order of similarities to target embeddings
        inds = np.argsort(dot_product[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]
    # compute retrieval metrics
    r1 = len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r50}
    return report_dict, top1, ranks
