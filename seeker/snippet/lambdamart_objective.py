#date: 2021-11-16T17:02:16Z
#url: https://api.github.com/gists/c25c228dab50d7c40045e3bce8472596
#owner: https://api.github.com/users/klymya

import numpy as np
from numba import njit


# const score_t kMinScore = -std::numeric_limits<score_t>::infinity();
# typedef float score_t;
# typedef int32_t data_size_t;
kMinScore = -np.inf
kMaxPosition = 10000


class DCGCalculator:
    def __init__(self, label_gain=None):
        if label_gain:
            self._label_gain = label_gain
        else:
            self._label_gain = self.default_label_gain()

        self._discount = np.zeros(kMaxPosition)
        for i in range(0, kMaxPosition):
            self._discount[i] = 1.0 / np.log2(2.0 + i)

    @property
    def label_gain(self):
        return self._label_gain

    def get_discount(self, k):
        return self._discount[k]

    def calc_max_dcg_at_k(self, k, label):
        ret = 0.0
        num_data = len(label)
        # counts for all labels
        label_cnt = np.zeros_like(self._label_gain)
        for i in range(0, num_data):
            label_cnt[int(label[i])] += 1

        top_label = len(self._label_gain) - 1

        if k > num_data:
            k = num_data

        #  start from top label, and accumulate DCG
        for j in range(0, k):
            while top_label > 0 and label_cnt[top_label] <= 0:
                top_label -= 1

            if (top_label < 0):
                break

            ret += self._discount[j] * self._label_gain[top_label]
            label_cnt[top_label] -= 1

        return ret

    @staticmethod
    def default_label_gain():
        # label_gain = 2^i - 1, may overflow, so we use 31 here
        max_label = 31
        label_gain = []
        label_gain.append(0.0)
        for i in range(1, max_label):
            label_gain.append((1 << i) - 1)

        return label_gain


class LambdarankNDCG:
    def __init__(
        self, label, group, weights=None, sigmoid=1.0, lambdarank_norm=True, lambdarank_truncation_level=30,
        label_gain=None, doc_weights=None
    ):
        # ! \brief Number of bins in simoid table */
        self._sigmoid_bins = 1024 * 1024
        # ! \brief Minimal input of sigmoid table */
        self._min_sigmoid_input = -50
        # ! \brief Maximal input of sigmoid table */
        self._max_sigmoid_input = 50

        self._weights = weights

        if doc_weights is None:
            doc_weights = [1] * len(label)

        self._doc_weights = doc_weights

        self._truncation_level = lambdarank_truncation_level
        self._norm = lambdarank_norm
        self._sigmoid = sigmoid

        self._sigmoid_table = None
        self._sigmoid_table_idx_factor = None
        self._inverse_max_dcgs = None

        self._dcg_calculator = DCGCalculator(label_gain=label_gain)
        self._label_gain = self._dcg_calculator.label_gain

        num_queries, query_boundaries = self.get_query_num_and_boundaries(group)
        self._init_dcg(num_queries, label, query_boundaries)
        self._construct_sigmoid_table()

    def lambdarank_ndcg(self, y_true, y_pred, group):
        num_queries, query_boundaries = self.get_query_num_and_boundaries(group)

        grad, hess = self._get_gradients(y_true, y_pred, num_queries, query_boundaries)
        return grad, hess

    def _get_gradients(self, label, score, num_queries, query_boundaries):
        gradients = []
        hessians = []
        for i in range(0, num_queries):
            start = query_boundaries[i]
            cnt = query_boundaries[i + 1] - query_boundaries[i]
            tmp_grad, tmp_hess = self._get_gradients_for_one_query_wrapper(  # self._get_gradients_for_one_query(
                i, cnt, label[start: start + cnt], score[start: start + cnt], self._doc_weights[start: start + cnt])

            if self._weights:
                for j in range(cnt):
                    gradients[j] = gradients[start + j] * self._weights[start + j]
                    hessians[j] = hessians[start + j] * self._weights[start + j]

            gradients += tmp_grad.tolist()
            hessians += tmp_hess.tolist()

        return gradients, hessians

    def _get_gradients_for_one_query_wrapper(self, query_id, cnt, label, score, doc_weights):
        return _get_gradients_for_one_query(
            query_id,
            cnt,
            label,
            score,
            doc_weights,
            self._inverse_max_dcgs,
            self._truncation_level,
            np.array(self._label_gain, dtype=np.int64),
            self._norm,
            self._sigmoid,
            self._min_sigmoid_input,
            self._max_sigmoid_input,
            self._sigmoid_table,
            self._sigmoid_bins,
            self._sigmoid_table_idx_factor,
            self._dcg_calculator._discount

        )

    def _get_sigmoid(self, score):
        if (score <= self._min_sigmoid_input):
            # too small, use lower bound
            return self._sigmoid_table[0]
        elif score >= self._max_sigmoid_input:
            # too large, use upper bound
            return self._sigmoid_table[self._sigmoid_bins - 1]
        else:
            return self._sigmoid_table[int((score - self._min_sigmoid_input) * self._sigmoid_table_idx_factor)]

    def _construct_sigmoid_table(self):
        # get boundary
        self._min_sigmoid_input = self._min_sigmoid_input / self._sigmoid / 2
        self._max_sigmoid_input = -self._min_sigmoid_input
        self._sigmoid_table = np.zeros(self._sigmoid_bins)
        # get score to bin factor
        self._sigmoid_table_idx_factor = self._sigmoid_bins / (self._max_sigmoid_input - self._min_sigmoid_input)
        # cache
        for i in range(self._sigmoid_bins):
            score = i / self._sigmoid_table_idx_factor + self._min_sigmoid_input
            self._sigmoid_table[i] = 1.0 / (1.0 + np.exp(score * self._sigmoid))

    def _init_dcg(self, num_queries, label, query_boundaries):
        if self._inverse_max_dcgs and num_queries == self._inverse_max_dcgs.shape[0]:
            return

        self._inverse_max_dcgs = np.zeros(num_queries)

        for i in range(0, num_queries):
            self._inverse_max_dcgs[i] = self._dcg_calculator.calc_max_dcg_at_k(
                self._truncation_level, label[query_boundaries[i]: query_boundaries[i + 1]])

            if self._inverse_max_dcgs[i] > 0.0:
                self._inverse_max_dcgs[i] = 1.0 / self._inverse_max_dcgs[i]

    @staticmethod
    def get_query_num_and_boundaries(group):
        num_queries = len(group)
        query_boundaries = [0]
        for i in group:
            query_boundaries.append(query_boundaries[-1] + i)

        return num_queries, query_boundaries


@njit
def _get_sigmoid(
    score, _min_sigmoid_input, _max_sigmoid_input, _sigmoid_table, _sigmoid_bins, _sigmoid_table_idx_factor
):
    if (score <= _min_sigmoid_input):
        # too small, use lower bound
        return _sigmoid_table[0]
    elif score >= _max_sigmoid_input:
        # too large, use upper bound
        return _sigmoid_table[_sigmoid_bins - 1]
    else:
        return _sigmoid_table[int((score - _min_sigmoid_input) * _sigmoid_table_idx_factor)]


@njit
def _get_gradients_for_one_query(
    query_id, cnt, label, score, doc_weights, _inverse_max_dcgs, _truncation_level, _label_gain, _norm, _sigmoid,
    _min_sigmoid_input, _max_sigmoid_input, _sigmoid_table, _sigmoid_bins, _sigmoid_table_idx_factor, _discount
):
    # get max DCG on current query
    inverse_max_dcg = _inverse_max_dcgs[query_id]

    # initialize with zero
    lambdas = np.zeros(cnt)
    hessians = np.zeros(cnt)

    # get sorted indices for scores
    sorted_idx = np.argsort(-score)  # , kind="stable")

    # get best and worst score
    best_score = score[sorted_idx[0]]
    worst_idx = cnt - 1
    if worst_idx > 0 and score[sorted_idx[worst_idx]] == kMinScore:
        worst_idx -= 1

    worst_score = score[sorted_idx[worst_idx]]
    sum_lambdas = 0.0

    # start accmulate lambdas by pairs that contain at least one document above truncation level
    for i in range(0, min(cnt - 1, _truncation_level)):
        if score[sorted_idx[i]] == kMinScore:
            continue
        for j in range(i + 1, cnt):
            if score[sorted_idx[j]] == kMinScore:
                continue
            # skip pairs with the same labels
            if label[sorted_idx[i]] == label[sorted_idx[j]]:
                continue

            if label[sorted_idx[i]] > label[sorted_idx[j]]:
                high_rank = i
                low_rank = j
            else:
                high_rank = j
                low_rank = i

            high = sorted_idx[high_rank]
            high_label = int(label[high])
            high_score = score[high]
            high_label_gain = _label_gain[high_label]
            high_discount = _discount[high_rank]
            low = sorted_idx[low_rank]
            low_label = int(label[low])
            low_score = score[low]
            low_label_gain = _label_gain[low_label]
            low_discount = _discount[low_rank]

            high_w = doc_weights[high]
            low_w = doc_weights[low]

            delta_score = high_score - low_score

            # get dcg gap
            dcg_gap = high_label_gain - low_label_gain
            # get discount of this pair
            paired_discount = np.abs(high_discount - low_discount)
            # get delta NDCG
            delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg
            # regular the delta_pair_NDCG by score distance
            if _norm and best_score != worst_score:
                delta_pair_NDCG /= (0.01 + np.abs(delta_score))

            # calculate lambda for this pair
            p_lambda = _get_sigmoid(
                delta_score, _min_sigmoid_input, _max_sigmoid_input, _sigmoid_table, _sigmoid_bins,
                _sigmoid_table_idx_factor
            )
            p_hessian = p_lambda * (1.0 - p_lambda)
            # update
            p_lambda *= -_sigmoid * delta_pair_NDCG
            p_hessian *= _sigmoid * _sigmoid * delta_pair_NDCG
            lambdas[low] -= p_lambda
            hessians[low] += p_hessian
            lambdas[high] += p_lambda
            hessians[high] += p_hessian
            lambdas[low] -= p_lambda * low_w
            hessians[low] += p_hessian * low_w
            lambdas[high] += p_lambda * high_w
            hessians[high] += p_hessian * high_w
            # lambda is negative, so use minus to accumulate
            # sum_lambdas -= 2 * p_lambda
            sum_lambdas -= (low_w + high_w) * p_lambda

    if _norm and sum_lambdas > 0:
        norm_factor = np.log2(1 + sum_lambdas) / sum_lambdas
        for i in range(0, cnt):
            lambdas[i] = lambdas[i] * norm_factor
            hessians[i] = hessians[i] * norm_factor

    return lambdas, hessians
