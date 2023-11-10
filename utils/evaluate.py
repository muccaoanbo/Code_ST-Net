import numpy as np


def concordance_correlation_coefficient(y_true, y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    # >>> from sklearn.metrics import concordance_correlation_coefficient
    # >>> y_true = [3, -0.5, 2, 7]
    # >>> y_pred = [2.5, 0.0, 2, 8]
    # >>> concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator

def KL(p, t, bm, nbins):
    t_pdf, bins = np.histogram(t, nbins, density=1)
    p_pdf, bins = np.histogram(p, nbins, density=1)
    epsilon = 0.00001
    T = t_pdf / np.sum(t_pdf)
    T = T + epsilon
    P = p_pdf / np.sum(p_pdf)
    P = P + epsilon
    k = 0
    for i in range(len(P)):
        k += P[i] * np.log(P[i] / T[i])
    return k

# def Dice(predicted, target):
#     smooth = 1
#     product = np.multiply(predicted, target)
#     intersection = np.sum(product)
#     coefficient = (2 * intersection + smooth) / (np.sum(predicted) + np)

def naive_roc_auc_score(y_true, y_pred):
    num_same_sign = 0
    num_pairs = 0
    large = 0
    equal = 0
    for a in range(len(y_true)):
        for b in range(len(y_true)):
            if y_true[a] > y_true[b]:
                num_pairs += 1
                if y_pred[a] > y_pred[b]:
                    num_same_sign += 1
                    large += 1
                elif y_pred[a] == y_pred[b]:
                    num_same_sign += .5
                    equal += 1
    return (num_same_sign / num_pairs), num_pairs, large, equal