import numpy as np
import scipy
from scipy.stats import t
from tqdm.notebook import tqdm
import torch
from torch.nn.functional import cross_entropy
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from config import args


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def meta_test(net, testloader, use_logit=True, is_norm=True, opt=None, r2d2_learner=None, mode=None):
    net = net.eval()
    acc = []

    for idx, data in tqdm(enumerate(testloader)):
        support_xs, support_ys, query_xs, query_ys = data
        support_xs = support_xs.cuda()
        support_ys = support_ys.cuda()
        query_xs = query_xs.cuda()
        query_ys = query_ys.cuda()
        batch_size, _, channel, height, width = support_xs.size()
        support_xs = support_xs.view(-1, channel, height, width)
        query_xs = query_xs.view(-1, channel, height, width)

        with torch.no_grad():
            if use_logit:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

        support_features = support_features.detach()
        query_features = query_features.detach()
        support_ys = support_ys.view(-1)
        query_ys = query_ys.view(-1)

        #  clf = SVC(gamma='auto', C=0.1)
        if args.classifier == 'LR':
            support_features = support_features.cpu().numpy()
            query_features = query_features.cpu().numpy()
            support_ys = support_ys.cpu().numpy()
            query_ys = query_ys.cpu().numpy()

            clf = LogisticRegression(penalty='l2',
                                     random_state=0,
                                     C=1.0,
                                     solver='lbfgs',
                                     max_iter=1000,
                                     multi_class='multinomial')
            clf.fit(support_features, support_ys)
            query_ys_pred = clf.predict(query_features)
        elif args.classifier == 'R2-D2':
            r2d2_learner.fit(support_features, support_ys, mode)
            query_logits = r2d2_learner.predict(query_features, mode)
            if mode == 'train':
                loss = cross_entropy(query_logits, query_ys)
                opt.zero_grad()
                loss.backward()
                opt.step()

                r2d2_learner.losses.append(loss.item())
                r2d2_learner.loss_stat.update(loss.item(), query_logits.shape[0])

            query_ys = query_ys.cpu().numpy()
            query_ys_pred = torch.argmax(query_logits, axis=-1).detach().cpu().numpy()
        else:
            raise NotImplementedError('classifier not supported: {}'.format(args.classifier))

        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    return mean_confidence_interval(acc)


# def R2D2(support, support_ys, query):
#     n = support.shape[0]
#     shuffled_ids = np.arange(n)
#     np.random.shuffle(shuffled_ids)
#
#     support_ys_ohe = np.zeros((support_ys.shape[0], args.n_ways), dtype=np.float64)
#     support_ys_ohe[np.arange(25), support_ys] = 1.
#     support_ys_ohe = support_ys_ohe[shuffled_ids]
#
#     X = np.concatenate((support, np.ones((support.shape[0], 1))), axis=1)
#     X = X[shuffled_ids]
#     query_features = np.concatenate((query, np.ones((query.shape[0], 1))), axis=1)
#
#     W = X.T @ np.linalg.inv(X @ X.T + args.lambd * np.eye(n)) @ support_ys_ohe
#
#     query_logits = query_features @ W
#     pred = np.argmax(query_logits, axis=-1)
#     return pred
