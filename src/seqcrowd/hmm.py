import numpy as np
import util
import os
import copy
import additive
import nltk
import crf
import scipy.special
import sklearn

class HMM:
    """
    Hidden Markov Model
    """

    def __init__(self, n, m):
        """
        fix n, m
        :param n: number of states
        :param m: number of observations
        """
        self.n = n
        self.m = m
        self.t = np.zeros((n, n))
        self.e = np.zeros((n, m))
        self.start = np.asarray([1.0 / n] * n)

    def pr_obs(self, i, list_features, t=None):
        """
        :param i: state
        :param list_features:
        :param t: time, not used here
        :return: probability of observing the features in state i
        """
        res = 1
        for f in list_features:
            res *= self.e[i, f]
        return res

    def decode(self, a, include_crowd_obs=False):
        """
        Viterbi decoding
        :param a: seq of observations, each observation is a list of features
        :return:
        """
        l = len(a)
        if l == 0:
            return []
        # c[t][i] = prob of best path time t, at state i
        c = np.zeros((l, self.n))
        c[0] = np.copy(self.start)   # * self.e[:, a[0]]
        # print self.n, c.shape
        for i in range(self.n):
            c[0][i] *= self.pr_obs(i, a[0])

        # b[t][i] = backpointer
        b = np.zeros((l, self.n))

        for t in range(1, l, 1):  # time
            ob = a[t]
            for i in range(self.n):  # current state
                for j in range(self.n):  # previous state
                    # todo: change to log scale
                    p = c[t - 1][j] * self.t[j, i] * self.pr_obs(i, ob)
                    if include_crowd_obs:
                        p *= self.pr_crowd_labs(t, i, self.current_list_cl)
                    # print t, i, j, p
                    if p > c[t][i]:
                        c[t][i] = p
                        b[t][i] = j

        res = np.zeros((l,))

        # trace
        p = 0
        for i in range(self.n):
            if c[l - 1][i] > p:
                p = c[l - 1][i]
                res[l - 1] = i

        for t in range(l - 2, -1, -1):
            res[t] = b[t + 1, int(res[t + 1])]

        # print c
        # print b
        return res

    def learn(self, sentences, smooth=0.001):
        """
        learn parameters from labeled data
        :param sentences: list of sentence, which is list of instance
        :return:
        """
        # counting
        self.t = smooth * np.ones((self.n, self.n))
        self.e = smooth * np.ones((self.n, self.m))
        self.start = smooth * np.ones((self.n,))

        for sentence in sentences:
            if len(sentence) > 0:
                i = sentence[0]
                self.start[i.label] += 1
                prev = -1  # previous state
                for i in sentence:
                    state = i.label
                    if prev != -1:
                        self.t[prev][state] += 1
                    for f in i.features:
                        self.e[state][f] += 1
                    prev = state

        # save count for e
        self.count_e = copy.deepcopy(self.e)

        # normalizing
        self.start = self.start * 1.0 / np.sum(self.start)
        for i in range(self.n):
            self.t[i] = self.t[i] * 1.0 / np.sum(self.t[i])
            self.e[i] = self.e[i] * 1.0 / np.sum(self.e[i])

    def decode_all(self, sentences):
        self.res = []
        for s in sentences:
            mls = self.decode(util.get_obs(s))
            self.res.append(mls)

##########################################################################
##########################################################################
##########################################################################
##########################################################################


class WorkerModel:
    """
    model of workers
    """
    def __init__(self, n_workers, n_class, smooth = 0.001, ne = 9, rep = 'cv'):
        """

        :param n_workers:
        :param n_class:
        :param smooth:
        :param ne:
        :param rep: representation. cv2 = confusion vec of accuracy in two cases: non-entity/ entity
        """
        self.n_workers = n_workers
        self.n = n_class
        self.smooth = smooth
        self.ne = ne
        self.rep = rep


    def learn_from_pos(self, data, pos):
        """

        :param data: crowd_data
        :param pos: sentence posterior
        :return:
        """
        count = self.smooth * np.ones( (self.n_workers, self.n, self.n))
        for i, sentence in enumerate(data.sentences):
            for j in range(len(sentence)):
                for l, w in data.get_lw(i, j):
                    for k in range(self.n):  # 'true' label = k
                        count[w][k][l] += pos[i][j][k]
        self.learn_from_count(count)


    def learn_from_count(self, count):
        """

        :return:
        """
        #save the count for debug
        self.count = count

        if self.rep == 'cv2':
            ne = self.ne
            self.cv = np.zeros((self.n_workers, 2))
            for w in range(self.n_workers):
                self.cv[w][0] = count[w][ne][ne] * 1.0 / np.sum(count[w][ne]) # accuracy for ne class

                cc = self.smooth; cw = self.smooth # count for correct and wrong for non ne classes
                for i in range(self.n):
                    if i != ne:
                        cc += count[w][i][i]
                        cw += np.sum(count[w][i]) - count[w][i][i]
                self.cv[w][1] = cc * 1.0 / (cc + cw)
        elif self.rep == 'cv':
            self.cv = np.zeros((self.n_workers, self.n))
            for w in range(self.n_workers):
                for i in range(self.n):
                    self.cv[w][i] = count[w][i][i] * 1.0 / np.sum(count[w][i]) # accuracy for ne class

        elif self.rep == 'cm_sage':
            self.cm = np.zeros((self.n_workers, self.n, self.n))
            # background dist
            m = np.sum(count, axis=0)
            for i in range(self.n): m[i] = m[i] * 1.0 / np.sum(m[i])
            m = np.log(m)

            for w in range(self.n_workers):
                for i in range(self.n):
                    temp = additive.estimate(count[w][i], m[i])
                    temp = np.reshape(temp, (self.n,) )
                    self.cm[w][i] = np.exp(temp + m[i])
                    self.cm[w][i] = self.cm[w][i] * 1.0 / np.sum(self.cm[w][i])

        else:
            self.cm = np.zeros((self.n_workers, self.n, self.n))
            for w in range(self.n_workers):
                for k in range(self.n):
                    self.cm[w][k] = count[w][k] * 1.0 / np.sum(count[w][k])




    def get_prob(self, w, true_lab, lab):
        """

        :param w: worker
        :param true_lab:
        :param lab:
        :return: probability of response lab given true label
        """
        #return 1.0
        if self.rep == 'cv2':
            if self.ne == true_lab:
                if true_lab == lab:
                    return self.cv[w][0]
                else:
                    return 1 - self.cv[w][0]
            else:
                if true_lab == lab:
                    return self.cv[w][1]
                else:
                    return 1 - self.cv[w][1]

        elif self.rep == 'cv':
            if true_lab == lab:
                return self.cv[w][true_lab]
            else:
                return 1 - self.cv[w][true_lab]
        elif self.rep == 'cm_sage':
            return self.cm[w][true_lab][lab]
        else:
            return self.cm[w][true_lab][lab]



class HMM_crowd(HMM):

    def __init__(self, n, m, data, features, labels, n_workers=47, init_w=0.9, smooth=0.001, smooth_w=10, ne = 9, vb = None):
        """
        :param data: util.crowd_data with crowd label
        :return:
        """
        HMM.__init__(self, n, m)
        self.data = data
        self.smooth = smooth
        self.n_workers = n_workers
        self.ep = 1e-300
        self.features = features
        self.labels = labels
        self.init_w = init_w
        self.ne = ne

        self.wca = np.zeros((n, n_workers))
        self.ne = ne
        self.smooth_w = smooth_w
        self.n_sens = len(data.sentences)
        self.vb = vb

    def pr_crowd_labs(self, t, i, list_cl):
        """
        :param t: time
        :param i: the state
        :param list_cl: list of util.crowddlab
        :return: probability of observing crowd labels at state i
        """
        res = 1
        for cl in list_cl:
            wid = cl.wid
            sen = cl.sen
            lab = sen[t]                        # crowd label
            res *= self.wm.get_prob(wid, i, lab)

        return res

    def inference(self, sentence, list_cl, return_ab=False):
        T = len(sentence)                   # number of timesteps
        alpha = np.zeros((T, self.n))  # T * states
        beta = np.zeros((T, self.n))

        # alpha (forward):
        for i in range(self.n):
            alpha[0][i] = self.pr_obs(
                i, sentence[0].features) * self.pr_crowd_labs(0, i, list_cl) * self.start[i]

        for t in range(1, T, 1):
            ins = sentence[t]
            for i in range(self.n):             # current state
                alpha[t][i] = 0
                for j in range(self.n):         # previous state
                    alpha[t][i] += self.pr_obs(i, ins.features) * self.t[j][i] * alpha[t - 1][j] \
                        * self.pr_crowd_labs(t, i, list_cl)

        # beta (backward):
        for i in range(self.n):
            beta[T - 1][i] = self.pr_obs(i, sentence[T - 1].features) * \
                self.pr_crowd_labs(T - 1, i, list_cl)

        for t in range(T - 2, -1, -1):
            ins = sentence[t + 1]
            for i in range(self.n):             # current state
                beta[t][i] = 0
                for j in range(self.n):         # next state
                    beta[t][i] += self.pr_obs(j, ins.features) * self.t[i][j] * beta[t + 1][j] \
                        * self.pr_crowd_labs(t + 1, j, list_cl)#\
                        #* (self.start[i] if t == 0 else 1)

        if return_ab:
            return (alpha, beta)

        sen_posterior = []
        # update counts
        p = np.zeros((self.n,))

        for t in range(T):
            for i in range(self.n):
                p[i] = self.ep + alpha[t][i] * beta[t][i]
            p = p * 1.0 / np.sum(p)  # normalilze

            #save the posterior
            sen_posterior.append(p.copy())

            if t == 0:  # update start counts
                self.count_start += p

            # update emission counts
            ins = sentence[t]
            for i in range(self.n):
                for f in ins.features:
                    self.count_e[i][f] += p[i]

            # update crowd params counts
            for i in range(self.n):                                 # state
                for cl in list_cl:
                    wid = cl.wid
                    # worker ans
                    lab = cl.sen[t]
                    self.count_wa[wid][i][lab] += p[i]



        trans_pos = []
        # update transition counts
        for t in range(T - 1):
            p = np.zeros((self.n, self.n))
            ins = sentence[t+1]
            for i in range(self.n):         # state at time t
                for j in range(self.n):     # state at time t+1
                    p[i][j] = self.ep + alpha[t][i] * self.t[i][j] * self.pr_obs(j, ins.features) \
                        * self.pr_crowd_labs(t + 1, j, list_cl) * beta[t + 1][j]

            # update transition counts
            p = p * 1.0 / np.sum(p)
            for i in range(self.n):
                #p[i] = p[i] * 1.0 / np.sum(p[i])
                self.count_t[i] += p[i]

            trans_pos.append(p.copy())

        # log likelihood
        ll = np.log( np.sum(alpha[t-1]) )

        return (sen_posterior, trans_pos, ll)

    def e_step(self):
        """
        do alpha-beta passes
        :return:
        """
        # setup counting
        self.count_t = self.smooth * np.ones((self.n, self.n))
        self.count_e = self.smooth * np.ones((self.n, self.m))
        self.count_start = self.smooth * np.ones((self.n,))
        self.count_wa = self.smooth * np.ones( (self.n_workers, self.n, self.n) )

        self.sen_posterior = []
        self.trans_posterior = []
        sum_ll = 0
        for i, sentence in enumerate(self.data.sentences):
            if len(sentence) > 0:
                sen_pos, trans_pos, ll = self.inference(sentence, self.data.crowdlabs[i])
                sum_ll += ll
            else:
                sen_pos, trans_pos = ([], [])

            self.sen_posterior.append (sen_pos)
            self.trans_posterior.append(trans_pos)

        # save sum of log-likelihood
        self.sum_ll = sum_ll

    def m_step(self):
        if self.vb != None:
            self.m_step_vb()
            return
        # normalize all the counts
        self.start = self.count_start * 1.0 / np.sum(self.count_start)
        #self.prior = self.count_prior * 1.0 / np.sum(self.count_prior)

        for i in range(self.n):
            self.t[i] = self.count_t[i] * 1.0 / np.sum(self.count_t[i])
            self.e[i] = self.count_e[i] * 1.0 / np.sum(self.count_e[i])

        self.wm.learn_from_count(self.count_wa)

    def m_step_vb(self):
        """
        use Variational Bayes
        """
        self.start = self.count_start * 1.0 / np.sum(self.count_start)
        f = lambda x: np.exp( scipy.special.digamma(x))

        for i in range(self.n):
            self.count_t[i] = self.count_t[i] - self.smooth + self.vb[0]
            self.count_e[i] = self.count_e[i] - self.smooth + self.vb[1]

            self.t[i] = f(self.count_t[i] * 1.0) / f(np.sum(self.count_t[i]))
            self.e[i] = f(self.count_e[i] * 1.0) / f(np.sum(self.count_e[i]))

        self.wm.learn_from_count(self.count_wa)

    def init_te_from_pos(self, pos):
        """
        init transition and emission from posterior
        """
        self.t = self.smooth * np.ones((self.n, self.n))
        self.e = self.smooth * np.ones((self.n, self.m))
        self.start = self.smooth * np.ones((self.n,))

        for sentence, p in zip(self.data.sentences, pos):
            if len(sentence) > 0:
                self.start += p[0]
                for t, ins in enumerate(sentence):
                    for i in range(self.n): #current state
                        for f in ins.features:
                            self.e[i][f] += p[t][i]
                        if t > 0:
                            for j in range(self.n): #previous state
                                self.t[j][i] += p[t-1][j] * p[t][i]



        # normalizing
        self.start = self.start * 1.0 / np.sum(self.start)
        for i in range(self.n):
            self.t[i] = self.t[i] * 1.0 / np.sum(self.t[i])
            self.e[i] = self.e[i] * 1.0 / np.sum(self.e[i])


    def init(self, init_type='dw', sen_a=1, sen_b=1, spe_a=1, spe_b=1, wm_rep =
            'cv', save_count_e = False, dw_em = 5, wm_smooth = 0.001):
        """

        :param init_type:

        :param sen_a:  :param sen_b: :param spe_a: :param spe_b: priors for sen, spe
        expect MV to over-estimate worker
        :return:
        """
        if init_type == 'dw':
            d = dw(self.n, self.m, self.data, self.features, self.labels,
                           self.n_workers, self.init_w, self.smooth)
            d.init()
            d.em(dw_em)
            d.mls()
            self.d = d

            h = HMM(self.n, self.m)
            sen = copy.deepcopy(self.data.sentences)
            util.make_sen(sen, d.res)
            h.learn(sen, smooth = self.smooth)

            self.wm = WorkerModel(n_workers = self.n_workers, n_class = self.n,
                    rep=wm_rep, ne = self.ne, smooth = wm_smooth)
            self.wm.learn_from_pos(self.data, d.pos)

            self.start = h.start
            for s in range(self.n):
                for s2 in range(self.n):
                    self.t[s][s2] = h.t[s][s2]
                    # for s in range(self.n):
                for o in range(self.m):
                    self.e[s][o] = h.e[s][o]

            #save the count of e for sage
            if save_count_e:
                self.count_e = h.count_e

            self.h = h

        else:
            # init params (uniform)
            for i in range(self.n):
                self.start[i] = 1.0 / self.n
                self.wa = [0.9] * self.n_workers
            for s in range(self.n):
                for s2 in range(self.n):
                    self.t[s][s2] = 1.0 / self.n

            for s in range(self.n):
                for o in range(self.m):
                    self.e[s][o] = 1.0 / self.m

            for w in range(self.n_workers):
                #self.wsen[i] = 0.9
                #self.wspe[i] = 0.6
                for i in range(self.n):
                    self.wca[i, w] = 0.8


    def init2(self):
        """
        init
        """
        pos = []
        self.prior = np.zeros( (self.n,) )

        for i, sentence in enumerate(self.data.sentences):
            pos.append( self.smooth * np.ones((len(sentence), self.n)) )
            for j in range(len(sentence)):
                for l in self.data.get_labs(i, j): # labels for sen i, pos j
                    pos[i][j][l] += 1
                pos[i][j] = pos[i][j] * 1.0 / np.sum(pos[i][j])
                self.prior += pos[i][j]

        self.prior = self.prior * 1.0 / np.sum(self.prior)



    def learn(self, num=4):
        """
        learn by EM
        :return:
        """
        self.init()
        self.em(num)

    def em(self, num=4):
        # run EM
        for it in range(num):
            print "HMM crowd, iteration", it
            self.e_step()
            self.m_step()

    def mls(self):
        """
        compute the most likely states seq for all sentences
        :return:
        """
        self.res = []
        for s, sentence in enumerate(self.data.sentences):
            if len(sentence) > 0:
                self.current_list_cl = self.data.crowdlabs[s]
                ml_states = self.decode(util.get_obs(
                    sentence), include_crowd_obs=True)
                self.res.append(ml_states)
            else:
                self.res.append([])

    def marginal_decode(self, th):
        """
        decode by marginal prob
        """
        self.res = []
        for i in range(len(self.data.sentences)):
            temp = []
            for j in range(len(self.sen_posterior[i])):
                temp.append ( np.argmax(self.sen_posterior[i][j]) )
            self.res.append(temp)


    def decode_sen_no(self, s):
        self.current_list_cl = self.data.crowdlabs[s]
        sentence = self.data.sentences[s]

        ml_states = self.decode(util.get_obs(
            sentence), include_crowd_obs=True)

        return ml_states


    def threshold(self, thresh = 0.9):
        self.flag = np.zeros((self.n_sens,), dtype=bool)
        for i, r in enumerate(self.res):
            for j, l in enumerate(r):
                if self.posterior[i][j][int(l)] < thresh:
                    self.flag[i] = True

def pos_decode(pos, th, ne = 9):
        """
        decode by posterior:
        res = argmax if pro > th
        else res = ne
        """
        res = []
        for i in range(len(pos)):
            temp = []
            for j in range(len(pos[i])):
                #p = copy.copy(pos[i][j])
                #p[ne] = -1
                k = np.argmax(pos[i][j])
                if pos[i][j][k] > th:
                    temp.append(k)
                else:
                    temp.append(ne)
            res.append(temp)
        return res

##########################################################################
##########################################################################


class HMM_sage(HMM_crowd):

    def __init__(self, n, m, data, features, labels, n_workers=47, init_w=0.9, smooth=0.001, smooth_w=10):
        HMM_crowd.__init__(self, n, m, data, features, labels,
                           n_workers, init_w, smooth)
        HMM.eta = np.zeros((self.m, self.n))

    def init(self, init_type = 'dw', wm_rep = 'cm'):
        HMM_crowd.init(self, init_type=init_type, wm_rep=wm_rep, save_count_e = True)
        self.estimate_sage()

    def estimate_sage(self, mult = 2.0):
        # dont do sage for non-entity
        self.count_e[self.ne, :] = np.zeros((self.m,))
        eq_m = np.sum(self.count_e, axis=0) / np.sum(self.count_e)
        #eq_m = 1.0 / self.m * np.ones((self.m))
        eq_m = np.log(eq_m)

        eta = additive.estimate(mult*self.count_e.T, eq_m)
        for i in range(self.n):
            if i != self.ne:
                self.e[i] = np.exp(eta[:, i] + eq_m) * 1.0 / \
                    np.sum(np.exp(eta[:, i] + eq_m))

        # save eq_m and eta
        self.eq_m = eq_m
        self.eta = eta

    def m_step(self):
        HMM_crowd.m_step(self)
        self.estimate_sage()




class dw(HMM_crowd):
    """
    """
    def __init__(self, n, m, data, features, labels, n_workers=47, init_w=0.9, smooth=0.001, smooth_w=10):
        """
        n: number of states
        :param data: util.crowd_data with crowd label
        :return:
        """
        HMM_crowd.__init__(self, n, m, data, features, labels,
                           n_workers, init_w, smooth)


    def init(self):
        self.pos = []
        self.prior = np.zeros( (self.n,) )

        for i, sentence in enumerate(self.data.sentences):
            self.pos.append( self.smooth * np.ones((len(sentence), self.n)) )
            for j in range(len(sentence)):
                for l in self.data.get_labs(i, j): # labels for sen i, pos j
                    self.pos[i][j][l] += 1
                self.pos[i][j] = self.pos[i][j] * 1.0 / np.sum(self.pos[i][j])
                self.prior += self.pos[i][j]

        self.prior = self.prior * 1.0 / np.sum(self.prior)


    def e_step(self):
       for i, sentence in enumerate(self.data.sentences):
            self.pos[i] = np.ones( (len(sentence), self.n) )
            for j in range(len(sentence)):
                self.pos[i][j] = self.prior.copy()
                for l, w in self.data.get_lw(i, j): # labels for sen i, pos j
                    self.pos[i][j] *= self.wa[w][:,l]
                self.pos[i][j] = self.pos[i][j] * 1.0 / np.sum(self.pos[i][j])


    def m_step(self):
        count = self.smooth * np.ones ( (self.n_workers, self.n, self.n) )
        count_prior = self.smooth * np.ones_like(self.prior)
        #get-another-label heuristic: 0.9 to diagonal, uniform to elsewhere
        for w in range(self.n_workers):
            for i in range(self.n):
                for j in range(self.n):
                    count[w][i][j] = 0.9 if i == j else 0.1 / (self.n-1)

        for i, sentence in enumerate(self.data.sentences):
            for j in range(len(sentence)):
                count_prior += self.pos[i][j]
                for l, w in self.data.get_lw(i,j):
                    for k in range(self.n): # 'true' label = k
                        count[w][k][l] += self.pos[i][j][k]

        self.prior = count_prior * 1.0 / np.sum(count_prior)

        self.wa = np.zeros( (self.n_workers, self.n, self.n) )
        for w in range(self.n_workers):
            for k in range(self.n):
                self.wa[w][k] = count[w][k] * 1.0 / np.sum(count[w][k])


    def em(self, iterations = 3):
        self.init()
        self.m_step()

        for it in range(iterations):
            print "dw iteration: ", it
            self.e_step()
            self.m_step()

    def mls(self):
        self.res = []
        for i, sentence in enumerate(self.data.sentences):
            self.res.append([0] * len(sentence))
            for j in range(len(sentence)):
                self.res[i][j] = np.argmax(self.pos[i][j])

