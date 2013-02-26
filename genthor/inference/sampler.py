""" Sampler code."""
# Standard
import os
# Third party
import numpy as np
import pandac.PandaModules as pm
# Project
from genthor.API import SimpleAPI
from genthor.renderer.lightbase import LightBase
import genthor.renderer.renderer as gr
import genthor.tools as tools
#
import pdb


def normlpdf(x, mu, sigma):
    """ Normal log pdf."""
    x = x.ravel()
    mu = mu.ravel()
    n = x.size
    c = -np.log(sigma) - 0.5 * np.log(2. * np.pi)
    y = x - mu
    lp = n * c - np.dot(y.T, y) / (2. * sigma ** 2)
    #lp = n * c - y**2 / (2. * sigma ** 2)
    return lp


def mlmlpdf(x, mu, sigma):
    """ Maximum local mass."""
    x = x.ravel()
    mu = mu.ravel()
    n = x.size
    c = -np.log(sigma) - 0.5 * np.log(2. * np.pi)
    y = x - mu
    #lp = n * c + np.exp(-y ** 2 / (2 * sigma ** 2))
    lp = np.sum(n * c + np.exp(-y ** 2 / (2. * sigma ** 2)))
    return lp


# def mvnlpdf(x, mu, cov):
#     import scipy.linalg as la
#     from scikits.learn.utils import extmath

#     n, k = x.shape
#     ldet = extmath.fast_logdet(cov)
#     c = -0.5 * (ldet + k * np.log(2. * np.pi))
#     y = x - mu
#     icov = la.inv(cov)
#     lp = n * c - 0.5 * np.sum(np.dot(y, icov) * y)
#     return lp

# plt.figure(10)
# plt.clf()
# s = 24
# x = np.linspace(-255, 255, 100)
# #y0 = normlpdf(x, np.zeros_like(x), s)
# y1 = mlmlpdf(x, np.zeros_like(x), s)
# #plt.plot(x, y0)
# plt.plot(x, y1)



class BaseSampler(object):
    """ Basic MCMC sampler."""
    def __init__(self, image, renderer=None, RSO=0):
        # Initialize a random object
        self.RSO = tools.init_RSO(RSO)
        # Initial/null values
        self.image = image
        self.state0 = None
        self.state = None
        self.score = None
        self.states = []
        self.proposals = []
        self._store_dtype = np.dtype([
            ("score", ("f8", 1)), ("proposal_score", ("f8", 1)),
            ("fwdprob", ("f8", 1)), ("bakprob", ("f8", 1)),
            ("flip", ("f8", 1)), ("accepted", ("b", 1))])
        self.store_info = None
        if renderer is None:
            # Create a renderer object
            self.renderer = Renderer()
        else:
            # Use the input renderer object
            self.renderer = renderer
        self.true_features = self.get_features(self.image)
        # Initialize proposal mechanism
        self.init_proposer(RSO=RSO)
        # Initialize the sampler
        self.init_state()
        # Parameters
        self.param = {"llik_sigma": 200.,}

    def init_proposer(self, RSO=0):
        """ Initialize the proposal mechanism."""
        # Create a proposal generation object
        self.proposer = Proposer(RSO=RSO)

    def init_state(self, state=None):
        """ Initialize the sampler's state to 'state'."""

        # Store it (state is the current state, state0 is a record of
        # the initial state
        self.state0 = state
        self.state = state

        if state is None:
            # If state is None, self.score=-inf will force the first
            # proposal to be accepted.
            self.score = -np.inf
        else:
            # Compute and store the initial state's score
            self.score = self.lposterior(state)

    def get_features(self, synth_image):
        """ Basic feature transform for the images. This could be
        replaced with some filtering op, SIFT, etc."""

        # The features of the image (here, it's trivially just the
        # synth_image)
        synth_features = synth_image
        return synth_features

    def loop(self, n_samples, verbose=True):
        """ Runs the sampler for 'n_samples' MCMC steps. The results
        are stored in self.{states,proposals,scores,proposal_scores}."""

        if verbose:
            print("Running %i samples" % n_samples)

        # Allocate
        self.store_info = np.empty(n_samples, dtype=self._store_dtype)

        for ind in xrange(n_samples):
            # Draw a proposal and its fwd/bak probabilities
            proposal, fwdprob, bakprob = self.proposer.draw(self.state)
            # Determine whether to accept or reject the sample
            accepted, proposal_score  = self.accept_reject(
                proposal, fwdprob, bakprob, store_info=self.store_info[ind],
                temperature=0.1)
            # If the proposal is accepted, update the sampler's state
            if accepted:
                self.score = proposal_score
                self.state = proposal
            # Store
            self.states.append(self.state)
            self.proposals.append(proposal)
            # Verbose status report
            if verbose:
                n_accepted = np.sum(self.store_info[:ind + 1]["accepted"])
                total = ind + 1
                percent = 100 * float(n_accepted) / total
                print("# acc, # sampled, # total, acc rate %i/%i/%i (%.2f%%)"
                      % (n_accepted, total, n_samples, percent))

    def accept_reject(self, proposal, fwdprob, bakprob, store_info=None,
                      temperature=1.):
        """ Inputs a proposal and fwd/bak probs, and returns a boolean
        indicating accept or reject, based on the Metropolis-Hastings
        ratio. Also returns proposal and current scores, as well as
        the MH ratio"""

        # Get posterior of proposal
        proposal_score = self.lposterior(proposal, temperature=temperature)
        # Get posterior of current state (just use the one computed last time)
        score = self.lposterior(temperature=temperature)
        # MH ratio
        ratio = proposal_score - score - fwdprob + bakprob
        # random flip
        flip = np.log(self.RSO.rand())
        # Boolean indicating accept vs reject
        accepted = ratio >= flip
        # Extra info useful to store
        if store_info is not None:
            store_info["score"] = score
            store_info["proposal_score"] = proposal_score
            store_info["fwdprob"] = fwdprob
            store_info["bakprob"] = bakprob
            store_info["flip"] = flip
            store_info["accepted"] = accepted
        return accepted, proposal_score

    def lposterior(self, state=None, temperature=1.):
        """ Inputs state, returns unnormalized log posterior
        probability. If state is None, then it returns that current
        state's score, stored in self.score."""

        if state is None:
            # If None was passed in, use the sampler's current score
            lpost = self.score
        else:
            # Compute the state's score:
            # Render an image
            synth_image = self.renderer.render(state)
            # Transform the image into features
            synth_features = self.get_features(synth_image)
            # Compute the likelihood, prior and posterior
            llik = self.llikelihood(synth_features, self.true_features)
            lpr = self.lprior(state)
            lpost = temperature * (llik + lpr)
        return lpost

    def llikelihood(self, f1, f2):
        """ Inputs latent state, returns log prior probability."""

        llik = normlpdf(f1, f2, self.param["llik_sigma"])
        #llik = mlmlpdf(f1, f2, self.param["llik_sigma"])
        return llik

    @staticmethod
    def lprior(state):
        """ Inputs latent state, returns log prior probability."""
        # TODO: make this more interesting
        lpr = 0
        return lpr



class BayesianSamplerWithModel(BayesianSampler):
    """ Main MCMC sampler class when a feedforward model is used to
    drive proposals and provide approximate likelihoods."""

    def __init__(self, image, get_features, get_margins, renderer=None, RSO=0):
        # Make self copies of the get_features and get_margins functions
        self.get_features = get_features
        self.get_margins = get_margins
        # Call parent constructor
        super(type(self), self).__init__(image, renderer=renderer, RSO=RSO)

    def init_proposer(self, RSO=0):
        """ Initialize the proposal mechanism."""

        # Compute the margins
        self.margins = self.get_margins(self.true_features)
        # Create a proposal generation object
        self.proposer = ThorProposer(RSO=RSO)

    def init_state(self, state=None):
        """ Use the feedforward model to initialize, when state is not
        supplied."""

        if state is None:
            # Initialize the sampler's state using the feedforward
            # predictions.
            state = np.argmax(self.margins)

        # initialize with the base's version and the locally-selected
        # state
        super(type(self), self).init_state(state=state)



class Proposer(object):
    """ Draw proposals conditioned on latent states."""

    def __init__(self, RSO=0):
        self.RSO = tools.init_RSO(RSO)
        self.init_state_info()

    def init_state_info(self):
        """ Gets info about the states and the ranges over which they
        can be sampled."""

        # TODO: do this right -- the rngs, below, and copied from
        # build_img_database, which is not a good way to do this...
        model_root = os.path.join(os.environ["GENTHOR"], "processed_models")
        bg_root = os.path.join(os.environ["GENTHOR"], "backgrounds")

        self._state_info = {
            "modelname": os.listdir(model_root),
            "bgname": os.listdir(bg_root),
            "category": (), # TODO: fill in
            "scale": (0.6667, 2.),
            "pos": ((-1.0, 1.0), (-1.0, 1.0)),
            "hpr": ((-180., 180.), (-180., 180.), (-180., 180.)),
            "bgscale": (1.0, 1.0), #(0.5, 2.0)
            "bghp": ((-180., 180.), (0., 0.)),
            }


    def propose(self, state=None):
        """ Draws proposal conditioned on 'state' (if 'state' is None,
        just draw an independent proposal)."""

        # initialize
        proposal = {}

        if state is None:
            # Independent proposals

            # modelname and bgname
            sn = "modelname"
            r = self.RSO.randint(len(self._state_info[sn]))
            proposal[sn] = self._state_info[sn][r]
            sn = "bgname"
            r = self.RSO.randint(len(self._state_info[sn]))
            proposal[sn] = self._state_info[sn][r]

            # the remaining states
            sample = tools.sample
            statenames = ("scale", "pos", "hpr", "bgscale", "bghp")
            for sn in statenames:
                proposal[sn] = sample(self._state_info[sn]).ravel()

        else:
            # Proposal conditioned on state

            # scale factors ([d]iscrete states, [c]ontinuous states)
            # on proximity of proposal from state
            Sd = 0.2
            Sc = 0.05

            # modelname
            sn = "modelname"
            if self.RSO.rand() < Sd:
                r = self.RSO.randint(len(self._state_info[sn]))
                proposal[sn] = self._state_info[sn][r]
            else:
                proposal[sn] = state[sn]

            # bgname
            sn = "bgname"
            if self.RSO.rand() < Sd:
                r = self.RSO.randint(len(self._state_info[sn]))
                proposal[sn] = self._state_info[sn][r]
            else:
                proposal[sn] = state[sn]

            sample = tools.sample
            statenames = ("scale", "pos", "hpr", "bgscale", "bghp")
            for sn in statenames:
                # Full range
                rng0 = np.array(self._state_info[sn])
                # Scaled proximity
                delta = (Sc * np.diff(rng0)).T
                # Proximal range
                rng = rng0.copy().T
                rng[0] = np.max(np.vstack((state[sn] - delta, rng[0])), 0)
                rng[1] = np.min(np.vstack((state[sn] + delta, rng[1])), 0)
                # Draw the proposal
                proposal[sn] = sample(rng.T, RSO=self.RSO).ravel()

        # # TODO: remove
        # proposal["modelname"] = "MB26897"
        # proposal["bgname"] = "MOUNT_33SN.jpg" #"DH209SN.jpg"#"DH-ITALY33SN.jpg"
        # proposal["bghp"] = np.array([115.27785513, 0.]) #27.26587075 #-172.12066367

        return proposal

    def compute_proposal_probs(self, proposal):
        """ Inputs proposed state, returns fwd/bak probabilities."""

        # TODO: Not currently implemented...
        fwdprob = bakprob = 0

        return fwdprob, bakprob


    def draw(self, *args):
        """ Draws a proposal."""

        proposal = self.propose(*args)
        fwdprob, bakprob = self.compute_proposal_probs(proposal)
        return proposal, fwdprob, bakprob



class ThorProposer(Proposer):
    """ Draw proposals conditioned on latent states, using Thor."""

    def __init__(self, RSO=0):
        super(type(self), self).__init__(RSO=RSO)

    def propose(self, state, margins):
        """ Draws proposal conditioned on 'state' and
        'reference_state', presumably using the feedforward model."""

        reference_state = self.RSO.multinomial(margins)
        diff = reference_state - state
        proposal = self.RSO.rand() * diff + state
        return proposal

    def compute_proposal_probs(self, proposal):
        """ Inputs proposed state, returns fwd/bak probabilities."""

        # TODO: Not currently implemented...
        fwdprob = bakprob = 0

        return fwdprob, bakprob



##
if __name__ == "__main__":
    import cPickle as pickle
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import time

    LightBase.destroy_windows()

    def plot_scores(sampler):
        scores = sampler.store_info["score"]
        proposal_scores = sampler.store_info["proposal_score"]
        accepted = sampler.store_info["accepted"]
        accepted_idx = np.flatnonzero(accepted)
        N = scores.size
        # Plot
        plt.figure(20)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(np.vstack((proposal_scores, scores)).T)
        plt.plot(accepted_idx, scores[accepted_idx], "go")
        plt.subplot(2, 1, 2)
        plt.plot(np.zeros(N), "k")
        y = proposal_scores - scores
        d = y[~np.isinf(y)][1:].ptp() / 16.
        plt.plot(y, "b")
        plt.plot(accepted_idx, np.zeros(np.sum(accepted)), "go")
        plt.axis((0, N, y[~np.isinf(y)].min() - d, y[~np.isinf(y)].max() + d))
        #
        plt.draw()

    # # Hand-picked test state
    # state = {
    #     "modelname": "MB26897",
    #     "bgname": "DH-ITALY33SN.jpg",
    #     "category": "cars",
    #     "scale": 1.,
    #     "pos": (0., 0.),
    #     "hpr": (0., 0., 0.),
    #     "bgscale": 1.,
    #     "bghp": (0., 0.),
    #     }

    # Test state from training set
    iscene = 2
    state_path = os.path.join(os.environ["GENTHOR"],
                              "training_data/scene%08i" % iscene)
    with open(state_path + ".pkl", "rb") as fid:
        state = pickle.load(fid)

    # Make a test image
    R = Renderer()
    image = R.render(state)

    # Display the test image and state
    plt.figure(10)
    plt.clf()
    plt.show()
    print("original state:")
    print(state)
    print("")
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(image[::-1], cmap=cm.gray)
    plt.title("Original")
    plt.draw()
    time.sleep(0.1)

    # Number of MCMC samples
    n_samples = 1000
    seed = 1

    # Dumb inference
    sampler0 = BayesianSampler(image, renderer=R, RSO=seed)
    # Initialize the sampler's state
    sampler0.init_state()
    # Run it
    sampler0.loop(n_samples, verbose=True)
    print("")
    print("sampler0 done.")
    print(sampler0.states[-1])
    print("")
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(R.render(sampler0.states[-1])[::-1], cmap=cm.gray)
    plt.title("Last sample")
    plt.draw()
    time.sleep(0.1)

    plot_scores(sampler0)

    plt.figure(10)
    plt.subplot(1, 3, 3)
    plt.axis("off")
    last_st = None
    for i, (st, pr) in enumerate(
        zip(sampler0.states[::10], sampler0.proposals[::10])):
        #if i < 700: continue
        if last_st is not st:
            plt.subplot(1, 3, 2)
            plt.cla()
            plt.axis("off")
            plt.imshow(R.render(st)[::-1], cmap=cm.gray)
            last_st = st
        plt.subplot(1, 3, 3)
        plt.cla()
        plt.axis("off")
        plt.imshow(R.render(pr)[::-1], cmap=cm.gray)
        plt.title("Proposal #%i" % (i * 10))
        plt.draw()
        time.sleep(0.1)

    # # Smart inference
    # sampler1 = BayesianSamplerWithModel(image, renderer=R, RSO=seed)
    # # Initialize the sampler's state
    # sampler1.init_state()
    # # Run it
    # sampler1.loop(n_samples, verbose=True)
    # print("")
    # print("sampler1 done.")
    # print(sampler1.states[-1])
    # print("")
    # plt.subplot(1, 3, 3)
    # plt.imshow(R.render(sampler1.states[-1])[::-1])



# 1. Importance sampling
# 2. interpolation proposals
# 3. Unidimensional Gaussian process -- adaptive/sequential importance sampling
# 4. Multidimensional Gaussian process
