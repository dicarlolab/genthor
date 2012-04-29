import numpy as np
import os
import pandac.PandaModules as pm
from genthor.renderer.lightbase import LightBase
import genthor.renderer.renderer as gr
import genthor.tools as tools
import pdb

class ImageProcessor(object):

    def __init__(self, thor_params, training_set):
        self.training_set = training_set
        self.thor_params = thor_params
        
    def initialize(self):
        self.train()

    def train(self):
        pass
    
    
class Renderer(object):
    """ Turns states into scenes, renders them and gets the images.

    Currently this is used for the within-inference image
    synthesizing, though it should probably be made copasetic with
    ImgRendererResize.

    Also, this class should probably be sub-classed from LightBase.
    """
    
    def __init__(self, size=(256, 256), lbase=None, output=None):
        """ Prepares graphics context in which to render synthetic
        images."""

        if lbase is None:
            # Create the renderer
            window_type = "texture"
            self.lbase, self.output = gr.setup_renderer(window_type, size=size)
        else:
            # Use the supplied lbase instance
            self.lbase = lbase
            if output is None:
                # Default to last output in lbase.output_list
                self.output = self.lbase.output_list[-1]
        # Get the RTT target
        self.tex = self.output.getTexture()

    def _state2scene(self, state):
        """ Input a state and return a scene node."""

        # TODO: get rid of this hack, and connect to Dan's fancy system
        model_root = "../processed_models" #"~/work/genthor/processed_models"
        bg_root = "../backgrounds" #"~/work/genthor/backgrounds"

        # TODO: Just assume it is a dict for now, but fix soon
        modelpth = os.path.join(model_root, state["modelname"],
                                state["modelname"] + ".bam")
        bgpth = os.path.join(bg_root, state["bgname"])
        scale = state["scale"]
        pos = state["pos"]
        hpr = state["hpr"]
        bgscale = state["bgscale"]
        bghp = state["bghp"]

        # Make the scene and attach it to a new NodePath
        scene = pm.NodePath("scene")
        gr.construct_scene(self.lbase, modelpth, bgpth,
                           scale, pos, hpr, bgscale, bghp, scene=scene)
        return scene

    def render(self, state):
        """ Take the state, create a scene, render it and get the
        image."""
        
        # Set up scene and attach it to the renderer's scenegraph
        scene = self._state2scene(state)
        scene.reparentTo(self.lbase.rootnode)
        # Render the scene
        self.lbase.render_frame()
        # Remove/destroy the scene
        scene.removeNode()
        # Get the image (it's a numpy.ndarray)
        img = self.lbase.get_tex_image(self.tex)
        return img

    def __del__(self):
        self.lbase.destroy()






class BayesianSampler(object):
    """ Basic MCMC sampler for visual inference."""
    
    def __init__(self, image, rand=0):
        # Initialize a random object
        self.rand = tools.init_rand(rand)
        # Initial/null values
        self.image = image
        self.state0 = None
        self.state = None
        self.score = None
        self.states = []
        self.proposals = []
        self.scores = []
        self.proposal_scores = []
        self.accepteds = []
        # Create a renderer object
        self.renderer = Renderer()
        self.true_features = self.get_features(self.image)
        # Initialize proposal mechanism
        self.init_proposer(rand=rand)
        # Initialize the sampler
        self.init_state()

    def init_proposer(self, rand=0):
        """ Initialize the proposal mechanism."""
        # Create a proposal generation object
        self.proposer = Proposer(rand=rand)

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

        for ind in xrange(n_samples):
            # Draw a proposal and its fwd/bak probabilities
            proposal, fwdprob, bakprob = self.proposer.draw(self.state)
            # Determine whether to accept or reject the sample
            accepted, proposal_score, score, ratio = self.accept_reject(
                proposal, fwdprob, bakprob)
            # If the proposal is accepted, update the sampler's state
            if accepted:
                self.score = proposal_score
                self.state = proposal
            # Store
            self.states.append(self.state)
            self.proposals.append(proposal)
            self.scores.append(self.score)
            self.proposal_scores.append(proposal_score)
            self.accepteds.append(accepted)
            # Verbose status report
            if verbose:
                n_accepted = sum(self.accepteds)
                total = ind + 1
                percent = float(n_accepted) / total
                print("# acc, # sampled, # total, acc rate %i/%i/%i (%.2f%%)"
                      % (n_accepted, total, n_samples, percent))
             
    def accept_reject(self, proposal, fwdprob, bakprob):
        """ Inputs a proposal and fwd/bak probs, and returns a boolean
        indicating accept or reject, based on the Metropolis-Hastings
        ratio. Also returns proposal and current scores, as well as
        the MH ratio"""

        # Get posterior of proposal
        proposal_score = self.lposterior(proposal)
        # Get posterior of current state (just use the one computed last time)
        score = self.lposterior() #self.lposterior(self.state)
        # MH ratio
        ratio = proposal_score - score - fwdprob + bakprob
        # random flip
        r = np.log(self.rand.rand())
        # Boolean indicating accept vs reject
        accepted = ratio >= r
        return accepted, proposal_score, score, ratio
        
    def lposterior(self, state=None):
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
            lpost = llik + lpr
        return lpost

    @staticmethod
    def llikelihood(f1, f2):
        """ Inputs latent state, returns log prior probability."""

        llik = np.sum(f1 - f2)
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
    
    def __init__(self, image, get_features, get_margins, rand=0):
        # Make self copies of the get_features and get_margins functions
        self.get_features = get_features
        self.get_margins = get_margins
        # Call parent constructor
        super(type(self), self).__init__(image, rand=rand)
    
    def init_proposer(self, rand=0):
        """ Initialize the proposal mechanism."""

        # Compute the margins
        self.margins = self.get_margins(self.true_features)
        # Create a proposal generation object
        self.proposer = ThorProposer(rand=rand)

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
    
    def __init__(self, rand=0):
        self.rand = tools.init_rand(rand)
        self.init_state_info()

    def init_state_info(self):
        """ Gets info about the states and the ranges over which they
        can be sampled."""

        # TODO: do this right -- the rngs, below, and copied from
        # build_img_database, which is not a good way to do this...
        model_root = "../processed_models" #"~/work/genthor/processed_models"
        bg_root = "../backgrounds" #"~/work/genthor/backgrounds"

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
            r = self.rand.randint(len(self._state_info[sn]))
            proposal[sn] = self._state_info[sn][r]
            sn = "bgname"
            r = self.rand.randint(len(self._state_info[sn]))
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
            Sc = 0.1

            # modelname
            sn = "modelname"
            if self.rand.rand() < Sd:
                r = self.rand.randint(len(self._state_info[sn]))
                proposal[sn] = self._state_info[sn][r]
            else:
                proposal[sn] = state[sn]

            # bgname
            sn = "bgname"
            if self.rand.rand() < Sd:
                r = self.rand.randint(len(self._state_info[sn]))
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
                proposal[sn] = sample(rng.T).ravel()
        
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
    
    def __init__(self, rand=0):
        super(type(self), self).__init__(rand=rand)
        
    def propose(self, state, margins):
        """ Draws proposal conditioned on 'state' and
        'reference_state', presumably using the feedforward model."""

        reference_state = self.rand.multinomial(margins)
        diff = reference_state - state
        proposal = self.rand.rand() * diff + state
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
    import time

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
    iscene = 0
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
    plt.imshow(image[::-1])
    plt.draw()
    time.sleep(0.1)
    
    # Number of MCMC samples
    n_samples = 1000
    seed = 1

    # Dumb inference
    sampler0 = BayesianSampler(image, rand=seed)
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
    plt.imshow(R.render(sampler0.states[-1])[::-1])
    plt.draw()
    time.sleep(0.1)


    plt.subplot(1, 3, 3)
    plt.axis("off")
    for st in sampler0.states[::10]:
        plt.imshow(R.render(st)[::-1])
        plt.draw()
        time.sleep(0.1)

    # # Smart inference
    # sampler1 = BayesianSamplerWithModel(image, rand=seed)
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
