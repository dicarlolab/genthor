import numpy as np
import pandac.PandaModules as pm
from rendering.lightbase import LightBase

class ImageProcessor(object):

    def __init__(self, thor_params, training_set):
        self.training_set = training_set
        self.thor_params = thor_params
        
    def initialize(self):
        self.train()

    def train(self):
        pass
    
    
class Renderer(object):
    """ Turns states into scenes, renders them and gets the
    images. Perhaps this class should be sub-classed from LightBase."""
    
    def __init__(self):
        """ Prepares graphics context in which to render synthetic
        images."""
    
        self.lbase = LightBase()
        self.output, self.tex = self.lbase.make_texture_buffer(size=(256, 256))

        # Create lights
        # Create camera


    def state2scene(self, state):
        # Create a NodePath to hold the scene
        scene = pm.NodePath("scene")
        # ... make the scene from the state
        # ...
        return scene

    def render(self, state):
        """ Take the state, create a scene, render it and get the
        image."""
        
        # Set up scene
        scene = self.state2scene(state)
        
        # Add to graphics scene
        scene.reparentTo(self.lbase.rootnode)
        # Render it
        self.lbase.render_frame()
        # Remove the scene from the self.lbase.rootnode
        scene.removeNode()
        # Get the image (it's a numpy.ndarray)
        img = self.lbase.get_tex_image(self.tex)

        return img


class BayesianSampler(object):
    pass

    
class BayesianSamplerWithModel(BayesianSampler):
    """ Main MCMC sampler class when a feedforward model is used to
    drive proposals and provide approximate likelihoods."""
    
    def __init__(self, image, get_features, get_margins, seed=0):
        self.rand = np.random.RandomState(seed=seed)
        self.get_features = get_features
        self.get_margins = get_margins
        self.image = image
        self.states = []
        self.proposals = []
        self.initialize_state()
        self.renderer = Renderer()
    
    def initialize_state(self, state=None):
        """ Initialize the sampler's state using the feedforward
        predictions."""
        
        features = self.get_features(self.image)
        self.true_features = features
        self.margins = self.get_margins(self.true_features)
        if state is None:
            self.state = np.argmax(self.margins)

    def propose(self):
        """ Draws a proposal."""
        
        reference_state = self.rand.multinomial(self.margins)
        proposal = self.propose_from_reference(reference_state)
        fwdprob, bakprob = self.compute_proposal_probs(proposal)
        return proposal, fwdprob, bakprob
        
    def propose_from_reference(self, reference_state):
        """ Draws proposal by consulting current self.state and a
        reference state, presumably provided by the feedforward model."""
        
        diff = reference_state - self.state
        proposal = self.rand.rand() * diff + self.state
        return proposal
        
    def compute_proposal_probs(self, proposal):
        """ Inputs proposed state, returns fwd/bak probabilities."""
        
        # not implemented...
        fwdprob = 0
        bakprob = 0
        
        return fwdprob, bakprob
                    
    def loop(self, N):
        """ Inputs 'N' corresponding to a number of MCMC steps, which
        are performed. The results are stored in self.states and
        self.proposals."""
        
        for ind in xrange(N):
            proposal, fwdprob, bakprob = self.propose()
            accepted = self.accept_reject(proposal, fwdprob, bakprob)
            if accepted:
                self.state = proposal
            self.states.append(self.state)
            self.proposals.append(proposal)
             
    def accept_reject(self, proposal, fwdprob, bakprob):
        """ Inputs a proposal and fwd/bak probs, and returns a boolean
        indicating accept or reject. The proposal is used to compute
        an unnormalized posterior, which is compared to the current
        state's unnormalized posterior, which are used to compute a
        Metropolis-Hastings ratio."""
        
        proposal_score = self.lposterior(proposal)
        current_score = self.lposterior(self.state)
        alpha = proposal_score - current_score - fwdprob + bakprob
        r = np.log(self.rand.rand())
        if alpha > r:
            return True
        else:
            return False
        
    def lposterior(self, state):
        """ Inputs latent state, returns unnormalized log posterior
        probability."""

        synth_image = self.renderer.render(state)
        synth_features = self.get_features(synth_image)
        llik = self.llikelihood(synth_features, self.true_features)
        lpr = self.lprior(state)
        lpost = llik + lpr
        return lpost
        
    def llikelihood(self, f1, f2):
        """ Inputs latent state, returns log prior probability.
        (could be made @staticmethod)"""

        llik = np.sum(f1 - f2)
        return llik
        
    def lprior(self, state):
        """ Inputs latent state, returns log prior probability.
        (could be made @staticmethod)"""
        
        # make this more interesting later
        lpr = 0
        return lpr
