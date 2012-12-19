import os
import json
import tempfile   
import hashlib
import cPickle
import copy

import boto
import Image
import numpy as np

import bson.json_util as json_util
from bson.objectid import ObjectId

import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.autoreload
from tornado.options import define, options

import genthor.datasets as gd

define("port", default=27719, help="run on the given port", type=int)


class App(tornado.web.Application):
    """
        Tornado app which serves the API.
    """
    def __init__(self):
        handlers = [(r"/renderimage", RenderImageHandler)]
        settings = dict(debug=True)
        tornado.web.Application.__init__(self, handlers, **settings)


class BaseHandler(tornado.web.RequestHandler):
    def post(self):
        args = self.request.arguments
        for k in args.keys():
            args[k] = args[k][0]
        args = dict([(str(x),y) for (x,y) in args.items()])  
        callback = args.pop('callback',None)   
        
        postdata = json.loads(self.request.body)

        resp = jsonize(self.get_response(args, postdata))

        if callback:
            self.write(callback + '(')
        self.write(json.dumps(resp, default=json_util.default))   
        if callback:
            self.write(')')

        self.finish()


class RenderImageHandler(BaseHandler):
    def get_response(self, args, postdata):
        return image_response(args, postdata)


def image_response(args, postdata):
    sha1 = hashlib.sha1(json.dumps(postdata)).hexdigest()
    filename = sha1 + '.png'
    conn = boto.connect_s3()
    bucket = conn.get_bucket('dicarlo-renderedimages')
    k = bucket.get_key(filename)
    if not k:
        dataset = gd.GenerativeBase() #pull out into global?   
        imarray = dataset.get_image(postdata['preproc'], postdata['spec'])[::-1]
        im = Image.fromarray(imarray, mode=postdata['preproc']['mode'])
        tmp = tempfile.TemporaryFile()
        im.save(tmp, "png")
        tmp.seek(0)
        k = bucket.new_key(filename)
        k.set_contents_from_string(tmp.read(), headers={'Content-Type' : 'image/png'})
        k.make_public()
    url = "http://dicarlo-renderedimages.s3.amazonaws.com/" + filename
    resp = {"url": url}     
    return resp
        
def jsonize(x):
    try:
        json.dumps(x)
    except:
        return SONify(x)
    else:
        return x


def SONify(arg, memo=None):
    if memo is None:
        memo = {}
    if id(arg) in memo:
        rval = memo[id(arg)]
    if isinstance(arg, ObjectId):
        rval = arg
    elif isinstance(arg, np.floating):
        rval = float(arg)
    elif isinstance(arg, np.integer):
        rval = int(arg)
    elif isinstance(arg, (list, tuple)):
        rval = type(arg)([SONify(ai, memo) for ai in arg])
    elif isinstance(arg, dict):
        rval = dict([(SONify(k, memo), SONify(v, memo))
            for k, v in arg.items()])
    elif isinstance(arg, (basestring, float, int, type(None))):
        rval = arg
    elif isinstance(arg, np.ndarray):
        if arg.ndim == 0:
            rval = SONify(arg.sum())
        else:
            rval = map(SONify, arg) # N.B. memo None
    # -- put this after ndarray because ndarray not hashable
    elif arg in (True, False):
        rval = int(arg)
    else:
        raise TypeError('SONify', arg)
    memo[id(rval)] = rval
    return rval


def main():
    """
        function which starts up the tornado IO loop and the app. 
    """
    tornado.options.parse_command_line()
    ioloop = tornado.ioloop.IOLoop.instance()
    http_server = tornado.httpserver.HTTPServer(App())
    http_server.listen(options.port)
    tornado.autoreload.start()
    ioloop.start()
    

if __name__ == "__main__":
    main()
