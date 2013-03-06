#!/usr/bin/env python
""" Handlers for each API."""
# Standard
import argparse
import os
# External
import requests


def is_connection_ok(address):
    """ Checks whether the address points to a valid server."""
    # Check that server is working.
    try:
        R = requests.get(address)
        f_up = R.status_code != requests.codes.ok
    except requests.ConnectionError:
        f_up = False
    return f_up


class APIConnectionError(Exception):
    """ Exception for invalid addresses."""
    pass


class BaseAPI(object):
    """ Base class for API communication."""
    # Curl headers.
    headers = {"Content-Type": "application/json"}
    # Suffix to append to address.
    address_suffix = ""

    def __init__(self, address=""):
        # Address of server
        self.address = address

    @property
    def address(self):
        """ Getter."""
        return self._address

    @address.setter
    def address(self, address0):
        """ Setter."""
        # Append suffix.
        if self.address_suffix:
            address = os.path.join(address0, self.address_suffix)
        else:
            address = address0
        # Check whether address is working.
        if not is_connection_ok(address):
            raise APIConnectionError("Cannot connect to: %s" % address)
        # Store address.
        self._address = address

    def _parse_response(self, response0):
        """ Parse response."""
        return response0

    def post(self, data):
        """ Send data payload to server and retrieve response."""
        # Post data to self.address
        R = requests.post(self.address, headers=self.headers, data=data)
        # Raise exception for a bad status code.
        R.raise_for_status()
        # Convert to desired response.
        response = self._parse_response(R.json())
        return response


class RendererAPI(BaseAPI):
    """ Handles communication with renderer API."""
    # Suffix to append to address.
    address_suffix = "renderimage"

    def _parse_response(self, response0):
        """ Implement class-specific response parsing."""
        key = "filename"
        response = response0[key]
        return response


class FeaturesAPI(BaseAPI):
    """ Handles communication with features API."""
    # Suffix to append to address.
    address_suffix = "computefeatures"

    def _parse_response(self, response0):
        """ Implement class-specific response parsing."""
        key = "features"
        response = response0[key]
        return response


class PredictionAPI(BaseAPI):
    """ Handles communication with prediction API."""
    # Suffix to append to address
    address_suffix = "predict"

    def _parse_response(self, response0):
        """ Implement class-specific response parsing."""
        key = "prediction"
        response = response0[key]
        return response


class ScoreAPI(BaseAPI):
    """ Handles communication with score API."""
    # Suffix to append to address.
    address_suffix = "score"

    def _parse_response(self, response0):
        """ Implement class-specific response parsing."""
        key = "score"
        response = response0[key]
        return response


class SimpleAPI(object):
    """ Handles communication with all APIs at once."""
    def __init__(self, address=""):
        # Initialize individual API objects.
        self.apis = {"renderer": RendererAPI(address=address),
                     "features": FeaturesAPI(address=address),
                     "prediction": PredictionAPI(address=address),
                     "score": ScoreAPI(address=address)}
        # Assign each API object's post method to an instance method.
        for key, val in self.apis.iteritems():
            setattr(self, key, val.post)

    @property
    def address(self):
        """ Getter."""
        return self._address

    @address.setter
    def address(self, address):
        # Set address for each API object.
        for val in self.apis.itervalue():
            val.address = address
        # Check whether address is working.
        if not is_connection_ok(address):
            raise APIConnectionError("Cannot connect to: %s" % address)
        # Store address.
        self._address = address


def test_api(address, data=None):
    """ Test API interfaces."""
    # Create SimpleAPI.
    S = SimpleAPI(address=address)
    # Handle data.
    if data is None:
        # Example data.
        data = {}
        data["renderer"] = {
            "obj": ["MB26897"], "tx": [0.0], "ty": [0], "texture_mode": [],
            "s": [2.5], "bgscale": 1.0, "rxy": [0], "bgpsi": 0.0, "rxz": [20],
            "ryz": [0], "texture": [], "tz": [-0.33],
            "bgname": "DH-ITALY03SN.jpg", "bgphi": 150.5}
        data["features"] = {
            "filename": "603cfbce676306c7f5ce79fef823a87dbdf71301.pkl"}
        data["prediction"] = {
            "filename": "603cfbce676306c7f5ce79fef823a87dbdf71301.pkl"}
        data["score"] = {
            "im1": {"texture_mode": [], "bgpsi": 0.0, "texture": [], "tz": [0],
                    "obj": ["Air_hostess_pose09"], "tx": [0], "ty": [-0.6],
                    "bgscale": 1.0, "rxy": [90], "s": [0.03], "rxz": [-90],
                    "ryz": [0], "bgname": "MOUNT_21SN.jpg", "bgphi": 30},
            "im2": {"texture_mode": [], "bgpsi": 0.0, "texture": [], "tz": [0],
                    "obj": ["Air_hostess_pose09"], "tx": [0], "ty": [-0.7],
                    "bgscale": 1.0, "rxy": [45], "s": [0.2], "rxz": [-90],
                    "ryz": [0.1], "bgname": "MOUNT_21SN.jpg", "bgphi": 30}}
    # Get responses.
    response = {}
    for key, val in data.iteritems():
        response[key] = getattr(S, key)(S, val)
    # Print
    for key, val in response.iteritems():
        print key, val
        print
    return response


if __name__ == "__main__":
    ## Cmd line interface.
    # parser object.
    parser = argparse.ArgumentParser(description="Determine API instructions.")
    # address argument.
    parser.add_argument("address", help="address")
    # api_type argument.
    parser.add_argument(
        "api_type", choices=["renderer", "features", "prediction", "score"],
        help="API type <- {renderer, features, prediction, score}")
    # payload argument.
    parser.add_argument("payload", nargs="?", default="",
                        help="payload (optional)")
    # Create parser and parse args.
    parsed = parser.parse_args()
    address = parsed.address
    data = {parsed.api_type: parsed.payload}
    # Run test.
    test_api(address, data)
