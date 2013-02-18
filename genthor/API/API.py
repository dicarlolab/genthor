""" API code."""
# Standard
import os
import requests


def is_up(address):
    """ Checks whether the address points to a valid server."""
    # Check that server is working.
    R = requests.get(address)
    f_up = R.status_code != requests.codes.ok
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
        f_up = is_up(address)
        if not f_up:
            raise APIConnectionError("Address is invalid: %s" % address)
        # Store address.
        self._address = address

    def _parse_response(self, response0):
        """ Parse response."""
        return response0

    def retrieve(self, data):
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

    # Suffix to append to address
    address_suffix = "score"

    def _parse_response(self, response0):
        """ Implement class-specific response parsing."""
        key = "score"
        response = response0[key]
        return response
