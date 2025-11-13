import logging
logging.getLogger('astropy.nddata.ccddata').setLevel(logging.WARNING)

from .fitslv0 import * 
from .fitslv1 import *
from .combmaster import *