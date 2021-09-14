from collections import namedtuple
import os
from stucco import cfg

NO_CONTACT_ID = -1
# keywords in place of method for storing special run information
RUN_AMBIGUITY = "ambiguity"
CONTACT_ID = "contact_id"
RUN_INFO_KEYWORDS = [RUN_AMBIGUITY, CONTACT_ID]

RunKey = namedtuple('RunKey', ['level', 'seed', 'method', 'params'])
CONTACT_RES_FILE = os.path.join(cfg.DATA_DIR, 'contact_res.pkl')
CONTACT_POINT_CACHE = os.path.join(cfg.DATA_DIR, 'contact_point_history.pkl')
