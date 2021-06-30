from collections import namedtuple
import os
from tampc import cfg

NO_CONTACT_ID = -1
RunKey = namedtuple('RunKey', ['level', 'seed', 'method', 'params'])
CONTACT_RES_FILE = os.path.join(cfg.DATA_DIR, 'contact_res.pkl')
