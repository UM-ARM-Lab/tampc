from arm_pytorch_utilities import draw
from meta_contact import model

plotter_map = {model.MDNUser: draw.plot_mdn_prediction, model.DeterministicUser: draw.plot_prediction}
