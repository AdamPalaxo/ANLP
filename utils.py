import numpy as np


# Prepares data for end to end model
# It finds maximal value for each decomposed part of the label 
# and then join prediction for composed and decomposed label
def prepare_data_for_end2end(y_pred, yd_pred):
    yd_pred = [np.argmax(label, axis=2) for label in yd_pred]
    yd_pred = np.stack(yd_pred, axis=2)
    y_pred = np.dstack((yd_pred, y_pred))

    return y_pred
