# "CLS Next Gen: Accurate Frequency-Frequency Correlation Functions from Center
# Slope Analysis of 2D Correlation Spectra using Artificial Neural Networks" by
# Hoffen and Fayer (https://dx.doi.org/10.1021/acs.jpca.0c04313)
# If you use this please cite that paper as well.


# %%
import numpy as np
import json
from pathlib import Path

p = Path(__file__).parent


def nn(input, network):
    """Apply a neural network to the input data, as defined in the nextcls paper."""
    input_off, input_gain, hidden, hidden_off, output, output_off, gain, offset = network
    input = np.asarray(input)[:, None]
    scaled_input = (input - input_off) * input_gain - 1
    hidden_out = np.tanh(np.dot(hidden, scaled_input) + hidden_off)
    output_out = np.dot(output, hidden_out) + output_off[0]
    output_out = (output_out + 1)/gain + offset
    return output_out


funcs = {}
for para in ['Sigma1', 'Sigma2', 'SigmaInf']:
    with (p / f'NextCLS_{para}.json').open('r') as f:
        data = json.load(f)
    input_off = np.asarray(data['input_off'])
    input_gain = np.asarray(data['input_gain'])
    hidden = np.asarray(data['hidden'])
    hidden_off = np.asarray(data['hidden_off'])
    output = np.asarray(data['out'])
    output_off = np.asarray(data['out_off'])
    gain = np.asarray(data['gain'])
    offset = np.asarray(data['offset'])

    network = (input_off, input_gain, hidden, hidden_off,
               output, output_off, gain, offset)
    funcs[para] = lambda x, network=network: nn(x, network)


Sigma1NN = funcs['Sigma1']
Sigma2NN = funcs['Sigma2']
SigmaInfNN = funcs['SigmaInf']

Sigma1NN([0.5, 0.5, 0.5, 0, 0])

# %%
