var nn_weights = %weights%;
var nn_in = %in%, nn_out = %out%;

function nn_process(acts) {
    var newActs;
    for(var l = 0; l < nn_weights.length; l++) {
        acts.unshift(1);
        newActs = new Array(nn_weights[l].length);
        for(var n = 0; n < nn_weights[l].length; n++) newActs[n] = act(acts, nn_weights[l][n]);
        acts = newActs;
    }
    return acts;
}

function act(acts, w) {
    var sum = 0;
    for(var el = 0; el < acts.length; el++) sum += acts[el] * w[el];
    return 1 / (1 + Math.exp(-sum));
}