import lasagne
import binary_connect


def count_weights(network):
    layers = lasagne.layers.get_all_layers(network)

    result = {}
    result['#zero'] = 0
    result['#one'] = 0
    result['#minusone'] = 0
    total_count = 0

    for layer in layers:
        if not isinstance(layer, binary_connect.DenseLayer):
            continue
        W = layer.Wb.eval()
        for row_id in range(W.shape[0]):
            for column_id in range(W.shape[1]):
                total_count += 1
                if W[row_id][column_id] == 0.:
                    result['#zero'] += 1
                elif W[row_id][column_id] == 1.:
                    result['#one'] += 1
                elif W[row_id][column_id] == -1.:
                    result['#minusone'] += 1
                else:
                    raise Exception('UNK weight value!')
    result['#total'] = result['#zero'] + result['#one'] + result['#minusone']
    return result


def print_weight_counts(network):
    result = count_weights(network)
    zero_weight_count = result['#zero']
    total_count = result['#total']

    print('Weight distribution (number of 0, +1 and -1 weights): #0: {}, #1: {}, #-1: {}'.format(
        result['#zero'], result['#one'], result['#minusone']
    ))
    print('{:.2f}% zero weights: {} out of {}'.format(
        100. * float(zero_weight_count) / total_count,
        zero_weight_count,
        total_count))
