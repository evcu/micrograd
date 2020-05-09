def top_k_param_dict(param_tuples, k, sort_fn=lambda p: abs(p.data)):
    # The list has (param, index, ..) tuples, so we apply function to the param.
    key_fn = lambda tp: sort_fn(tp[0])
    # We can do top_k(k=n_update), but for now we are sorting it all.
    sorted_list = sorted(param_tuples, key=key_fn, reverse=True)
    return sorted_list[:k]

def rigl_update_neuron(model, update_fraction=0.3):
    total_loss, acc = loss(model, dense_grad=True)
    # backward
    model.zero_grad()
    total_loss.backward()

    for layer in model.layers:
        for neuron in layer.neurons:
            n_weights = len(neuron.w)
            n_update = math.floor(n_weights * update_fraction)
            if n_update == 0:
                # Not updating
                continue
            # Decide connections to grow (pick top gradient magnitude).
            zero_params = [(p, i) for i, p in neuron.zero_ws.items()]
            params = [(p, i) for i, p in neuron.w.items()]
            top_grad_fn = lambda p: abs(p.grad)
            top_k_zero_params = top_k_param_dict(zero_params, n_update, sort_fn=top_grad_fn)
            # Find connections to drop (pick least magnitude).
            least_magnutide_fn = lambda p: -abs(p.data)
            bottom_k_params = top_k_param_dict(params, n_update, sort_fn=least_magnutide_fn)
            for (p, i), (_, i_new) in zip(bottom_k_params, top_k_zero_params):
                # Update weights
                del neuron.w[i]
                p.data = 0.
                neuron.w[i_new] = p
            # Done with zero_params delete them.
            neuron.zero_ws = {}

def rigl_update_layer(model, update_fraction=0.3):
    total_loss, acc = loss(model, dense_grad=True)
    # backward
    model.zero_grad()
    total_loss.backward()

    for layer in model.layers:
        n_weights = 0
        zero_params, params = [], []
        for j, neuron in enumerate(layer.neurons):
            n_weights += len(neuron.w)
            # Decide connections to grow (pick top gradient magnitude).
            zero_params.extend([(p, i, j)  for i, p in neuron.zero_ws.items()])
            params.extend([(p, i, j)  for i, p in neuron.w.items()])
            # Done with zero_params delete them.
            neuron.zero_ws = {}
        n_update = math.floor(n_weights * update_fraction)
        if n_update == 0:
            # Not updating
            continue

        top_grad_fn = lambda p: abs(p.grad)
        top_k_zero_params = top_k_param_dict(zero_params, n_update, sort_fn=top_grad_fn)
        # Find connections to drop (pick least magnitude).
        least_magnutide_fn = lambda p: -abs(p.data)
        bottom_k_params = top_k_param_dict(params, n_update, sort_fn=least_magnutide_fn)
        for (p, i, j), (_, i_new, j_new) in zip(bottom_k_params, top_k_zero_params):
            # Update weights
            del layer.neurons[j].w[i]
            p.data = 0.
            layer.neurons[j_new].w[i_new] = p

def rigl_update_model(model, update_fraction=0.3):
    total_loss, acc = loss(model, dense_grad=True)
    # backward
    model.zero_grad()
    total_loss.backward()
    n_weights = 0
    zero_params, params = [], []
    for k, layer in enumerate(model.layers):
        for j, neuron in enumerate(layer.neurons):
            n_weights += len(neuron.w)
            # Decide connections to grow (pick top gradient magnitude).
            zero_params.extend([(p, i, j, k)  for i, p in neuron.zero_ws.items()])
            params.extend([(p, i, j, k)  for i, p in neuron.w.items()])
            # Done with zero_params delete them.
            neuron.zero_ws = {}
    n_update = math.floor(n_weights * update_fraction)
    if n_update == 0:
        # Not updating
        return
    top_grad_fn = lambda p: abs(p.grad)
    top_k_zero_params = top_k_param_dict(zero_params, n_update, sort_fn=top_grad_fn)
    # Find connections to drop (pick least magnitude).
    least_magnutide_fn = lambda p: -abs(p.data)
    bottom_k_params = top_k_param_dict(params, n_update, sort_fn=least_magnutide_fn)
    for (p, i, j, k), (_, i_new, j_new, k_new) in zip(bottom_k_params, top_k_zero_params):
        # Update weights
        del model.layers[k].neurons[j].w[i]
        p.data = 0.
        model.layers[k_new].neurons[j_new].w[i_new] = p
