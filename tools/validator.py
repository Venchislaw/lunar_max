def validate_input(X, y, epochs, lr):
    try:
        assert (X.shape[0] == y.shape[0])
    except:
        raise ValueError("X shape doesn't correspond to y shape  (number of samples doesn't match number of labels)")

    assert (type(epochs) == int)
    assert (type(lr) == float)

    # if everything is fine
    return 0
