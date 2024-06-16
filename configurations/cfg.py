# supported loss dictionary:
import Main.components.loss

supported_loss = {
    'linear_reg': ['mae', 'mse'],
    'logistic_reg': ['binary_crossentropy']
}


losses_addr = {
    'mae': Main.components.loss.mean_absolute_error,
    'mse': Main.components.loss.mean_squared_error,
    'binary_crossentropy': Main.components.loss.binary_crossentropy
}
