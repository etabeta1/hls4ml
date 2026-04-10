from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

in_shape = 124
in_filt = 5
atol = 5e-3

@pytest.fixture(scope='module')
def data_2d():
    return np.random.rand(100, in_shape, in_shape, in_filt)


@pytest.fixture(scope='module')
def keras_model_2d(request):
    model_type = request.param['model_type']
    pads = request.param['padding']
    strides = request.param.get('strides', None)
    model = Sequential()
    if model_type == 'avg':
        model.add(AveragePooling2D(input_shape=(in_shape, in_shape, in_filt), padding=pads, strides=strides))
    elif model_type == 'max':
        model.add(MaxPooling2D(input_shape=(in_shape, in_shape, in_filt), padding=pads, strides=strides))
    model.compile()
    return model, model_type, pads, strides


@pytest.mark.parametrize('backend', ['Quartus', 'Vitis', 'Vivado', 'Catapult', 'oneAPI', 'Bambu'])
@pytest.mark.parametrize(
    'keras_model_2d',
    [
        {'model_type': 'max', 'padding': 'valid', 'strides': None},
    ],
    ids=[
        'model_type-max-padding-valid',
    ],
    indirect=True,
)
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_pool2d(test_case_id, backend, keras_model_2d, data_2d, io_type):
    model, model_type, padding, strides = keras_model_2d

    config = hls4ml.utils.config_from_keras_model(
        model, default_precision='ap_fixed<32,9>', granularity='name', backend=backend
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / test_case_id),
        backend=backend,
        write_top_level_pragmas=False if backend == 'Vivado' else True,
    )
    hls_model.compile()

    if backend in ['Vivado', 'Vitis']:
        hls_model.build(
            synth=True,
            cosim=True,
            validation=True,
            vsynth=True,        
        )

    if backend == 'Bambu':
        hls_model.build(
        cosim=True,
        vsynth=True  # Adds --evaluation which triggers XML generation
        )

    y_keras = model.predict(data_2d)
    y_hls = hls_model.predict(data_2d).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)





'''
        {'model_type': 'max', 'padding': 'same', 'strides': None},
        {'model_type': 'avg', 'padding': 'valid', 'strides': None},
        {'model_type': 'avg', 'padding': 'same', 'strides': None},
        {'model_type': 'max', 'padding': 'same', 'strides': (4, 2)},
        {'model_type': 'avg', 'padding': 'valid', 'strides': (1, 3)},
'''

'''
'model_type-max-padding-same',
        'model_type-avg-padding-valid',
        'model_type-avg-padding-same',
        'model_type-max-padding-same-strides-4-2',
        'model_type-avg-padding-valid-strides-1-3',
'''